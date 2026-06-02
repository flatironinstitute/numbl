/**
 * Single-file JS codegen for mtoc2 IR. Mirrors `emit.ts` (the C path)
 * intentionally — same control flow, same variable layout, same call
 * shape — differing only in language and the I/O primitive (`$write(s)`
 * vs `printf(...)`).
 *
 * Pass-by-value protocol: owned-typed RHSes (Var / MemberLoad /
 * HandleCaptureLoad) at Assign, user-function Call args, and
 * MultiAssignCall args route through `emitOwnedRhsJs`, which wraps
 * the expression in `mtoc2_deep_clone(...)`. JS GC handles lifetime,
 * but value-semantics still need an explicit copy — otherwise
 * `b = a; b.f = rhs` would mutate `a`. Owned-producing expressions
 * (TensorBuild, StructLit, HandleLit, fresh-allocating helpers) are
 * left alone — they already yield a fresh value.
 *
 * Builtin dispatch routes through `builtin.emitJs`. If a needed
 * builtin has no `emitJs` hook yet, `requireEmitJs` throws a clear
 * "no emitJs hook" error.
 *
 * The emitted module has the shape:
 *   ```js
 *   <inlined runtime snippet bodies>
 *   function <user-fn-spec-1>(...) { ... }
 *   function <user-fn-spec-2>(...) { ... }
 *   function run($h) {
 *     globalThis.$write = $h.write;
 *     let v1, v2, ...;          // pre-declared locals
 *     ... top-level stmts ...
 *   }
 *   return run;
 *   ```
 *
 * The CLI / engine evals the source via `new Function(source)(...)`
 * which returns `run`, then invokes `run(ctx.helpers)`.
 */

import type {
  IRExpr,
  IRStmt,
  IRProgram,
  IRFunc,
  Assign,
  MultiAssignCall,
} from "../lowering/ir.js";
import { UnsupportedConstruct } from "../lowering/errors.js";
import {
  isColVecTy,
  isOwned,
  isRowVecTy,
  isScalar,
} from "../lowering/types.js";
import { requireEmitJs } from "../builtins/registry.js";
import {
  lookupBuiltin,
  newRuntimeState,
  renderJsRuntimeBodies,
  useRuntimeByName,
  type RuntimeState,
  type WorkspaceLike,
  type InlineSnippet,
} from "./runtime.js";
import { emitTensorConcatJs } from "./emitTensorConcatJs.js";

export interface EmitJsOptions {
  /** Workspace context. Threaded through `RuntimeState` so emit-time
   *  builtin lookups consult `.mtoc2.js` user builtins. Optional. */
  workspace?: WorkspaceLike;
  /** Include the activated runtime helper bodies inline at the top of
   *  the emitted module. Default true. When false the placeholder
   *  comment lands instead — useful for the IDE preview. */
  includeRuntime?: boolean;
  /** JIT-mode emit. When set, the cName of a single user-function spec
   *  to expose: the outer module returns a `($h) => specFn` factory
   *  instead of the script-style `function run($h)` wrapper. Requires
   *  `prog.topLevelStmts` to be empty and `prog.functions` to contain
   *  the named spec (which must be reachable along with any specs it
   *  transitively calls). Used by the numbl JIT bridge. */
  exposeSpec?: string;
}

export interface EmitJsResult {
  /** Full module source — runtime snippets, user-function defs, and
   *  either `function run($h)` + `return run;` (script mode) or
   *  `return ($h) => specFn;` (when `opts.exposeSpec` is set). Ready
   *  for `new Function(...)`. */
  source: string;
  /** Names of every runtime snippet activated during emit (debug aid;
   *  also useful to gate dependency wiring in tests). */
  activatedSnippets: ReadonlyArray<string>;
}

// ── Entry ────────────────────────────────────────────────────────────────

export function emitJsProgram(
  prog: IRProgram,
  opts: EmitJsOptions = {}
): EmitJsResult {
  const state = newRuntimeState(opts.workspace);
  const includeRuntime = opts.includeRuntime ?? true;

  if (opts.exposeSpec !== undefined) {
    if (prog.topLevelStmts.length > 0) {
      throw new Error(
        "emitJsProgram: 'exposeSpec' is incompatible with non-empty 'topLevelStmts'"
      );
    }
    if (!prog.functions.has(opts.exposeSpec)) {
      throw new Error(
        `emitJsProgram: 'exposeSpec' references unknown spec '${opts.exposeSpec}'`
      );
    }
  }

  // Function bodies are emitted first so call sites in `main`
  // reference already-defined functions. (JS hoisting would make this
  // unnecessary, but the resulting source stays human-readable.)
  const userParts: string[] = [];
  for (const fn of prog.functions.values()) {
    userParts.push(emitFunction(fn, state));
    userParts.push("");
  }

  // Outer wrapper. Script mode (default) emits `function run($h) { ... top-level ... }`
  // and `return run;`. JIT mode (`opts.exposeSpec` set) emits a factory that
  // binds `$h.write` and returns the named spec function directly.
  //
  // `$h.write` is the only required hook. Optional hooks (`$h.plotDispatch`)
  // surface as globals so the runtime helper snippets can call them without
  // threading `$h` through every emit site; helper-side `typeof === "function"`
  // checks gate behavior when a host doesn't provide the hook.
  const wrapperLines: string[] = [];
  if (opts.exposeSpec !== undefined) {
    wrapperLines.push(
      `return function ($h) { globalThis.$write = $h.write; globalThis.$plotDispatch = $h.plotDispatch; return ${opts.exposeSpec}; };`
    );
  } else {
    wrapperLines.push("function run($h) {");
    wrapperLines.push("  globalThis.$write = $h.write;");
    wrapperLines.push("  globalThis.$plotDispatch = $h.plotDispatch;");
    const locals = collectAssignedLocals(prog.topLevelStmts);
    if (locals.length > 0) {
      wrapperLines.push(`  let ${locals.join(", ")};`);
    }
    const bodyLines = emitBody(prog.topLevelStmts, "  ", state);
    if (bodyLines.length > 0) wrapperLines.push(bodyLines);
    wrapperLines.push("}");
    wrapperLines.push("return run;");
  }

  const out: string[] = [];
  if (state.active.size > 0) {
    out.push(
      includeRuntime ? renderJsRuntimeBodies(state) : runtimePlaceholder(state)
    );
    out.push("");
  }
  out.push(...userParts);
  out.push(...wrapperLines);

  return {
    source: out.join("\n"),
    activatedSnippets: Array.from(state.active),
  };
}

function runtimePlaceholder(state: RuntimeState): string {
  if (state.active.size === 0) return "";
  const names = Array.from(state.active).join(", ");
  return `/* runtime helpers omitted (${state.active.size}): ${names} */`;
}

// ── User function specialization ─────────────────────────────────────────

function emitFunction(fn: IRFunc, state: RuntimeState): string {
  // Multi-output user functions return an array; single-output return
  // the bare value; zero-output return nothing. We keep parameter
  // passing identical (positional args, no out-pointers).
  const params = fn.cParams.join(", ");
  const lines: string[] = [];
  lines.push(`function ${fn.cName}(${params}) {`);

  // Pre-declare every locally-assigned name. Parameters are already
  // in scope via the function signature; outputs that are declared by
  // Assign get included naturally. Skip names that match a parameter
  // (they'd shadow with `let`).
  const paramSet = new Set(fn.cParams);
  const locals = collectAssignedLocals(fn.body).filter(n => !paramSet.has(n));
  if (locals.length > 0) {
    lines.push(`  let ${locals.join(", ")};`);
  }

  // Stash the current function's output cNames on the state so any
  // in-body `ReturnFromFunction` can emit `return <fn outputs>` with
  // the matching shape (bare value / array / nothing). Cleared after.
  const prevOutputs = state.currentFnOutputs;
  state.currentFnOutputs = fn.cOutputs;
  const bodyLines = emitBody(fn.body, "  ", state);
  state.currentFnOutputs = prevOutputs;
  if (bodyLines.length > 0) lines.push(bodyLines);

  // Implicit return at function end based on declared outputs.
  if (fn.outputs.length === 0) {
    // nothing — JS returns undefined.
  } else if (fn.outputs.length === 1) {
    lines.push(`  return ${fn.cOutputs[0]};`);
  } else {
    lines.push(`  return [${fn.cOutputs.join(", ")}];`);
  }
  lines.push("}");
  return lines.join("\n");
}

function emitReturnFromFunction(state: RuntimeState, indent: string): string {
  const outs = state.currentFnOutputs;
  if (outs === undefined || outs.length === 0) {
    return `${indent}return;`;
  }
  if (outs.length === 1) {
    return `${indent}return ${outs[0]};`;
  }
  return `${indent}return [${outs.join(", ")}];`;
}

// ── Statement emission ───────────────────────────────────────────────────

function emitBody(
  stmts: ReadonlyArray<IRStmt>,
  indent: string,
  state: RuntimeState
): string {
  const out: string[] = [];
  for (const s of stmts) {
    const line = emitStmt(s, indent, state);
    if (line.length > 0) out.push(line);
  }
  return out.join("\n");
}

/** Emit a JS RHS expression with deep-clone wrapping when the
 *  expression yields an owned-typed value via an alias path (Var,
 *  MemberLoad, HandleCaptureLoad). Mirrors the c-aot
 *  `emitOwnedRhs` shape — without this, `b = a; b.f = rhs` would
 *  mutate `a` because the JS object reference is shared.
 *  Owned-producing expressions (TensorBuild, StructLit, HandleLit,
 *  fresh-allocating Calls / Binary / Unary) already yield a fresh
 *  value, so they fall through to `emitExpr` unchanged. */
function emitOwnedRhsJs(e: IRExpr, state: RuntimeState): string {
  if (
    e.kind === "Var" ||
    (e.kind === "MemberLoad" && isOwned(e.ty)) ||
    (e.kind === "HandleCaptureLoad" && isOwned(e.ty))
  ) {
    useRuntimeByName(state, "mtoc2_deep_clone");
    return `mtoc2_deep_clone(${emitExpr(e, state)})`;
  }
  return emitExpr(e, state);
}

function emitStmt(s: IRStmt, indent: string, state: RuntimeState): string {
  switch (s.kind) {
    case "ExprStmt":
      return `${indent}${emitExpr(s.expr, state)};`;

    case "Assign": {
      const rhs = isOwned(s.ty)
        ? emitOwnedRhsJs(s.expr, state)
        : emitExpr(s.expr, state);
      return `${indent}${s.cName} = ${rhs};`;
    }

    case "If": {
      const lines: string[] = [];
      lines.push(`${indent}if (${truthy(s.cond, state)}) {`);
      const thenBody = emitBody(s.thenBody, indent + "  ", state);
      if (thenBody.length > 0) lines.push(thenBody);
      if (s.elseBody.length > 0) {
        lines.push(`${indent}} else {`);
        const elseBody = emitBody(s.elseBody, indent + "  ", state);
        if (elseBody.length > 0) lines.push(elseBody);
      }
      lines.push(`${indent}}`);
      return lines.join("\n");
    }

    case "While": {
      const lines: string[] = [];
      lines.push(`${indent}while (${truthy(s.cond, state)}) {`);
      const body = emitBody(s.body, indent + "  ", state);
      if (body.length > 0) lines.push(body);
      lines.push(`${indent}}`);
      return lines.join("\n");
    }

    case "For": {
      // Mirror the C path's loop shape so JS output stays
      // bit-identical to numbl: snapshot start/end once at loop
      // entry, derive the iteration count via `mtoc2_loop_count`,
      // and rebind the loop var via `mtoc2_range_value` per iter.
      // This keeps the loop body insensitive to mid-body mutations
      // of `start`/`end` (matching MATLAB) and leaves the loop var
      // at its last actual iterated value after the loop ends.
      // `s.cVar` is pre-declared at function top by
      // `collectAssignedLocals` so reads after the loop see it.
      useRuntimeByName(state, "mtoc2_loop_count");
      useRuntimeByName(state, "mtoc2_range_value");
      const lines: string[] = [];
      const startE = emitExpr(s.start, state);
      const endE = emitExpr(s.end, state);
      const stepStr = formatJsNumber(s.step);
      lines.push(`${indent}{`);
      lines.push(`${indent}  const _mtoc2_for_start = ${startE};`);
      lines.push(`${indent}  const _mtoc2_for_end = ${endE};`);
      lines.push(
        `${indent}  const _mtoc2_for_n = mtoc2_loop_count(_mtoc2_for_start, _mtoc2_for_end, ${stepStr});`
      );
      lines.push(
        `${indent}  for (let _mtoc2_for_i = 0; _mtoc2_for_i < _mtoc2_for_n; _mtoc2_for_i++) {`
      );
      lines.push(
        `${indent}    ${s.cVar} = mtoc2_range_value(_mtoc2_for_start, ${stepStr}, _mtoc2_for_end, _mtoc2_for_n, _mtoc2_for_i);`
      );
      const body = emitBody(s.body, indent + "    ", state);
      if (body.length > 0) lines.push(body);
      lines.push(`${indent}  }`);
      lines.push(`${indent}}`);
      return lines.join("\n");
    }

    case "ReturnFromFunction":
      // Mid-body return: bail with whatever the declared outputs hold
      // right now. `emitFunction` stashes the active fn's cOutputs on
      // `state.currentFnOutputs` so the return form here matches the
      // function-end return (bare value / array / nothing).
      return emitReturnFromFunction(state, indent);

    case "Break":
      return `${indent}break;`;

    case "Continue":
      return `${indent}continue;`;

    case "TypeComment": {
      // Debug aid — emit as a JS comment so the source stays readable.
      // mtoc2's TypeComment can show multiple variables in one block;
      // each gets its own comment line.
      const lines = s.entries.map(
        en =>
          `${indent}// type ${en.name} (${en.cName}) :: ${typeToShortString(en.ty)}`
      );
      return lines.join("\n");
    }

    case "MultiAssignCall":
      return emitMultiAssignCall(s, indent, state);

    case "IndexStore": {
      useRuntimeByName(state, "mtoc2_scalar_index");
      // Member-rooted writes target the field slot via dotted access
      // on the (JS-side) struct/class object. The base Var still names
      // the owning root for liveness; here we walk the field path to
      // build the actual slot expression.
      const baseName =
        s.fieldPath !== undefined
          ? `${s.base.cName}.${s.fieldPath.join(".")}`
          : s.base.cName;
      const slotTy = s.leafTy ?? s.base.ty;
      const idxs = s.indices.map(ix => emitExpr(ix, state));
      // Grow-aware store subscripts: a store past the runtime extent
      // throws a tagged `mtoc2GrowBail` sentinel (the array would grow —
      // unsupported in the JIT) so the executor bails to the
      // interpreter. See `scalar_index.js`.
      let offset: string;
      if (idxs.length === 1) {
        offset = `mtoc2_idx_lin_grow_js(${baseName}, ${idxs[0]})`;
      } else {
        const terms: string[] = [];
        for (let i = 0; i < idxs.length; i++) {
          const checked = `mtoc2_idx_axis_grow_js(${baseName}, ${i}, ${idxs[i]})`;
          if (i === 0) {
            terms.push(checked);
          } else {
            const strides: string[] = [];
            for (let j = 0; j < i; j++) strides.push(`${baseName}.shape[${j}]`);
            terms.push(`${checked} * ${strides.join(" * ")}`);
          }
        }
        offset = terms.join(" + ");
      }
      const rhs = emitExpr(s.rhs, state);
      if (slotTy.kind === "Numeric" && slotTy.isComplex) {
        // Two-lane write. Hoist the RHS into a temp so we evaluate
        // the right-hand expression once even when it's an
        // unparenthesized {re, im} producer like `mtoc2_cmul(...)`.
        // The temp-var names use the root's cName to avoid clashes
        // when multiple fields of the same struct are written in
        // a single block (dot wouldn't be legal in a JS identifier).
        const safeBase = s.base.cName;
        const tmpOff = `_o_${safeBase}`;
        const tmpRhs = `_r_${safeBase}`;
        const rhsRe =
          s.rhs.ty.kind === "Numeric" && s.rhs.ty.isComplex
            ? `${tmpRhs}.re`
            : `${tmpRhs}`;
        const rhsIm =
          s.rhs.ty.kind === "Numeric" && s.rhs.ty.isComplex
            ? `${tmpRhs}.im`
            : `0`;
        return `${indent}{ const ${tmpOff} = ${offset}; const ${tmpRhs} = ${rhs}; ${baseName}.data[${tmpOff}] = ${rhsRe}; ${baseName}.imag[${tmpOff}] = ${rhsIm}; }`;
      }
      return `${indent}${baseName}.data[${offset}] = ${rhs};`;
    }

    case "MemberStore": {
      // `s.f = rhs` / `s.a.b = rhs` — walk the field path and assign
      // into the leaf. JS objects are mutable; no copy needed beyond
      // what the RHS expression already produced.
      const rhs = emitExpr(s.rhs, state);
      const path = s.fieldPath
        .map(f =>
          /^[A-Za-z_][A-Za-z0-9_]*$/.test(f)
            ? `.${f}`
            : `[${JSON.stringify(f)}]`
        )
        .join("");
      return `${indent}${s.base.cName}${path} = ${rhs};`;
    }

    case "IndexSliceStore":
      return emitIndexSliceStoreJs(s, indent, state);

    case "CellIndexStore": {
      // `c{i} = rhs` / `c{i, j} = rhs`. The cell's `data` is a JS
      // array; we compute the column-major linear offset (1-based →
      // 0-based) and store the rhs. Owned aliases are deep-cloned
      // by `emitOwnedRhsJs` so the cell owns its copy.
      const baseName = s.base.cName;
      const idxs = s.indices.map(ix => emitExpr(ix, state));
      let offset: string;
      if (idxs.length === 1) {
        offset = `(${idxs[0]} - 1)`;
      } else {
        const rowsExpr = `${baseName}.shape[0]`;
        offset = `((${idxs[1]} - 1) * ${rowsExpr} + (${idxs[0]} - 1))`;
      }
      const rhs = isOwned(s.rhs.ty)
        ? emitOwnedRhsJs(s.rhs, state)
        : emitExpr(s.rhs, state);
      return `${indent}${baseName}.data[${offset}] = ${rhs};`;
    }
  }
}

function emitIndexSliceStoreJs(
  s: Extract<IRStmt, { kind: "IndexSliceStore" }>,
  indent: string,
  state: RuntimeState
): string {
  // Same member-rooted slot-path treatment as the scalar IndexStore.
  const baseName =
    s.fieldPath !== undefined
      ? `${s.base.cName}.${s.fieldPath.join(".")}`
      : s.base.cName;
  const slotTy = s.leafTy ?? s.base.ty;
  const baseIsComplex = slotTy.kind === "Numeric" && slotTy.isComplex;
  const rhsTy = s.rhs.ty;
  const rhsIsScalarReal =
    rhsTy.kind === "Numeric" &&
    !rhsTy.isComplex &&
    rhsTy.dims.every(d => d.kind === "exact" && d.value === 1);
  const rhsIsScalarComplex =
    rhsTy.kind === "Numeric" &&
    rhsTy.isComplex &&
    rhsTy.dims.every(d => d.kind === "exact" && d.value === 1);
  const rhsIsScalar = rhsIsScalarReal || rhsIsScalarComplex;
  const rhsIsComplex = rhsTy.kind === "Numeric" && rhsTy.isComplex;
  const rhsExpr = emitExpr(s.rhs, state);

  // Per-slot write template. Reads one element from the RHS (a scalar
  // value or `_mtoc2_rhs.data/imag[k]`) and writes both data + imag
  // lanes into `baseName` at the given dst offset. Complex base with
  // real RHS sets imag = 0; real base with complex RHS would drop the
  // imag lane (we route through `transfer` earlier, so this is rare).
  const writeAt = (dstOff: string, srcIdx: string): string => {
    if (rhsIsScalarReal) {
      const reLine = `${baseName}.data[${dstOff}] = _mtoc2_rhs;`;
      const imLine = baseIsComplex ? ` ${baseName}.imag[${dstOff}] = 0;` : "";
      return reLine + imLine;
    }
    if (rhsIsScalarComplex) {
      const reLine = `${baseName}.data[${dstOff}] = _mtoc2_rhs.re;`;
      const imLine = baseIsComplex
        ? ` ${baseName}.imag[${dstOff}] = _mtoc2_rhs.im;`
        : "";
      return reLine + imLine;
    }
    // Tensor RHS: read one element from rhs.data (and rhs.imag when
    // the rhs is complex).
    const reLine = `${baseName}.data[${dstOff}] = _mtoc2_rhs.data[${srcIdx}];`;
    if (!baseIsComplex) return reLine;
    const imSrc = rhsIsComplex ? `_mtoc2_rhs.imag[${srcIdx}]` : `0`;
    return reLine + ` ${baseName}.imag[${dstOff}] = ${imSrc};`;
  };

  // Single-slot linear store.
  if (s.index.length === 1) {
    const slot = s.index[0];
    if (slot.kind === "Colon") {
      return (
        `${indent}{ ` +
        `const _mtoc2_rhs = ${rhsExpr}; ` +
        `const _mtoc2_n = ${baseName}.data.length; ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `${writeAt("_mtoc2_k", "_mtoc2_k")} ` +
        `} ` +
        `}`
      );
    }
    if (slot.kind === "Range") {
      useRuntimeByName(state, "mtoc2_loop_count");
      useRuntimeByName(state, "mtoc2_range_value");
      useRuntimeByName(state, "mtoc2_scalar_index");
      const sE = emitExpr(slot.start, state);
      const stE = emitExpr(slot.step, state);
      const eE = emitExpr(slot.end, state);
      // Tensor RHS: enforce numbl's `numel(src) == range_len`
      // runtime check so a mismatched-length assignment throws
      // instead of silently truncating.
      const lenCheck = rhsIsScalar
        ? ``
        : `if (_mtoc2_rhs.data.length !== _mtoc2_n) throw new Error("Unable to perform assignment because the size of the left side is " + _mtoc2_n + " and the size of the right side is " + _mtoc2_rhs.data.length + "."); `;
      // Each target offset goes through the grow-aware checker: a sub-1
      // index errors, and an index past the end grow-bails to the
      // interpreter (which has full grow semantics) — matching the
      // canonical `v(end+1:end+k) = ...` append, which previously
      // silently dropped the out-of-range writes here.
      return (
        `${indent}{ ` +
        `const _mtoc2_rhs = ${rhsExpr}; ` +
        `const _mtoc2_s = ${sE}, _mtoc2_e = ${eE}, _mtoc2_st = ${stE}; ` +
        `const _mtoc2_n = mtoc2_loop_count(_mtoc2_s, _mtoc2_e, _mtoc2_st); ` +
        lenCheck +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `const _mtoc2_v = mtoc2_range_value(_mtoc2_s, _mtoc2_st, _mtoc2_e, _mtoc2_n, _mtoc2_k); ` +
        `const _mtoc2_off = mtoc2_idx_lin_grow_js(${baseName}, _mtoc2_v); ` +
        `${writeAt("_mtoc2_off", "_mtoc2_k")} ` +
        `} ` +
        `}`
      );
    }
    if (slot.kind === "IndexVec") {
      useRuntimeByName(state, "mtoc2_scalar_index");
      const idxE = emitExpr(slot.expr, state);
      return (
        `${indent}{ ` +
        `const _mtoc2_rhs = ${rhsExpr}; ` +
        `const _mtoc2_ix = ${idxE}; ` +
        `const _mtoc2_ixd = _mtoc2_ix.mtoc2Tag === "tensor" ? _mtoc2_ix.data : [_mtoc2_ix]; ` +
        `const _mtoc2_n = _mtoc2_ixd.length; ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `const _mtoc2_off = mtoc2_idx_lin_grow_js(${baseName}, _mtoc2_ixd[_mtoc2_k]); ` +
        `${writeAt("_mtoc2_off", "_mtoc2_k")} ` +
        `} ` +
        `}`
      );
    }
    if (slot.kind === "LogicalMask") {
      // Linear single-slot logical mask write: scan, collect 0-based
      // truthy positions, write the RHS into each. Scalar RHS
      // broadcasts; tensor RHS reads positionally.
      const maskE = emitExpr(slot.expr, state);
      return (
        `${indent}{ ` +
        `const _mtoc2_rhs = ${rhsExpr}; ` +
        `const _mtoc2_m = ${maskE}; ` +
        `const _mtoc2_md = _mtoc2_m.data; ` +
        `const _mtoc2_ix = []; ` +
        `for (let _mtoc2_mi = 0; _mtoc2_mi < _mtoc2_md.length; _mtoc2_mi++) ` +
        `if (_mtoc2_md[_mtoc2_mi] !== 0) _mtoc2_ix.push(_mtoc2_mi); ` +
        `const _mtoc2_n = _mtoc2_ix.length; ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `${writeAt("_mtoc2_ix[_mtoc2_k]", "_mtoc2_k")} ` +
        `} ` +
        `}`
      );
    }
    // Single-slot Scalar writes go through IndexStore, not here.
    // Any other kind would be a new IndexSliceArg variant.
    throw new UnsupportedConstruct(
      `emitJs: IndexSliceStore single-slot '${slot.kind}' not yet wired`,
      s.span
    );
  }

  // Multi-slot per-axis store. Mirrors emitIndexSliceJs's multi-slot
  // form but writes into the base rather than reading into a fresh
  // tensor. RHS may be scalar (broadcast to every cell) or tensor
  // (cells read column-major).
  const ndim = s.index.length;
  const setup: string[] = [`const _mtoc2_rhs = ${rhsExpr};`];
  const idxFns: string[] = [];
  const dims: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const slot = s.index[i];
    if (slot.kind === "Colon") {
      setup.push(`const _mtoc2_n_${i} = ${baseName}.shape[${i}] ?? 1;`);
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(`(_mtoc2_k_${i} + 1)`);
    } else if (slot.kind === "Range") {
      useRuntimeByName(state, "mtoc2_loop_count");
      useRuntimeByName(state, "mtoc2_range_value");
      const sE = emitExpr(slot.start, state);
      const stE = emitExpr(slot.step, state);
      const eE = emitExpr(slot.end, state);
      setup.push(
        `const _mtoc2_s_${i} = ${sE}, _mtoc2_e_${i} = ${eE}, _mtoc2_st_${i} = ${stE};`
      );
      setup.push(
        `const _mtoc2_n_${i} = mtoc2_loop_count(_mtoc2_s_${i}, _mtoc2_e_${i}, _mtoc2_st_${i});`
      );
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(
        `mtoc2_range_value(_mtoc2_s_${i}, _mtoc2_st_${i}, _mtoc2_e_${i}, _mtoc2_n_${i}, _mtoc2_k_${i})`
      );
    } else if (slot.kind === "Scalar") {
      const v = emitExpr(slot.expr, state);
      setup.push(`const _mtoc2_s_${i} = ${v};`);
      setup.push(`const _mtoc2_n_${i} = 1;`);
      dims.push(`1`);
      idxFns.push(`_mtoc2_s_${i}`);
    } else if (slot.kind === "IndexVec") {
      const idxE = emitExpr(slot.expr, state);
      setup.push(
        `const _mtoc2_ix_${i} = ${idxE}; ` +
          `const _mtoc2_ixd_${i} = _mtoc2_ix_${i}.mtoc2Tag === "tensor" ? _mtoc2_ix_${i}.data : [_mtoc2_ix_${i}];`
      );
      setup.push(`const _mtoc2_n_${i} = _mtoc2_ixd_${i}.length;`);
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(`_mtoc2_ixd_${i}[_mtoc2_k_${i}]`);
    } else if (slot.kind === "LogicalMask") {
      const maskE = emitExpr(slot.expr, state);
      setup.push(
        `const _mtoc2_m_${i} = ${maskE}; ` +
          `const _mtoc2_md_${i} = _mtoc2_m_${i}.data; ` +
          `const _mtoc2_ixd_${i} = []; ` +
          `for (let _mtoc2_mi = 0; _mtoc2_mi < _mtoc2_md_${i}.length; _mtoc2_mi++) ` +
          `if (_mtoc2_md_${i}[_mtoc2_mi] !== 0) _mtoc2_ixd_${i}.push(_mtoc2_mi + 1);`
      );
      setup.push(`const _mtoc2_n_${i} = _mtoc2_ixd_${i}.length;`);
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(`_mtoc2_ixd_${i}[_mtoc2_k_${i}]`);
    } else {
      // Exhaustive: every IndexSliceArg kind handled above.
      const _exhaustive: never = slot;
      void _exhaustive;
      throw new UnsupportedConstruct(
        `emitJs: IndexSliceStore multi-slot unknown kind`,
        s.span
      );
    }
  }
  const lines: string[] = [`${indent}{`];
  for (const ln of setup) lines.push(`${indent}  ${ln}`);
  // Tensor RHS: enforce that the source has exactly slot-product
  // elements so a size-mismatched assignment throws instead of
  // silently truncating or padding.
  if (!rhsIsScalar) {
    const slotProduct = dims.join(" * ");
    lines.push(
      `${indent}  { const _mtoc2_total = ${slotProduct}; if (_mtoc2_rhs.data.length !== _mtoc2_total) throw new Error("Unable to perform assignment because the size of the left side is " + _mtoc2_total + " and the size of the right side is " + _mtoc2_rhs.data.length + "."); }`
    );
  }
  for (let i = ndim - 1; i >= 0; i--) {
    lines.push(
      `${indent}  for (let _mtoc2_k_${i} = 0; _mtoc2_k_${i} < _mtoc2_n_${i}; _mtoc2_k_${i}++) {`
    );
  }
  // Each axis index goes through the grow-aware checker: a sub-1 index
  // errors and an index past that axis's extent grow-bails to the
  // interpreter (which grows), instead of silently dropping the write.
  useRuntimeByName(state, "mtoc2_scalar_index");
  for (let i = 0; i < ndim; i++) {
    lines.push(
      `${indent}    const _mtoc2_ai_${i} = mtoc2_idx_axis_grow_js(${baseName}, ${i}, ${idxFns[i]});`
    );
  }
  const srcTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++)
      strideParts.push(`(${baseName}.shape[${j}] ?? 1)`);
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    srcTerms.push(`_mtoc2_ai_${i} * ${stride}`);
  }
  const rhsTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++) strideParts.push(dims[j]);
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    rhsTerms.push(`_mtoc2_k_${i} * ${stride}`);
  }
  lines.push(`${indent}    const _mtoc2_dst = ${srcTerms.join(" + ")};`);
  if (rhsIsScalar) {
    lines.push(`${indent}    ${writeAt("_mtoc2_dst", "0")}`);
  } else {
    lines.push(`${indent}    const _mtoc2_src = ${rhsTerms.join(" + ")};`);
    lines.push(`${indent}    ${writeAt("_mtoc2_dst", "_mtoc2_src")}`);
  }
  for (let i = ndim - 1; i >= 0; i--) lines.push(`${indent}  }`);
  lines.push(`${indent}}`);
  return lines.join("\n");
}

function emitMultiAssignCall(
  s: MultiAssignCall,
  indent: string,
  state: RuntimeState
): string {
  // Builtin path: dispatch through emitJs with outTargetsJs pre-built
  // as the destructure target names. JS destructure handles the
  // multi-return naturally. Owned-typed args route through the
  // deep-clone wrapper for the same pass-by-value reason single-
  // output user-function calls do.
  const args = s.args.map(a =>
    isOwned(a.ty) && !s.isBuiltin
      ? emitOwnedRhsJs(a, state)
      : emitExpr(a, state)
  );
  // Slot targets: each declared output's cName, or a synthesized
  // discard slot name (matched by `collectAssignedLocals`).
  const targets: string[] = s.outputs.map((slot, i) =>
    slot.binding === null ? discardSlotName(i) : slot.binding.cName
  );

  const b = lookupBuiltin(state, s.name);
  if (b !== undefined) {
    const emit = requireEmitJs(b);
    // Builtins control the call shape. For JS we don't pass out-pointers
    // (no such concept); the emitJs hook receives `outTargetsJs` so it
    // can decide whether to emit a destructure (preferred) or direct
    // assignments. We default to a destructure wrapper here unless the
    // hook returns an explicit statement string.
    const callExpr = emit({
      argsJs: args,
      argTypes: s.args.map(a => a.ty),
      nargout: s.outputs.length,
      outTargetsJs: targets,
      useRuntime: makeJsUseRuntime(state),
    });
    return `${indent}[${targets.join(", ")}] = ${callExpr};`;
  }

  // User-function call: array-return ABI. The JS spec returns
  // `[o1, o2, ...]`; destructure at the call site.
  return `${indent}[${targets.join(", ")}] = ${s.cName}(${args.join(", ")});`;
}

// ── Expression emission ──────────────────────────────────────────────────

function emitExpr(e: IRExpr, state: RuntimeState): string {
  switch (e.kind) {
    case "NumLit":
      return formatJsNumber(e.value);

    case "ImagLit":
      // Scalar complex literal. The JS-side representation is
      // `{re, im}` (matching the `mtoc2_cscalar.js` snippet's helpers).
      // Activate it lazily so the call site can rely on it being
      // available; the actual helper functions get inlined alongside.
      // (Phase 5 will populate `cscalar.js`.)
      return `{re: 0, im: ${formatJsNumber(e.value)}}`;

    case "StringLit": {
      // Match the interpreter's RuntimeValue conventions so JS-side
      // builtin call hooks (disp, error, fprintf, …) see the same shape
      // regardless of which backend executes them: `String` → bare JS
      // string; `Char` → `{mtoc2Tag:"char", value:"..."}` wrapper.
      const lit = JSON.stringify(e.value);
      if (e.ty.kind === "Char") {
        return `({mtoc2Tag: "char", value: ${lit}})`;
      }
      return lit;
    }

    case "Var":
      return e.cName;

    case "Binary": {
      const b = lookupBuiltin(state, e.builtin);
      if (!b) {
        throw new Error(`emitJs: builtin '${e.builtin}' not found`);
      }
      return requireEmitJs(b)({
        argsJs: [emitExpr(e.left, state), emitExpr(e.right, state)],
        argTypes: [e.left.ty, e.right.ty],
        nargout: 1,
        useRuntime: makeJsUseRuntime(state),
      });
    }

    case "Unary": {
      const b = lookupBuiltin(state, e.builtin);
      if (!b) {
        throw new Error(`emitJs: builtin '${e.builtin}' not found`);
      }
      return requireEmitJs(b)({
        argsJs: [emitExpr(e.operand, state)],
        argTypes: [e.operand.ty],
        nargout: 1,
        useRuntime: makeJsUseRuntime(state),
      });
    }

    case "Call": {
      const b = lookupBuiltin(state, e.name);
      if (b !== undefined) {
        return requireEmitJs(b)({
          argsJs: e.args.map(a => emitExpr(a, state)),
          argTypes: e.args.map(a => a.ty),
          nargout: 1,
          useRuntime: makeJsUseRuntime(state),
        });
      }
      // User-function call. Owned-typed args need a deep-clone
      // wrapper at the call site: MATLAB pass-by-value requires
      // that a callee's indexed write into its parameter not bleed
      // back to the caller. JS GC takes care of lifetime, but not
      // value-semantics — the deep_clone helper provides those.
      const args = e.args
        .map(a =>
          isOwned(a.ty) ? emitOwnedRhsJs(a, state) : emitExpr(a, state)
        )
        .join(", ");
      return `${e.cName}(${args})`;
    }

    case "MakeRange": {
      const startE = emitExpr(e.start, state);
      const stepE = emitExpr(e.step, state);
      const endE = emitExpr(e.end, state);
      useRuntimeByName(state, "mtoc2_tensor_make_range");
      return `mtoc2_tensor_make_range(${startE}, ${stepE}, ${endE})`;
    }

    case "TensorBuild": {
      const [rows, cols] = e.shape;
      const ty = e.ty;
      if (ty.kind === "Numeric" && ty.isComplex) {
        // For each element, split into a real / imag pair using the
        // element's static type. Statically-real elements pad imag
        // with 0; statically-complex elements emit `.re` / `.im`.
        useRuntimeByName(state, "mtoc2_cscalar");
        const res: string[] = [];
        const ims: string[] = [];
        for (const el of e.elements) {
          const exprStr = emitExpr(el, state);
          if (el.ty.kind === "Numeric" && el.ty.isComplex) {
            res.push(`(${exprStr}).re`);
            ims.push(`(${exprStr}).im`);
          } else {
            res.push(exprStr);
            ims.push(`0`);
          }
        }
        if (rows === 1) {
          useRuntimeByName(state, "mtoc2_tensor_from_row_complex");
          return `mtoc2_tensor_from_row_complex([${res.join(", ")}], [${ims.join(", ")}], ${cols})`;
        }
        useRuntimeByName(state, "mtoc2_tensor_from_matrix_complex");
        return `mtoc2_tensor_from_matrix_complex([${res.join(", ")}], [${ims.join(", ")}], ${rows}, ${cols})`;
      }
      const flat = e.elements.map(el => emitExpr(el, state)).join(", ");
      if (rows === 1) {
        useRuntimeByName(state, "mtoc2_tensor_from_row");
        return `mtoc2_tensor_from_row([${flat}], ${cols})`;
      }
      useRuntimeByName(state, "mtoc2_tensor_from_matrix");
      return `mtoc2_tensor_from_matrix([${flat}], ${rows}, ${cols})`;
    }

    case "IndexLoad": {
      // Scalar tensor read: `v(i)` or `M(i, j)`. ANF guarantees the
      // base is a bare Var; bounds-checking happens at runtime via
      // the shared `scalar_index.js` helpers.
      if (e.base.kind !== "Var") {
        throw new Error(
          `emitJs internal: IndexLoad base must be a Var after ANF (got ${e.base.kind})`
        );
      }
      useRuntimeByName(state, "mtoc2_scalar_index");
      const baseName = e.base.cName;
      const idxs = e.indices.map(ix => emitExpr(ix, state));
      let offset: string;
      if (idxs.length === 1) {
        offset = `mtoc2_idx_lin_js(${baseName}, ${idxs[0]})`;
      } else {
        const terms: string[] = [];
        for (let i = 0; i < idxs.length; i++) {
          const checked = `mtoc2_idx_axis_js(${baseName}, ${i}, ${idxs[i]})`;
          if (i === 0) {
            terms.push(checked);
          } else {
            const strides: string[] = [];
            for (let j = 0; j < i; j++) strides.push(`${baseName}.shape[${j}]`);
            terms.push(`${checked} * ${strides.join(" * ")}`);
          }
        }
        offset = terms.join(" + ");
      }
      if (e.base.ty.kind === "Numeric" && e.base.ty.isComplex) {
        // Build a fresh `{re, im}` from the two lanes. Hoist the
        // offset to a temp inline so we don't double-evaluate the
        // bounds-checked index expression.
        useRuntimeByName(state, "mtoc2_cscalar");
        return `(() => { const _o = ${offset}; return {re: ${baseName}.data[_o], im: ${baseName}.imag[_o]}; })()`;
      }
      return `${baseName}.data[${offset}]`;
    }

    case "EndRef": {
      if (e.baseTy.kind !== "Numeric") {
        throw new Error("emitJs internal: EndRef with non-numeric baseTy");
      }
      if (e.axis === "linear") return `${e.baseCName}.data.length`;
      return `${e.baseCName}.shape[${e.axis}]`;
    }

    case "TensorConcat": {
      return emitTensorConcatJs(e, state, emitExpr);
    }

    case "CellLit": {
      // `{a, b, c}` → fresh `mtoc2_cell_make`-built cell. Slot values
      // are owned-consumed; route owned aliases through deep_clone so
      // the cell owns its own copy (mirrors the c-aot consume rule).
      useRuntimeByName(state, "mtoc2_cell_make");
      const slots = e.elements
        .map(el =>
          isOwned(el.ty) ? emitOwnedRhsJs(el, state) : emitExpr(el, state)
        )
        .join(", ");
      const shapeLit = JSON.stringify(e.shape);
      return `mtoc2_cell_make([${slots}], ${shapeLit})`;
    }

    case "CellEmpty": {
      // `cell(n, m, ...)` → fresh cell with empty-double slots.
      // The 1-arg square form expands to two dim args at lowering
      // time; everything here is the resolved per-axis form.
      useRuntimeByName(state, "mtoc2_cell_empty");
      const dimsJs = e.dims.map(d => emitExpr(d, state)).join(", ");
      return `mtoc2_cell_empty([${dimsJs}])`;
    }

    case "CellIndexLoad": {
      // `c{i}` / `c{i, j}` — column-major linear offset into
      // `base.data`. Indices are 1-based in source; offset by -1.
      if (e.base.kind !== "Var") {
        throw new Error(
          `emitJs internal: CellIndexLoad base must be a Var (got ${e.base.kind})`
        );
      }
      const baseName = e.base.cName;
      const idxs = e.indices.map(ix => emitExpr(ix, state));
      if (idxs.length === 1) {
        return `${baseName}.data[(${idxs[0]}) - 1]`;
      }
      return `${baseName}.data[((${idxs[1]}) - 1) * ${baseName}.shape[0] + ((${idxs[0]}) - 1)]`;
    }

    case "IndexSlice":
      return emitIndexSliceJs(e, state);

    case "StructLit": {
      // Plain JS object — field-name keys carry the lowered values.
      // Owned-typed field values are fresh producers (ANF guaranteed)
      // and JS GC handles lifetime, so no wrapping needed.
      const fields = e.fields
        .map(f => `${JSON.stringify(f.name)}: ${emitExpr(f.value, state)}`)
        .join(", ");
      return `({${fields}})`;
    }

    case "MemberLoad": {
      // `s.f` — JS member access. Identifier-safe field names skip
      // the bracket form for readability; everything else uses
      // bracket notation.
      const baseE = emitExpr(e.base, state);
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(e.field)) {
        return `(${baseE}).${e.field}`;
      }
      return `(${baseE})[${JSON.stringify(e.field)}]`;
    }

    case "HandleLit": {
      // Captures-by-value snapshot, stored under their source names
      // so HandleCaptureLoad can read them back. The dispatch target
      // is encoded in the lowered Call IR at the call site; the JS
      // handle object itself only needs to carry the captures.
      const caps = e.captures
        .map(c => `${JSON.stringify(c.name)}: ${emitExpr(c.value, state)}`)
        .join(", ");
      return `({${caps}})`;
    }

    case "HandleCaptureLoad": {
      // `base.<name>` — read a previously-snapshot capture out of the
      // handle struct.
      const baseE = emitExpr(e.base, state);
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(e.captureName)) {
        return `(${baseE}).${e.captureName}`;
      }
      return `(${baseE})[${JSON.stringify(e.captureName)}]`;
    }
  }
}

function emitIndexSliceJs(
  e: Extract<IRExpr, { kind: "IndexSlice" }>,
  state: RuntimeState
): string {
  if (e.base.kind !== "Var") {
    throw new Error(
      `emitJs internal: IndexSlice base must be a Var after ANF (got ${e.base.kind})`
    );
  }
  const isComplexBase = e.base.ty.kind === "Numeric" && e.base.ty.isComplex;
  const allocFn = isComplexBase
    ? "mtoc2_tensor_alloc_nd_complex"
    : "mtoc2_tensor_alloc_nd";
  useRuntimeByName(state, allocFn);
  const baseName = e.base.cName;
  // Per-element copy template — fills both lanes for complex bases.
  // Used inside every per-slot loop body so the same emit path serves
  // real and complex slices.
  const copy = (dstIdx: string, srcIdx: string): string =>
    isComplexBase
      ? `_mtoc2_t.data[${dstIdx}] = ${baseName}.data[${srcIdx}]; _mtoc2_t.imag[${dstIdx}] = ${baseName}.imag[${srcIdx}];`
      : `_mtoc2_t.data[${dstIdx}] = ${baseName}.data[${srcIdx}];`;

  // Single-slot linear form. Covers `v(:)`, `v(a:b)`, `v(idx_vec)`,
  // `v(mask)`. Multi-slot Scalar single-slot reads route through
  // IndexLoad, not IndexSlice, so we don't see them here.
  if (e.index.length === 1) {
    const slot = e.index[0];
    const baseNum = e.base.ty.kind === "Numeric" ? e.base.ty : undefined;
    const isColVec = baseNum !== undefined && isColVecTy(baseNum);
    const isRowVec = baseNum !== undefined && isRowVecTy(baseNum);
    // A slice that is statically a single element — `v(2:2)`, `v(:)` on a
    // 1×1 base, or `v([k])` — is typed scalar at the consume site (`y = …`
    // then used as a number), so it MUST evaluate to a scalar value (a JS
    // number, or `{re, im}` for a complex base), NOT an allocated 1×1
    // tensor object. The C producer (emitIndex.ts) already does this;
    // without it the JS backend leaks a tensor object into scalar
    // arithmetic/conditions (e.g. `v(2:2) + 1` → `"[object Object]1"`,
    // `if v(2:2) > 15` taking the wrong branch). LogicalMask is never
    // statically single-element, so it is not handled here.
    if (isScalar(e.ty)) {
      let off: string;
      if (slot.kind === "Colon") {
        off = "0";
      } else if (slot.kind === "Range") {
        useRuntimeByName(state, "mtoc2_loop_count");
        useRuntimeByName(state, "mtoc2_range_value");
        const s = emitExpr(slot.start, state);
        const st = emitExpr(slot.step, state);
        const en = emitExpr(slot.end, state);
        off =
          `(() => { const _mtoc2_s = ${s}, _mtoc2_e = ${en}, _mtoc2_st = ${st}; ` +
          `const _mtoc2_n = mtoc2_loop_count(_mtoc2_s, _mtoc2_e, _mtoc2_st); ` +
          `const _mtoc2_idx = Math.round(mtoc2_range_value(_mtoc2_s, _mtoc2_st, _mtoc2_e, _mtoc2_n, 0)); ` +
          `const _mtoc2_len = ${baseName}.data.length; ` +
          `if (_mtoc2_idx < 1 || _mtoc2_idx > _mtoc2_len) throw new Error("Index in position 1 exceeds array bounds. Index must not exceed " + _mtoc2_len + "."); ` +
          `return _mtoc2_idx - 1; })()`;
      } else if (slot.kind === "IndexVec") {
        const idx = emitExpr(slot.expr, state);
        off =
          `(() => { const _mtoc2_ix = ${idx}; ` +
          `const _mtoc2_ixd = _mtoc2_ix.mtoc2Tag === "tensor" ? _mtoc2_ix.data : [_mtoc2_ix]; ` +
          `return Math.round(_mtoc2_ixd[0]) - 1; })()`;
      } else {
        throw new UnsupportedConstruct(
          `emitJs internal: scalar IndexSlice slot kind ${slot.kind}`,
          e.span
        );
      }
      if (isComplexBase) {
        return `(() => { const _o = ${off}; return {re: ${baseName}.data[_o], im: ${baseName}.imag[_o]}; })()`;
      }
      return `${baseName}.data[${off}]`;
    }
    if (slot.kind === "Colon") {
      return (
        `(() => { ` +
        `const _mtoc2_n = ${baseName}.data.length; ` +
        `const _mtoc2_t = ${allocFn}(2, [_mtoc2_n, 1]); ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `${copy("_mtoc2_k", "_mtoc2_k")} ` +
        `} ` +
        `return _mtoc2_t; ` +
        `})()`
      );
    }
    if (slot.kind === "Range") {
      useRuntimeByName(state, "mtoc2_loop_count");
      useRuntimeByName(state, "mtoc2_range_value");
      const s = emitExpr(slot.start, state);
      const st = emitExpr(slot.step, state);
      const en = emitExpr(slot.end, state);
      const rows = isColVec ? "_mtoc2_n" : "1";
      const cols = isColVec ? "1" : "_mtoc2_n";
      // Per-element 1-based bounds check against the source's linear
      // length. Mirrors numbl's RuntimeError on out-of-range slice
      // indices (e.g. `oob(0:3)` → index 0 fails the lower-bound
      // guard).
      return (
        `(() => { ` +
        `const _mtoc2_s = ${s}; const _mtoc2_e = ${en}; const _mtoc2_st = ${st}; ` +
        `const _mtoc2_n = mtoc2_loop_count(_mtoc2_s, _mtoc2_e, _mtoc2_st); ` +
        `const _mtoc2_t = ${allocFn}(2, [${rows}, ${cols}]); ` +
        `const _mtoc2_len = ${baseName}.data.length; ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `const _mtoc2_v = mtoc2_range_value(_mtoc2_s, _mtoc2_st, _mtoc2_e, _mtoc2_n, _mtoc2_k); ` +
        `const _mtoc2_idx = Math.round(_mtoc2_v); ` +
        `if (_mtoc2_idx < 1 || _mtoc2_idx > _mtoc2_len) throw new Error("Index in position 1 exceeds array bounds. Index must not exceed " + _mtoc2_len + "."); ` +
        `${copy("_mtoc2_k", "_mtoc2_idx - 1")} ` +
        `} ` +
        `return _mtoc2_t; ` +
        `})()`
      );
    }
    if (slot.kind === "Scalar") {
      // Single-slot Scalar should have routed to IndexLoad; this is
      // a safety net.
      throw new UnsupportedConstruct(
        `emitJs internal: single-slot Scalar IndexSlice should route to IndexLoad`,
        e.span
      );
    }
    if (slot.kind === "IndexVec") {
      const idx = emitExpr(slot.expr, state);
      const rows = isRowVec ? "1" : "_mtoc2_n";
      const cols = isRowVec ? "_mtoc2_n" : "1";
      return (
        `(() => { ` +
        `const _mtoc2_ix = ${idx}; ` +
        `const _mtoc2_ixd = _mtoc2_ix.mtoc2Tag === "tensor" ? _mtoc2_ix.data : [_mtoc2_ix]; ` +
        `const _mtoc2_n = _mtoc2_ixd.length; ` +
        `const _mtoc2_t = ${allocFn}(2, [${rows}, ${cols}]); ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `${copy("_mtoc2_k", "Math.round(_mtoc2_ixd[_mtoc2_k]) - 1")} ` +
        `} ` +
        `return _mtoc2_t; ` +
        `})()`
      );
    }
    if (slot.kind === "LogicalMask") {
      // Single-slot logical mask: linear scan, collect 1-based
      // positions of truthy elements, then gather. Result orientation
      // follows the base: row base → row, otherwise column.
      const mask = emitExpr(slot.expr, state);
      const rows = isRowVec ? "1" : "_mtoc2_n";
      const cols = isRowVec ? "_mtoc2_n" : "1";
      return (
        `(() => { ` +
        `const _mtoc2_m = ${mask}; ` +
        `const _mtoc2_md = _mtoc2_m.data; ` +
        `const _mtoc2_blen = ${baseName}.data.length; ` +
        `const _mtoc2_ix = []; ` +
        `for (let _mtoc2_mi = 0; _mtoc2_mi < _mtoc2_md.length; _mtoc2_mi++) ` +
        `if (_mtoc2_md[_mtoc2_mi] !== 0) { ` +
        // A truthy mask bit past the base end is an error in MATLAB (the C
        // path aborts too); without this the gather reads OOB -> NaN.
        `if (_mtoc2_mi >= _mtoc2_blen) throw new Error("Index exceeds array bounds"); ` +
        `_mtoc2_ix.push(_mtoc2_mi); } ` +
        `const _mtoc2_n = _mtoc2_ix.length; ` +
        `const _mtoc2_t = ${allocFn}(2, [${rows}, ${cols}]); ` +
        `for (let _mtoc2_k = 0; _mtoc2_k < _mtoc2_n; _mtoc2_k++) { ` +
        `${copy("_mtoc2_k", "_mtoc2_ix[_mtoc2_k]")} ` +
        `} ` +
        `return _mtoc2_t; ` +
        `})()`
      );
    }
    // Exhaustive: every IndexSliceArg kind is handled above.
    const _exhaustive: never = slot;
    void _exhaustive;
    throw new UnsupportedConstruct(
      `emitJs: IndexSlice single-slot unknown kind`,
      e.span
    );
  }

  // Multi-slot per-axis form. Each slot maps to a per-axis count and
  // a per-iteration 1-based source-index expression. The result is
  // allocated with the slot counts as its dims (rank-2 floor; trailing
  // exact-1 axes already collapsed by the lowerer's transfer).
  const ndim = e.index.length;
  const setup: string[] = [];
  const idxFns: string[] = [];
  const dims: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const slot = e.index[i];
    if (slot.kind === "Colon") {
      setup.push(`const _mtoc2_n_${i} = ${baseName}.shape[${i}] ?? 1;`);
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(`(_mtoc2_k_${i} + 1)`);
    } else if (slot.kind === "Range") {
      useRuntimeByName(state, "mtoc2_loop_count");
      useRuntimeByName(state, "mtoc2_range_value");
      const s = emitExpr(slot.start, state);
      const st = emitExpr(slot.step, state);
      const en = emitExpr(slot.end, state);
      setup.push(
        `const _mtoc2_s_${i} = ${s}, _mtoc2_e_${i} = ${en}, _mtoc2_st_${i} = ${st};`
      );
      setup.push(
        `const _mtoc2_n_${i} = mtoc2_loop_count(_mtoc2_s_${i}, _mtoc2_e_${i}, _mtoc2_st_${i});`
      );
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(
        `mtoc2_range_value(_mtoc2_s_${i}, _mtoc2_st_${i}, _mtoc2_e_${i}, _mtoc2_n_${i}, _mtoc2_k_${i})`
      );
    } else if (slot.kind === "Scalar") {
      const v = emitExpr(slot.expr, state);
      setup.push(`const _mtoc2_s_${i} = ${v};`);
      setup.push(`const _mtoc2_n_${i} = 1;`);
      dims.push(`1`);
      idxFns.push(`_mtoc2_s_${i}`);
    } else if (slot.kind === "IndexVec") {
      const idxE = emitExpr(slot.expr, state);
      setup.push(
        `const _mtoc2_ix_${i} = ${idxE}; ` +
          `const _mtoc2_ixd_${i} = _mtoc2_ix_${i}.mtoc2Tag === "tensor" ? _mtoc2_ix_${i}.data : [_mtoc2_ix_${i}];`
      );
      setup.push(`const _mtoc2_n_${i} = _mtoc2_ixd_${i}.length;`);
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(`_mtoc2_ixd_${i}[_mtoc2_k_${i}]`);
    } else if (slot.kind === "LogicalMask") {
      // Multi-slot logical mask: scan, collect 0-based truthy
      // positions along this axis. The selected axis size is the
      // truthy count; each iteration of `_mtoc2_k_${i}` picks the
      // k-th truthy position (1-based at idxFn-time).
      const maskE = emitExpr(slot.expr, state);
      setup.push(
        `const _mtoc2_m_${i} = ${maskE}; ` +
          `const _mtoc2_md_${i} = _mtoc2_m_${i}.data; ` +
          `const _mtoc2_ixd_${i} = []; ` +
          `for (let _mtoc2_mi = 0; _mtoc2_mi < _mtoc2_md_${i}.length; _mtoc2_mi++) ` +
          `if (_mtoc2_md_${i}[_mtoc2_mi] !== 0) _mtoc2_ixd_${i}.push(_mtoc2_mi + 1);`
      );
      setup.push(`const _mtoc2_n_${i} = _mtoc2_ixd_${i}.length;`);
      dims.push(`_mtoc2_n_${i}`);
      idxFns.push(`_mtoc2_ixd_${i}[_mtoc2_k_${i}]`);
    } else {
      // Exhaustive: every IndexSliceArg kind is handled above.
      const _exhaustive: never = slot;
      void _exhaustive;
      throw new UnsupportedConstruct(
        `emitJs: IndexSlice multi-slot unknown kind`,
        e.span
      );
    }
  }
  // Multi-slot slice that is statically a single element — e.g.
  // `M(1:1, 2)` or `M(i, 2:2)` — is typed scalar at the consume site, so
  // yield the scalar element rather than a 1×1 tensor object (mirrors the
  // C producer in emitIndex.ts). Pin every axis loop index to 0, compute
  // the one source offset (with the same per-axis bounds checks), read.
  if (isScalar(e.ty)) {
    const slines: string[] = [];
    for (const s of setup) slines.push(s);
    for (let i = 0; i < ndim; i++) slines.push(`const _mtoc2_k_${i} = 0;`);
    for (let i = 0; i < ndim; i++) {
      slines.push(`const _mtoc2_idx_${i} = Math.round(${idxFns[i]});`);
      slines.push(
        `if (_mtoc2_idx_${i} < 1 || _mtoc2_idx_${i} > (${baseName}.shape[${i}] ?? 1)) ` +
          `throw new Error("Index in position ${i + 1} exceeds array bounds.");`
      );
    }
    const srcTerms: string[] = [];
    for (let i = 0; i < ndim; i++) {
      const strideParts: string[] = [];
      for (let j = 0; j < i; j++)
        strideParts.push(`(${baseName}.shape[${j}] ?? 1)`);
      const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
      srcTerms.push(`(_mtoc2_idx_${i} - 1) * ${stride}`);
    }
    slines.push(`const _mtoc2_off = ${srcTerms.join(" + ")};`);
    if (isComplexBase) {
      slines.push(
        `return {re: ${baseName}.data[_mtoc2_off], im: ${baseName}.imag[_mtoc2_off]};`
      );
    } else {
      slines.push(`return ${baseName}.data[_mtoc2_off];`);
    }
    return `(() => { ${slines.join(" ")} })()`;
  }

  // Result shape canonicalization: trim trailing statically-known-1
  // axes down to a 2-axis floor, then pad up to 2 if shorter. Mirrors
  // numbl's display convention so `t(:, :, 2)` (slot pattern Colon /
  // Colon / Scalar) produces a 2×3 result rather than 2×3×1, and a
  // single-axis remainder squares up to 2-axis for disp.
  const resultDims = dims.slice();
  while (resultDims.length > 2 && resultDims[resultDims.length - 1] === "1") {
    resultDims.pop();
  }
  while (resultDims.length < 2) resultDims.push("1");

  const lines: string[] = [];
  for (const s of setup) lines.push(s);
  lines.push(
    `const _mtoc2_t = ${allocFn}(${resultDims.length}, [${resultDims.join(", ")}]);`
  );
  for (let i = ndim - 1; i >= 0; i--) {
    lines.push(
      `for (let _mtoc2_k_${i} = 0; _mtoc2_k_${i} < _mtoc2_n_${i}; _mtoc2_k_${i}++) {`
    );
  }
  // Per-axis 1-based index, rounded, with a bounds check against that
  // axis's extent — mirrors the interpreter's "Index in position N
  // exceeds array bounds" RuntimeError. Without it, an out-of-range
  // axis index reads `undefined` from the buffer and silently yields
  // NaN. Colon slots are always in range; the check is cheap and
  // uniform across slot kinds (Range / Scalar / IndexVec / mask).
  for (let i = 0; i < ndim; i++) {
    lines.push(`const _mtoc2_idx_${i} = Math.round(${idxFns[i]});`);
    lines.push(
      `if (_mtoc2_idx_${i} < 1 || _mtoc2_idx_${i} > (${baseName}.shape[${i}] ?? 1)) ` +
        `throw new Error("Index in position ${i + 1} exceeds array bounds.");`
    );
  }
  // Column-major source offset: sum_i (idx[i] - 1) * stride[i] where
  // stride[i] = product of base.shape[0..i).
  const srcTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++) {
      strideParts.push(`(${baseName}.shape[${j}] ?? 1)`);
    }
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    srcTerms.push(`(_mtoc2_idx_${i} - 1) * ${stride}`);
  }
  // Column-major destination offset using the result's own dims.
  const dstTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++) strideParts.push(dims[j]);
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    dstTerms.push(`_mtoc2_k_${i} * ${stride}`);
  }
  lines.push(`const _mtoc2_src_off = ${srcTerms.join(" + ")};`);
  lines.push(`const _mtoc2_dst_off = ${dstTerms.join(" + ")};`);
  lines.push(copy("_mtoc2_dst_off", "_mtoc2_src_off"));
  for (let i = ndim - 1; i >= 0; i--) lines.push(`}`);
  lines.push(`return _mtoc2_t;`);
  return `(() => { ${lines.join(" ")} })()`;
}

// ── Helpers ──────────────────────────────────────────────────────────────

function makeJsUseRuntime(
  state: RuntimeState
): (spec: string | InlineSnippet) => void {
  return spec => {
    if (typeof spec === "string") {
      useRuntimeByName(state, spec);
    } else {
      // Inline snippet — mirror the C path's `useRuntimeInline` but
      // keep both code + jsCode fields populated.
      if (!state.extraSnippets.has(spec.name)) {
        const stored = {
          headers: spec.headers ?? [],
          code: spec.code,
          deps: spec.deps ?? [],
          ...(spec.jsCode !== undefined ? { jsCode: spec.jsCode } : {}),
        };
        state.extraSnippets.set(spec.name, stored);
      }
      useRuntimeByName(state, spec.name);
    }
  };
}

/** Wrap a cond expression with the truthiness conversion that matches
 *  numbl/MATLAB's `toBool` semantics — a scalar is true iff it is
 *  nonzero (`x !== 0`). Real numeric scalars must use an explicit
 *  `!== 0` because JS truthiness disagrees: `NaN` is falsy in JS but
 *  numbl treats a (nonzero) `NaN` as true. Logical scalars are already
 *  JS booleans, so they pass through directly. Complex scalars route
 *  through `mtoc2_cnonzero` — a `{re: 0, im: 0}` object is "truthy" in
 *  JS as a non-null reference, which would loop forever in `while (z)`. */
function truthy(e: IRExpr, state: RuntimeState): string {
  const expr = emitExpr(e, state);
  if (e.ty.kind === "Numeric") {
    if (e.ty.isComplex) {
      useRuntimeByName(state, "mtoc2_cscalar");
      return `mtoc2_cnonzero(${expr})`;
    }
    // Real double: NaN must be truthy (matches the C backend's
    // `!= 0.0` and the interpreter's `toBool`). A logical scalar is
    // already a JS boolean and passes through unchanged.
    if (e.ty.elem === "double") return `(${expr} !== 0)`;
  }
  return expr;
}

/** Walk every Assign / MultiAssignCall in `stmts` and return the
 *  unique cNames they introduce (in source order). Used to pre-declare
 *  all locals at the top of a function / top-level body so subsequent
 *  bare assignments compile cleanly. */
function collectAssignedLocals(stmts: ReadonlyArray<IRStmt>): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  const add = (name: string): void => {
    if (seen.has(name)) return;
    seen.add(name);
    order.push(name);
  };
  const visit = (s: IRStmt): void => {
    switch (s.kind) {
      case "Assign":
        add((s as Assign).cName);
        break;
      case "MultiAssignCall":
        s.outputs.forEach((slot, i) => {
          if (slot.binding !== null) {
            add(slot.binding.cName);
          } else {
            // Discard slot needs a local too, to receive the JS
            // destructure (we can't use bare `_` because two adjacent
            // multi-assigns would conflict on it).
            add(discardSlotName(i));
          }
        });
        break;
      case "For":
        // Pre-declare the loop var at the enclosing function scope so
        // it stays visible after the loop ends (MATLAB-style).
        add(s.cVar);
        for (const sub of s.body) visit(sub);
        break;
      case "While":
        for (const sub of s.body) visit(sub);
        break;
      case "If":
        for (const sub of s.thenBody) visit(sub);
        for (const sub of s.elseBody) visit(sub);
        break;
      default:
        break;
    }
  };
  for (const s of stmts) visit(s);
  return order;
}

function formatJsNumber(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "Infinity";
  if (v === -Infinity) return "-Infinity";
  if (Object.is(v, -0)) return "-0";
  return String(v);
}

function typeToShortString(t: import("../lowering/types.js").Type): string {
  // Cheap shape preview — full pretty-printing lives in
  // `lowering/types.ts::typeToString`. The comment is debug-only.
  return t.kind;
}

/** Stable name for a discard slot in a `MultiAssignCall` — used both
 *  by `collectAssignedLocals` (to pre-declare it) and by
 *  `emitMultiAssignCall` (to emit the destructure target). The index
 *  uniquifies adjacent discards at the same call site so a
 *  `[~, ~] = foo()` doesn't collide on one name. */
function discardSlotName(slotIdx: number): string {
  return `_mtoc2_discard_${slotIdx}`;
}

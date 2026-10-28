/**
 * Single-file JS codegen for JIT IR. Mirrors `emit.ts` (the C path)
 * intentionally — same control flow, same variable layout, same call
 * shape — differing only in language and the I/O primitive (`$write(s)`
 * vs `printf(...)`).
 *
 * Pass-by-value protocol: owned-typed RHSes (Var / MemberLoad /
 * HandleCaptureLoad) at Assign, user-function Call args, and
 * MultiAssignCall args route through `emitOwnedRhsJs`, which wraps
 * the expression in `jit_deep_clone(...)`. JS GC handles lifetime,
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
 *   let $write, $plotDispatch, $rand, $elemwise, $cancel, $loc;
 *   <inlined runtime snippet bodies>
 *   function <user-fn-spec-1>(...) { ... }
 *   function <user-fn-spec-2>(...) { ... }
 *   return function ($h) { $write = $h.write; ...; return <entry-spec>; };
 *   ```
 *
 * The host evals the source via `new Function(source)()`, then calls
 * the returned factory with its helper hooks to get the entry spec.
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
import { getBuiltin } from "../builtins/index.js";
import { requireEmitJs } from "../builtins/registry.js";
import {
  locCode,
  newRuntimeState,
  renderJsRuntimeBodies,
  useRuntimeByName,
  type RuntimeState,
  type InlineSnippet,
} from "./runtime.js";
import { emitTensorConcatJs } from "./emitTensorConcatJs.js";
import { maybeUnassignedReads } from "../lowering/definiteAssign.js";
import { callsSpec } from "../lowering/callGraph.js";
import {
  STRIP_CHUNK,
  breakLabel,
  enterLoop,
  exitLoop,
  isConstantTimeLoop,
} from "./cancelPoll.js";

export interface EmitJsOptions {
  /** cName of the entry spec the module's factory returns.
   *  `prog.functions` must contain it and every spec it transitively
   *  calls. */
  exposeSpec: string;
}

export interface EmitJsResult {
  /** Full module source — runtime snippets, user-function defs, and
   *  `return ($h) => specFn;`. Ready for `new Function(...)`. */
  source: string;
  /** Names of every runtime snippet activated during emit (debug aid;
   *  also useful to gate dependency wiring in tests). */
  activatedSnippets: ReadonlyArray<string>;
  /** Files named by statement locations, by index (see `locCode`). */
  locFiles: string[];
}

// ── Entry ────────────────────────────────────────────────────────────────

export function emitJsProgram(
  prog: IRProgram,
  opts: EmitJsOptions
): EmitJsResult {
  const state = newRuntimeState();

  if (!prog.functions.has(opts.exposeSpec)) {
    throw new Error(
      `emitJsProgram: 'exposeSpec' references unknown spec '${opts.exposeSpec}'`
    );
  }
  // The raise channel (`JitRaise`), shared by every runtime error site.
  useRuntimeByName(state, "jit_raise");
  state.maybeUnbound = new Set();
  for (const fn of prog.functions.values()) {
    for (const read of maybeUnassignedReads(fn)) state.maybeUnbound.add(read);
  }

  const userParts: string[] = [];
  for (const fn of prog.functions.values()) {
    userParts.push(emitFunction(fn, state, callsSpec(prog, fn)));
    userParts.push("");
  }

  // The host hooks (`$write`, `$plotDispatch`, `$rand`, `$elemwise`) are
  // module-scoped bindings that the factory assigns from `$h`, so emitted
  // code and runtime helper snippets call them by bare name without
  // threading `$h` through every emit site. They are per-module (not
  // globals) so two runtimes in one process never share a host. Helper-side
  // `typeof === "function"` checks gate behavior when a host omits a hook.
  const out: string[] = [];
  // `$loc` is the statement location (see `locCode`), read by the host
  // through `$h.loc` when a raise escapes. `$cancel` is the host's cancel
  // flag (an Int32Array; see `CANCEL_POLL_JS`).
  out.push("let $write, $plotDispatch, $rand, $elemwise, $cancel, $loc = -1;");
  if (state.active.size > 0) {
    out.push(renderJsRuntimeBodies(state));
    out.push("");
  }
  out.push(...userParts);
  out.push(
    `return function ($h) { $write = $h.write; $plotDispatch = $h.plotDispatch; $rand = $h.rand; $elemwise = $h.elemwise; $cancel = $h.cancel; $h.loc = () => $loc; return ${opts.exposeSpec}; };`
  );

  return {
    source: out.join("\n"),
    activatedSnippets: Array.from(state.active),
    locFiles: state.locFiles,
  };
}

/** Cancel polls. The host's flag (`$cancel`, an Int32Array the thread that
 *  cancels writes) is read through `_jit_cancel`, a function-local alias,
 *  with a plain load: reading the module binding, or `Atomics.load`, costs
 *  several times more in a tight loop. V8 does not eliminate typed-array
 *  loads across iterations, so the other thread's store is seen (the cancel
 *  tests check this).
 *
 *  A loop polls in its condition (`CANCEL_OK_JS`) and raises after it exits
 *  (`CANCEL_POLL_JS`): a call to the raise helper inside the loop body, even
 *  one never taken, slows a tight V8 loop by up to 3x. The extra condition
 *  still costs 10% to 80% in a one-statement scalar loop, so a loop whose
 *  iterations take constant time is strip-mined (`cancelPoll.ts`).
 *  Functions that call specs (and so may recurse) poll on entry. See
 *  `raise.js`. */
const CANCEL_OK_JS = "_jit_cancel[0] === 0";
const CANCEL_POLL_JS = "if (_jit_cancel[0] !== 0) jit_raise_cancel();";
// ── User function specialization ─────────────────────────────────────────

function emitFunction(
  fn: IRFunc,
  state: RuntimeState,
  pollOnEntry: boolean
): string {
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
  const pollAt = lines.length;
  if (pollOnEntry) lines.push(`  ${CANCEL_POLL_JS}`);

  // Stash the current function's output cNames on the state so any
  // in-body `ReturnFromFunction` can emit `return <fn outputs>` with
  // the matching shape (bare value / array / nothing). Cleared after.
  const prevOutputs = state.currentFnOutputs;
  const prevOutputTypes = state.currentFnOutputTypes;
  state.currentFnOutputs = fn.cOutputs;
  state.currentFnOutputTypes = new Map(
    fn.cOutputs.map((c, i) => [c, fn.outputTypes[i]] as const)
  );
  const bodyLines = emitBody(fn.body, "  ", state);
  state.currentFnOutputs = prevOutputs;
  state.currentFnOutputTypes = prevOutputTypes;
  if (bodyLines.length > 0) lines.push(bodyLines);
  if (pollOnEntry || bodyLines.includes("_jit_cancel")) {
    lines.splice(pollAt, 0, "  const _jit_cancel = $cancel;");
  }

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
  let prevSrc: IRStmt["src"];
  for (const s of stmts) {
    // Statement location, as the interpreter sets `rt.$line` per
    // statement (mirrors the C `emitBody`).
    if (s.src !== undefined && s.src !== prevSrc) {
      out.push(`${indent}$loc = ${locCode(state, s.src)};`);
    }
    prevSrc = s.src ?? prevSrc;
    const line = emitStmt(s, indent, state);
    if (line.length > 0) out.push(line);
  }
  return out.join("\n");
}

/** Emit a JS RHS expression with deep-clone wrapping when the
 *  expression yields an owned-typed value via an alias path (Var,
 *  MemberLoad, HandleCaptureLoad). Mirrors the C
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
    useRuntimeByName(state, "jit_deep_clone");
    return `jit_deep_clone(${emitExpr(e, state)})`;
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
      const outTy = state.currentFnOutputTypes?.get(s.cName);
      if (outTy !== undefined && isOwned(outTy) && !isOwned(s.ty)) {
        // A scalar stored to an output whose (merged) type is a tensor,
        // e.g. a recursive function's scalar base case when the recursive
        // result widened the output: store a 1x1 tensor (as the C emitter).
        if (
          outTy.kind !== "Numeric" ||
          s.ty.kind !== "Numeric" ||
          outTy.isComplex ||
          s.ty.isComplex
        ) {
          throw new UnsupportedConstruct(
            `scalar assigned to the tensor output '${s.name}'`,
            s.span
          );
        }
        useRuntimeByName(state, "jit_tensor_alloc_nd");
        return `${indent}${s.cName} = jit_tensor_alloc_nd(2, [1, 1]); ${s.cName}.data[0] = ${rhs};`;
      }
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
      // Strip-mined as the `for` loop below, with a chunk counter.
      const label = enterLoop(state, isConstantTimeLoop(s.body, s.cond));
      const strip = label !== null;
      let ind = indent;
      if (strip) {
        lines.push(`${indent}${label}: while (${CANCEL_OK_JS}) {`);
        lines.push(
          `${indent}  for (let _jit_w = 0; _jit_w < ${STRIP_CHUNK}; _jit_w++) {`
        );
        lines.push(
          `${indent}    if (!(${truthy(s.cond, state)})) break ${label};`
        );
        ind += "    ";
      } else {
        lines.push(
          `${indent}while (${CANCEL_OK_JS} && (${truthy(s.cond, state)})) {`
        );
        ind += "  ";
      }
      const body = emitBody(s.body, ind, state);
      exitLoop(state);
      if (body.length > 0) lines.push(body);
      if (strip) lines.push(`${indent}  }`);
      lines.push(`${indent}}`);
      lines.push(`${indent}${CANCEL_POLL_JS}`);
      return lines.join("\n");
    }

    case "For": {
      // Mirror the C path's loop shape so JS output stays
      // bit-identical to numbl: snapshot start/end once at loop
      // entry, derive the iteration count via `jit_loop_count`,
      // and rebind the loop var via `jit_range_value` per iter.
      // This keeps the loop body insensitive to mid-body mutations
      // of `start`/`end` (matching MATLAB) and leaves the loop var
      // at its last actual iterated value after the loop ends.
      // `s.cVar` is pre-declared at function top by
      // `collectAssignedLocals` so reads after the loop see it.
      useRuntimeByName(state, "jit_loop_count");
      useRuntimeByName(state, "jit_range_value");
      const lines: string[] = [];
      const startE = emitExpr(s.start, state);
      const endE = emitExpr(s.end, state);
      const stepStr = formatJsNumber(s.step);
      lines.push(`${indent}{`);
      lines.push(`${indent}  const _jit_for_start = ${startE};`);
      lines.push(`${indent}  const _jit_for_end = ${endE};`);
      lines.push(
        `${indent}  const _jit_for_n = jit_loop_count(_jit_for_start, _jit_for_end, ${stepStr});`
      );
      // Strip-mined when its iterations take constant time (see
      // `cancelPoll.ts`).
      const label = enterLoop(state, isConstantTimeLoop(s.body));
      const strip = label !== null;
      let ind = indent + "  ";
      if (strip) {
        lines.push(`${indent}  let _jit_for_i = 0;`);
        lines.push(
          `${indent}  ${label}: while (_jit_for_i < _jit_for_n && ${CANCEL_OK_JS}) {`
        );
        lines.push(
          `${indent}    const _jit_for_m = Math.min(_jit_for_n, _jit_for_i + ${STRIP_CHUNK});`
        );
        lines.push(
          `${indent}    for (; _jit_for_i < _jit_for_m; _jit_for_i++) {`
        );
        ind += "  ";
      } else {
        lines.push(
          `${indent}  for (let _jit_for_i = 0; _jit_for_i < _jit_for_n && ${CANCEL_OK_JS}; _jit_for_i++) {`
        );
      }
      lines.push(
        `${ind}  ${s.cVar} = jit_range_value(_jit_for_start, ${stepStr}, _jit_for_end, _jit_for_n, _jit_for_i);`
      );
      const body = emitBody(s.body, ind + "  ", state);
      exitLoop(state);
      if (body.length > 0) lines.push(body);
      if (strip) lines.push(`${indent}    }`);
      lines.push(`${indent}  }`);
      lines.push(`${indent}  ${CANCEL_POLL_JS}`);
      lines.push(`${indent}}`);
      return lines.join("\n");
    }

    case "ReturnFromFunction":
      // Mid-body return: bail with whatever the declared outputs hold
      // right now. `emitFunction` stashes the active fn's cOutputs on
      // `state.currentFnOutputs` so the return form here matches the
      // function-end return (bare value / array / nothing).
      return emitReturnFromFunction(state, indent);

    case "Break": {
      // Leave both JS loops of a strip-mined loop.
      const label = breakLabel(state);
      return label === null ? `${indent}break;` : `${indent}break ${label};`;
    }

    case "Continue":
      return `${indent}continue;`;

    case "TypeComment": {
      // Debug aid — emit as a JS comment so the source stays readable.
      // The JIT's TypeComment can show multiple variables in one block;
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
      useRuntimeByName(state, "jit_scalar_index");
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
      // raises a bail (the array would grow, which the JIT does not
      // support) so the executor bails to the interpreter. See `scalar_index.js`.
      let offset: string;
      if (idxs.length === 1) {
        offset = `jit_idx_lin_grow_js(${baseName}, ${idxs[0]})`;
      } else {
        const terms: string[] = [];
        for (let i = 0; i < idxs.length; i++) {
          const checked = `jit_idx_axis_grow_js(${baseName}, ${i}, ${idxs[i]})`;
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
        // unparenthesized {re, im} producer like `jit_cmul(...)`.
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
  // Bounds and size-mismatch helpers (`scalar_index.js`).
  useRuntimeByName(state, "jit_scalar_index");
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
  // value or `_jit_rhs.data/imag[k]`) and writes both data + imag
  // lanes into `baseName` at the given dst offset. Complex base with
  // real RHS sets imag = 0; real base with complex RHS would drop the
  // imag lane (we route through `transfer` earlier, so this is rare).
  const writeAt = (dstOff: string, srcIdx: string): string => {
    if (rhsIsScalarReal) {
      const reLine = `${baseName}.data[${dstOff}] = _jit_rhs;`;
      const imLine = baseIsComplex ? ` ${baseName}.imag[${dstOff}] = 0;` : "";
      return reLine + imLine;
    }
    if (rhsIsScalarComplex) {
      const reLine = `${baseName}.data[${dstOff}] = _jit_rhs.re;`;
      const imLine = baseIsComplex
        ? ` ${baseName}.imag[${dstOff}] = _jit_rhs.im;`
        : "";
      return reLine + imLine;
    }
    // Tensor RHS: read one element from rhs.data (and rhs.imag when
    // the rhs is complex).
    const reLine = `${baseName}.data[${dstOff}] = _jit_rhs.data[${srcIdx}];`;
    if (!baseIsComplex) return reLine;
    const imSrc = rhsIsComplex ? `_jit_rhs.imag[${srcIdx}]` : `0`;
    return reLine + ` ${baseName}.imag[${dstOff}] = ${imSrc};`;
  };

  // Single-slot linear store.
  if (s.index.length === 1) {
    const slot = s.index[0];
    if (slot.kind === "Colon") {
      return (
        `${indent}{ ` +
        `const _jit_rhs = ${rhsExpr}; ` +
        `const _jit_n = ${baseName}.data.length; ` +
        (rhsIsScalar
          ? ``
          : `if (_jit_rhs.data.length !== _jit_n) jit_numel_mismatch(); `) +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `${writeAt("_jit_k", "_jit_k")} ` +
        `} ` +
        `}`
      );
    }
    if (slot.kind === "Range") {
      useRuntimeByName(state, "jit_loop_count");
      useRuntimeByName(state, "jit_range_value");
      useRuntimeByName(state, "jit_scalar_index");
      const sE = emitExpr(slot.start, state);
      const stE = emitExpr(slot.step, state);
      const eE = emitExpr(slot.end, state);
      // Tensor RHS: enforce numbl's `numel(src) == range_len`
      // runtime check so a mismatched-length assignment throws
      // instead of silently truncating.
      const lenCheck = rhsIsScalar
        ? ``
        : `if (_jit_rhs.data.length !== _jit_n) jit_numel_mismatch(); `;
      // Each target offset goes through the grow-aware checker: a sub-1
      // index errors, and an index past the end grow-bails to the
      // interpreter (which has full grow semantics) — matching the
      // canonical `v(end+1:end+k) = ...` append, which previously
      // silently dropped the out-of-range writes here.
      return (
        `${indent}{ ` +
        `const _jit_rhs = ${rhsExpr}; ` +
        `const _jit_s = ${sE}, _jit_e = ${eE}, _jit_st = ${stE}; ` +
        `const _jit_n = jit_loop_count(_jit_s, _jit_e, _jit_st); ` +
        lenCheck +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `const _jit_v = jit_range_value(_jit_s, _jit_st, _jit_e, _jit_n, _jit_k); ` +
        `const _jit_off = jit_idx_lin_grow_js(${baseName}, _jit_v); ` +
        `${writeAt("_jit_off", "_jit_k")} ` +
        `} ` +
        `}`
      );
    }
    if (slot.kind === "IndexVec") {
      useRuntimeByName(state, "jit_scalar_index");
      const idxE = emitExpr(slot.expr, state);
      // Tensor RHS must match the number of selected positions; a
      // mismatch is a "Subscripted assignment dimension mismatch" in the
      // interpreter (--opt 0). Without the check the gather reads past
      // the RHS end → silent NaN/truncation.
      const lenCheck = rhsIsScalar
        ? ``
        : `if (_jit_rhs.data.length !== _jit_n) jit_numel_mismatch(); `;
      return (
        `${indent}{ ` +
        `const _jit_rhs = ${rhsExpr}; ` +
        `const _jit_ix = ${idxE}; ` +
        `const _jit_ixd = _jit_ix.jitTag === "tensor" ? _jit_ix.data : [_jit_ix]; ` +
        `const _jit_n = _jit_ixd.length; ` +
        lenCheck +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `const _jit_off = jit_idx_lin_grow_js(${baseName}, _jit_ixd[_jit_k]); ` +
        `${writeAt("_jit_off", "_jit_k")} ` +
        `} ` +
        `}`
      );
    }
    if (slot.kind === "LogicalMask") {
      // Linear single-slot logical mask write: scan, collect 0-based
      // truthy positions, write the RHS into each. Scalar RHS
      // broadcasts; tensor RHS reads positionally.
      const maskE = emitExpr(slot.expr, state);
      // Tensor RHS must have exactly as many elements as truthy mask
      // positions; otherwise the interpreter (--opt 0) raises
      // "Subscripted assignment dimension mismatch". Without this check
      // the per-slot write reads past the RHS end (short RHS → NaN) or
      // ignores trailing RHS elements (long RHS → silent truncation).
      const lenCheck = rhsIsScalar
        ? ``
        : `if (_jit_rhs.data.length !== _jit_n) jit_numel_mismatch(); `;
      return (
        `${indent}{ ` +
        `const _jit_rhs = ${rhsExpr}; ` +
        `const _jit_m = ${maskE}; ` +
        `const _jit_md = _jit_m.data; ` +
        `const _jit_ix = []; ` +
        `for (let _jit_mi = 0; _jit_mi < _jit_md.length; _jit_mi++) ` +
        `if (_jit_md[_jit_mi] !== 0) _jit_ix.push(_jit_mi); ` +
        `const _jit_n = _jit_ix.length; ` +
        lenCheck +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `${writeAt("_jit_ix[_jit_k]", "_jit_k")} ` +
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
  const setup: string[] = [`const _jit_rhs = ${rhsExpr};`];
  const idxFns: string[] = [];
  const dims: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const slot = s.index[i];
    if (slot.kind === "Colon") {
      setup.push(`const _jit_n_${i} = ${baseName}.shape[${i}] ?? 1;`);
      dims.push(`_jit_n_${i}`);
      idxFns.push(`(_jit_k_${i} + 1)`);
    } else if (slot.kind === "Range") {
      useRuntimeByName(state, "jit_loop_count");
      useRuntimeByName(state, "jit_range_value");
      const sE = emitExpr(slot.start, state);
      const stE = emitExpr(slot.step, state);
      const eE = emitExpr(slot.end, state);
      setup.push(
        `const _jit_s_${i} = ${sE}, _jit_e_${i} = ${eE}, _jit_st_${i} = ${stE};`
      );
      setup.push(
        `const _jit_n_${i} = jit_loop_count(_jit_s_${i}, _jit_e_${i}, _jit_st_${i});`
      );
      dims.push(`_jit_n_${i}`);
      idxFns.push(
        `jit_range_value(_jit_s_${i}, _jit_st_${i}, _jit_e_${i}, _jit_n_${i}, _jit_k_${i})`
      );
    } else if (slot.kind === "Scalar") {
      const v = emitExpr(slot.expr, state);
      setup.push(`const _jit_s_${i} = ${v};`);
      setup.push(`const _jit_n_${i} = 1;`);
      dims.push(`1`);
      idxFns.push(`_jit_s_${i}`);
    } else if (slot.kind === "IndexVec") {
      const idxE = emitExpr(slot.expr, state);
      setup.push(
        `const _jit_ix_${i} = ${idxE}; ` +
          `const _jit_ixd_${i} = _jit_ix_${i}.jitTag === "tensor" ? _jit_ix_${i}.data : [_jit_ix_${i}];`
      );
      setup.push(`const _jit_n_${i} = _jit_ixd_${i}.length;`);
      dims.push(`_jit_n_${i}`);
      idxFns.push(`_jit_ixd_${i}[_jit_k_${i}]`);
    } else if (slot.kind === "LogicalMask") {
      const maskE = emitExpr(slot.expr, state);
      setup.push(
        `const _jit_m_${i} = ${maskE}; ` +
          `const _jit_md_${i} = _jit_m_${i}.data; ` +
          `const _jit_ixd_${i} = []; ` +
          `for (let _jit_mi = 0; _jit_mi < _jit_md_${i}.length; _jit_mi++) ` +
          `if (_jit_md_${i}[_jit_mi] !== 0) _jit_ixd_${i}.push(_jit_mi + 1);`
      );
      setup.push(`const _jit_n_${i} = _jit_ixd_${i}.length;`);
      dims.push(`_jit_n_${i}`);
      idxFns.push(`_jit_ixd_${i}[_jit_k_${i}]`);
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
  // Tensor RHS: its non-singleton dimensions must match the selection's
  // (the interpreter's rule), so a mismatched assignment raises instead of
  // silently truncating or padding.
  if (!rhsIsScalar) {
    lines.push(
      `${indent}  jit_store_shape_check(${baseName}, [${dims.join(", ")}], _jit_rhs);`
    );
  }
  for (let i = ndim - 1; i >= 0; i--) {
    lines.push(
      `${indent}  for (let _jit_k_${i} = 0; _jit_k_${i} < _jit_n_${i}; _jit_k_${i}++) {`
    );
  }
  // Each axis index goes through the grow-aware checker: a sub-1 index
  // errors and an index past that axis's extent grow-bails to the
  // interpreter (which grows), instead of silently dropping the write.
  useRuntimeByName(state, "jit_scalar_index");
  for (let i = 0; i < ndim; i++) {
    lines.push(
      `${indent}    const _jit_ai_${i} = jit_idx_axis_grow_js(${baseName}, ${i}, ${idxFns[i]});`
    );
  }
  const srcTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++)
      strideParts.push(`(${baseName}.shape[${j}] ?? 1)`);
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    srcTerms.push(`_jit_ai_${i} * ${stride}`);
  }
  const rhsTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++) strideParts.push(dims[j]);
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    rhsTerms.push(`_jit_k_${i} * ${stride}`);
  }
  lines.push(`${indent}    const _jit_dst = ${srcTerms.join(" + ")};`);
  if (rhsIsScalar) {
    lines.push(`${indent}    ${writeAt("_jit_dst", "0")}`);
  } else {
    lines.push(`${indent}    const _jit_src = ${rhsTerms.join(" + ")};`);
    lines.push(`${indent}    ${writeAt("_jit_dst", "_jit_src")}`);
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

  const b = getBuiltin(s.name);
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
      // `{re, im}` (matching the `jit_cscalar.js` snippet's helpers).
      // Activate it lazily so the call site can rely on it being
      // available; the actual helper functions get inlined alongside.
      return `{re: 0, im: ${formatJsNumber(e.value)}}`;

    case "StringLit": {
      // JS runtime value conventions: `String` → bare JS string;
      // `Char` → `{jitTag:"char", value:"..."}` wrapper.
      const lit = JSON.stringify(e.value);
      if (e.ty.kind === "Char") {
        return `({jitTag: "char", value: ${lit}})`;
      }
      return lit;
    }

    case "Var":
      // A read of a local that may still be unassigned (`undefined` in
      // JS): raise rather than compute with `undefined`.
      if (state.maybeUnbound?.has(e)) return `jit_bound(${e.cName})`;
      return e.cName;

    case "Binary": {
      const b = getBuiltin(e.builtin);
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
      const b = getBuiltin(e.builtin);
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
      const b = getBuiltin(e.name);
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
      useRuntimeByName(state, "jit_tensor_make_range");
      return `jit_tensor_make_range(${startE}, ${stepE}, ${endE})`;
    }

    case "TensorBuild": {
      const [rows, cols] = e.shape;
      const ty = e.ty;
      if (ty.kind === "Numeric" && ty.isComplex) {
        // For each element, split into a real / imag pair using the
        // element's static type. Statically-real elements pad imag
        // with 0; statically-complex elements emit `.re` / `.im`.
        useRuntimeByName(state, "jit_cscalar");
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
          useRuntimeByName(state, "jit_tensor_from_row_complex");
          return `jit_tensor_from_row_complex([${res.join(", ")}], [${ims.join(", ")}], ${cols})`;
        }
        useRuntimeByName(state, "jit_tensor_from_matrix_complex");
        return `jit_tensor_from_matrix_complex([${res.join(", ")}], [${ims.join(", ")}], ${rows}, ${cols})`;
      }
      const flat = e.elements.map(el => emitExpr(el, state)).join(", ");
      if (rows === 1) {
        useRuntimeByName(state, "jit_tensor_from_row");
        return `jit_tensor_from_row([${flat}], ${cols})`;
      }
      useRuntimeByName(state, "jit_tensor_from_matrix");
      return `jit_tensor_from_matrix([${flat}], ${rows}, ${cols})`;
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
      useRuntimeByName(state, "jit_scalar_index");
      const baseName = e.base.cName;
      const idxs = e.indices.map(ix => emitExpr(ix, state));
      let offset: string;
      if (idxs.length === 1) {
        offset = `jit_idx_lin_js(${baseName}, ${idxs[0]})`;
      } else {
        const terms: string[] = [];
        for (let i = 0; i < idxs.length; i++) {
          const checked = `jit_idx_axis_js(${baseName}, ${i}, ${idxs[i]})`;
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
        useRuntimeByName(state, "jit_cscalar");
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
      // `{a, b, c}` → fresh `jit_cell_make`-built cell. Slot values
      // are owned-consumed; route owned aliases through deep_clone so
      // the cell owns its own copy (mirrors the C consume rule).
      useRuntimeByName(state, "jit_cell_make");
      const slots = e.elements
        .map(el =>
          isOwned(el.ty) ? emitOwnedRhsJs(el, state) : emitExpr(el, state)
        )
        .join(", ");
      const shapeLit = JSON.stringify(e.shape);
      return `jit_cell_make([${slots}], ${shapeLit})`;
    }

    case "CellEmpty": {
      // `cell(n, m, ...)` → fresh cell with empty-double slots.
      // The 1-arg square form expands to two dim args at lowering
      // time; everything here is the resolved per-axis form.
      useRuntimeByName(state, "jit_cell_empty");
      const dimsJs = e.dims.map(d => emitExpr(d, state)).join(", ");
      return `jit_cell_empty([${dimsJs}])`;
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
  // Index conversion and error helpers (`scalar_index.js`).
  useRuntimeByName(state, "jit_scalar_index");
  if (e.base.kind !== "Var") {
    throw new Error(
      `emitJs internal: IndexSlice base must be a Var after ANF (got ${e.base.kind})`
    );
  }
  const isComplexBase = e.base.ty.kind === "Numeric" && e.base.ty.isComplex;
  const allocFn = isComplexBase
    ? "jit_tensor_alloc_nd_complex"
    : "jit_tensor_alloc_nd";
  useRuntimeByName(state, allocFn);
  const baseName = e.base.cName;
  // Per-element copy template — fills both lanes for complex bases.
  // Used inside every per-slot loop body so the same emit path serves
  // real and complex slices.
  const copy = (dstIdx: string, srcIdx: string): string =>
    isComplexBase
      ? `_jit_t.data[${dstIdx}] = ${baseName}.data[${srcIdx}]; _jit_t.imag[${dstIdx}] = ${baseName}.imag[${srcIdx}];`
      : `_jit_t.data[${dstIdx}] = ${baseName}.data[${srcIdx}];`;

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
        useRuntimeByName(state, "jit_loop_count");
        useRuntimeByName(state, "jit_range_value");
        const s = emitExpr(slot.start, state);
        const st = emitExpr(slot.step, state);
        const en = emitExpr(slot.end, state);
        off =
          `(() => { const _jit_s = ${s}, _jit_e = ${en}, _jit_st = ${st}; ` +
          `const _jit_n = jit_loop_count(_jit_s, _jit_e, _jit_st); ` +
          `const _jit_idx = jit_to_index(jit_range_value(_jit_s, _jit_st, _jit_e, _jit_n, 0)); ` +
          `const _jit_len = ${baseName}.data.length; ` +
          `if (_jit_idx < 1 || _jit_idx > _jit_len) jit_index_error(0, _jit_idx, _jit_len); ` +
          `return _jit_idx - 1; })()`;
      } else if (slot.kind === "IndexVec") {
        useRuntimeByName(state, "jit_scalar_index");
        const idx = emitExpr(slot.expr, state);
        off =
          `(() => { const _jit_ix = ${idx}; ` +
          `const _jit_ixd = _jit_ix.jitTag === "tensor" ? _jit_ix.data : [_jit_ix]; ` +
          `return jit_idx_lin_js(${baseName}, _jit_ixd[0]); })()`;
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
        `const _jit_n = ${baseName}.data.length; ` +
        `const _jit_t = ${allocFn}(2, [_jit_n, 1]); ` +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `${copy("_jit_k", "_jit_k")} ` +
        `} ` +
        `return _jit_t; ` +
        `})()`
      );
    }
    if (slot.kind === "Range") {
      useRuntimeByName(state, "jit_loop_count");
      useRuntimeByName(state, "jit_range_value");
      const s = emitExpr(slot.start, state);
      const st = emitExpr(slot.step, state);
      const en = emitExpr(slot.end, state);
      const rows = isColVec ? "_jit_n" : "1";
      const cols = isColVec ? "1" : "_jit_n";
      // Per-element 1-based bounds check against the source's linear
      // length. Mirrors numbl's RuntimeError on out-of-range slice
      // indices (e.g. `oob(0:3)` → index 0 fails the lower-bound
      // guard).
      return (
        `(() => { ` +
        `const _jit_s = ${s}; const _jit_e = ${en}; const _jit_st = ${st}; ` +
        `const _jit_n = jit_loop_count(_jit_s, _jit_e, _jit_st); ` +
        `const _jit_t = ${allocFn}(2, [${rows}, ${cols}]); ` +
        `const _jit_len = ${baseName}.data.length; ` +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `const _jit_v = jit_range_value(_jit_s, _jit_st, _jit_e, _jit_n, _jit_k); ` +
        `const _jit_idx = jit_to_index(_jit_v); ` +
        `if (_jit_idx < 1 || _jit_idx > _jit_len) jit_index_error(0, _jit_idx, _jit_len); ` +
        `${copy("_jit_k", "_jit_idx - 1")} ` +
        `} ` +
        `return _jit_t; ` +
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
      useRuntimeByName(state, "jit_scalar_index");
      const idx = emitExpr(slot.expr, state);
      const rows = isRowVec ? "1" : "_jit_n";
      const cols = isRowVec ? "_jit_n" : "1";
      return (
        `(() => { ` +
        `const _jit_ix = ${idx}; ` +
        `const _jit_ixd = _jit_ix.jitTag === "tensor" ? _jit_ix.data : [_jit_ix]; ` +
        `const _jit_n = _jit_ixd.length; ` +
        `const _jit_t = ${allocFn}(2, [${rows}, ${cols}]); ` +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `${copy("_jit_k", `jit_idx_lin_js(${baseName}, _jit_ixd[_jit_k])`)} ` +
        `} ` +
        `return _jit_t; ` +
        `})()`
      );
    }
    if (slot.kind === "LogicalMask") {
      // Single-slot logical mask: linear scan, collect 1-based
      // positions of truthy elements, then gather. Result orientation
      // follows the base: row base → row, otherwise column.
      const mask = emitExpr(slot.expr, state);
      const rows = isRowVec ? "1" : "_jit_n";
      const cols = isRowVec ? "_jit_n" : "1";
      return (
        `(() => { ` +
        `const _jit_m = ${mask}; ` +
        `const _jit_md = _jit_m.data; ` +
        `const _jit_blen = ${baseName}.data.length; ` +
        `const _jit_ix = []; ` +
        `for (let _jit_mi = 0; _jit_mi < _jit_md.length; _jit_mi++) ` +
        `if (_jit_md[_jit_mi] !== 0) { ` +
        // A truthy mask bit past the base end is an error in MATLAB (the C
        // path raises too); without this the gather reads OOB -> NaN.
        `if (_jit_mi >= _jit_blen) jit_mask_oob(0); ` +
        `_jit_ix.push(_jit_mi); } ` +
        `const _jit_n = _jit_ix.length; ` +
        `const _jit_t = ${allocFn}(2, [${rows}, ${cols}]); ` +
        `for (let _jit_k = 0; _jit_k < _jit_n; _jit_k++) { ` +
        `${copy("_jit_k", "_jit_ix[_jit_k]")} ` +
        `} ` +
        `return _jit_t; ` +
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
      setup.push(`const _jit_n_${i} = ${baseName}.shape[${i}] ?? 1;`);
      dims.push(`_jit_n_${i}`);
      idxFns.push(`(_jit_k_${i} + 1)`);
    } else if (slot.kind === "Range") {
      useRuntimeByName(state, "jit_loop_count");
      useRuntimeByName(state, "jit_range_value");
      const s = emitExpr(slot.start, state);
      const st = emitExpr(slot.step, state);
      const en = emitExpr(slot.end, state);
      setup.push(
        `const _jit_s_${i} = ${s}, _jit_e_${i} = ${en}, _jit_st_${i} = ${st};`
      );
      setup.push(
        `const _jit_n_${i} = jit_loop_count(_jit_s_${i}, _jit_e_${i}, _jit_st_${i});`
      );
      dims.push(`_jit_n_${i}`);
      idxFns.push(
        `jit_range_value(_jit_s_${i}, _jit_st_${i}, _jit_e_${i}, _jit_n_${i}, _jit_k_${i})`
      );
    } else if (slot.kind === "Scalar") {
      const v = emitExpr(slot.expr, state);
      setup.push(`const _jit_s_${i} = ${v};`);
      setup.push(`const _jit_n_${i} = 1;`);
      dims.push(`1`);
      idxFns.push(`_jit_s_${i}`);
    } else if (slot.kind === "IndexVec") {
      const idxE = emitExpr(slot.expr, state);
      setup.push(
        `const _jit_ix_${i} = ${idxE}; ` +
          `const _jit_ixd_${i} = _jit_ix_${i}.jitTag === "tensor" ? _jit_ix_${i}.data : [_jit_ix_${i}];`
      );
      setup.push(`const _jit_n_${i} = _jit_ixd_${i}.length;`);
      dims.push(`_jit_n_${i}`);
      idxFns.push(`_jit_ixd_${i}[_jit_k_${i}]`);
    } else if (slot.kind === "LogicalMask") {
      // Multi-slot logical mask: scan, collect 0-based truthy
      // positions along this axis. The selected axis size is the
      // truthy count; each iteration of `_jit_k_${i}` picks the
      // k-th truthy position (1-based at idxFn-time).
      const maskE = emitExpr(slot.expr, state);
      setup.push(
        `const _jit_m_${i} = ${maskE}; ` +
          `const _jit_md_${i} = _jit_m_${i}.data; ` +
          `const _jit_ixd_${i} = []; ` +
          `for (let _jit_mi = 0; _jit_mi < _jit_md_${i}.length; _jit_mi++) ` +
          `if (_jit_md_${i}[_jit_mi] !== 0) _jit_ixd_${i}.push(_jit_mi + 1);`
      );
      setup.push(`const _jit_n_${i} = _jit_ixd_${i}.length;`);
      dims.push(`_jit_n_${i}`);
      idxFns.push(`_jit_ixd_${i}[_jit_k_${i}]`);
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
    for (let i = 0; i < ndim; i++) slines.push(`const _jit_k_${i} = 0;`);
    for (let i = 0; i < ndim; i++) {
      slines.push(`const _jit_idx_${i} = jit_to_index(${idxFns[i]});`);
      slines.push(
        `if (_jit_idx_${i} < 1 || _jit_idx_${i} > (${baseName}.shape[${i}] ?? 1)) ` +
          `jit_index_error(${i + 1}, _jit_idx_${i}, ${baseName}.shape[${i}] ?? 1);`
      );
    }
    const srcTerms: string[] = [];
    for (let i = 0; i < ndim; i++) {
      const strideParts: string[] = [];
      for (let j = 0; j < i; j++)
        strideParts.push(`(${baseName}.shape[${j}] ?? 1)`);
      const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
      srcTerms.push(`(_jit_idx_${i} - 1) * ${stride}`);
    }
    slines.push(`const _jit_off = ${srcTerms.join(" + ")};`);
    if (isComplexBase) {
      slines.push(
        `return {re: ${baseName}.data[_jit_off], im: ${baseName}.imag[_jit_off]};`
      );
    } else {
      slines.push(`return ${baseName}.data[_jit_off];`);
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
    `const _jit_t = ${allocFn}(${resultDims.length}, [${resultDims.join(", ")}]);`
  );
  for (let i = ndim - 1; i >= 0; i--) {
    lines.push(
      `for (let _jit_k_${i} = 0; _jit_k_${i} < _jit_n_${i}; _jit_k_${i}++) {`
    );
  }
  // Per-axis 1-based index, rounded, with a bounds check against that
  // axis's extent — mirrors the interpreter's "Index in position N
  // exceeds array bounds" RuntimeError. Without it, an out-of-range
  // axis index reads `undefined` from the buffer and silently yields
  // NaN. Colon slots are always in range; the check is cheap and
  // uniform across slot kinds (Range / Scalar / IndexVec / mask).
  for (let i = 0; i < ndim; i++) {
    lines.push(`const _jit_idx_${i} = jit_to_index(${idxFns[i]});`);
    lines.push(
      `if (_jit_idx_${i} < 1 || _jit_idx_${i} > (${baseName}.shape[${i}] ?? 1)) ` +
        `jit_index_error(${i + 1}, _jit_idx_${i}, ${baseName}.shape[${i}] ?? 1);`
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
    srcTerms.push(`(_jit_idx_${i} - 1) * ${stride}`);
  }
  // Column-major destination offset using the result's own dims.
  const dstTerms: string[] = [];
  for (let i = 0; i < ndim; i++) {
    const strideParts: string[] = [];
    for (let j = 0; j < i; j++) strideParts.push(dims[j]);
    const stride = strideParts.length === 0 ? "1" : strideParts.join(" * ");
    dstTerms.push(`_jit_k_${i} * ${stride}`);
  }
  lines.push(`const _jit_src_off = ${srcTerms.join(" + ")};`);
  lines.push(`const _jit_dst_off = ${dstTerms.join(" + ")};`);
  lines.push(copy("_jit_dst_off", "_jit_src_off"));
  for (let i = ndim - 1; i >= 0; i--) lines.push(`}`);
  lines.push(`return _jit_t;`);
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
 *  through `jit_cnonzero` — a `{re: 0, im: 0}` object is "truthy" in
 *  JS as a non-null reference, which would loop forever in `while (z)`. */
function truthy(e: IRExpr, state: RuntimeState): string {
  const expr = emitExpr(e, state);
  if (e.ty.kind === "Numeric") {
    if (e.ty.isComplex) {
      useRuntimeByName(state, "jit_cscalar");
      return `jit_cnonzero(${expr})`;
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
  return `_jit_discard_${slotIdx}`;
}

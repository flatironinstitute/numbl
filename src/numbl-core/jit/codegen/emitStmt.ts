/**
 * Function-shape + statement-level codegen — `emitFunction`,
 * `emitBody`, `emitStmt`, and the locals-collection / param-list
 * helpers they share. Statement emit is mutually recursive with body
 * emit (If/While/For wrap nested bodies), so they live together.
 *
 * Expression-side emit lives in `emitExpr.ts`; index slice store /
 * scalar offset and named-typedef helpers are in their own modules.
 */

import type { IRExpr, IRFunc, IRStmt } from "../lowering/ir.js";
import {
  isNumeric,
  isOwned,
  scalarDouble,
  typeToString,
  type Type,
} from "../lowering/types.js";
import { cTypeFor, formatDouble, requireOwnedHelpers } from "./cHelpers.js";
import {
  lookupBuiltin,
  makeEmitUseRuntime,
  useRuntimeByName,
  type RuntimeState,
} from "./runtime.js";
import { requireEmitC } from "../builtins/registry.js";
import {
  computeFutureTouches,
  earlyFreeCandidates,
  type FutureTouchMap,
} from "./liveness.js";
import { irFuncDocComment, irStmtHeader } from "./prettyIR.js";
import { emitTensorAssignFused, isFusableAssign } from "./emitTensorFused.js";
import { emitIndexSliceStore, emitNdScalarOffset } from "./emitIndex.js";
import { emitCondToBoolExpr, emitExpr, emitOwnedRhs } from "./emitExpr.js";
import { activateOwnedRuntime, activatedOwnedHelpers } from "./emit.js";

/** Default initializer for a freshly-declared non-owned local. Owned
 *  types (tensors, structs, classes, handles) route through their
 *  `_empty()` helper instead and never hit this path; everything
 *  reaching here is scalar real numeric, so `0.0` always fits. */
export function defaultInitFor(): string {
  return "0.0";
}

export function fnRetType(fn: IRFunc): string {
  // 0 outputs → C `void`. 1 output → classic return-by-value. N≥2
  // outputs → `void` return + out-pointer params (see `fnParamList`).
  if (fn.outputs.length !== 1) return "void";
  const t = fn.outputTypes[0];
  if (!t) return "double";
  return cTypeFor(t);
}

export function fnParamList(fn: IRFunc): string {
  const parts: string[] = [];
  for (let i = 0; i < fn.cParams.length; i++) {
    parts.push(`${cTypeFor(fn.paramTypes[i])} ${fn.cParams[i]}`);
  }
  // Multi-output convention: append one `T_i *_mtoc2_o<i>` per output
  // after the user params, in declaration order. The body's sret
  // writes (emitted at every return point and at the function-end
  // fall-through) target these pointers.
  if (fn.outputs.length >= 2) {
    for (let i = 0; i < fn.outputs.length; i++) {
      const ty = fn.outputTypes[i] ?? { kind: "Unknown" as const };
      parts.push(`${cTypeFor(ty)} *_mtoc2_o${i}`);
    }
  }
  return parts.length === 0 ? "void" : parts.join(", ");
}

/** Walk the body and collect every local (any kind) that needs a top-
 *  of-function predeclaration. Includes Assign LHSes, MultiAssignCall
 *  slot bindings, and For loop vars (the For node carries its own
 *  loop var so reads after the loop see the last iteration's value).
 *  Callers partition the result into owned / non-owned with `isOwned`. */
function collectAllLocals(stmts: IRStmt[]): { cName: string; ty: Type }[] {
  const seen = new Map<string, Type>();
  const order: string[] = [];
  const note = (cName: string, ty: Type): void => {
    if (seen.has(cName)) return;
    seen.set(cName, ty);
    order.push(cName);
  };
  const visit = (ss: IRStmt[]): void => {
    for (const s of ss) {
      switch (s.kind) {
        case "Assign":
          note(s.cName, s.ty);
          break;
        case "MultiAssignCall":
          // v1 restricts slots to scalar real numeric, so owned-typed
          // bindings won't appear here today; kept for the future
          // tensor-output extension.
          for (const slot of s.outputs) {
            if (slot.binding !== null) note(slot.binding.cName, slot.ty);
          }
          break;
        case "If":
          visit(s.thenBody);
          visit(s.elseBody);
          break;
        case "While":
          visit(s.body);
          break;
        case "For":
          note(s.cVar, scalarDouble());
          visit(s.body);
          break;
      }
    }
  };
  visit(stmts);
  return order.map(cName => ({ cName, ty: seen.get(cName)! }));
}

/** Walk `stmts` once and split the result into owned vs non-owned
 *  locals. Every local is pre-declared at function top — owned with
 *  the per-type `_empty()` helper, scalars with the default init
 *  (`0.0` / `0`). The non-owned half also picks up For loop vars and
 *  multi-assign slot bindings so that a later read of a variable
 *  first written inside an If / While / For body (reachable via
 *  `mergeBranchEnvs` keeping partial-branch keys) doesn't reference
 *  an out-of-scope C local. */
export function collectLocals(stmts: IRStmt[]): {
  owned: { cName: string; ty: Type }[];
  scalars: { cName: string; ty: Type }[];
} {
  const owned: { cName: string; ty: Type }[] = [];
  const scalars: { cName: string; ty: Type }[] = [];
  for (const local of collectAllLocals(stmts)) {
    if (isOwned(local.ty)) owned.push(local);
    else scalars.push(local);
  }
  return { owned, scalars };
}

/** Emits the function/scope's exit cleanup + return statements at a
 *  given indent. Called from two places: at every `ReturnFromFunction`
 *  site (inline cleanup — no goto), and at the fall-through bottom of
 *  the function. Single source of truth so the two callers can't
 *  drift on which locals get freed in what order.
 *
 *  Why inline rather than goto-to-a-shared-tail: c2js can't translate
 *  C `goto`, and we want a single C emit that works for both targets.
 *  Inlining produces some duplication when a function has many early
 *  returns and many owned locals, but typical mtoc2 functions are
 *  small and the C compiler dedupes-by-tail-merge anyway. */
export type EmitReturnTail = (indent: string) => string;

export function emitFunction(
  fn: IRFunc,
  state: RuntimeState,
  /** Storage-class qualifier on the function definition. Default is
   *  `static` (file-scope, the C-AOT whole-program convention). Pass
   *  `""` to omit the qualifier — the function then gets external
   *  linkage so a JIT host can resolve it via `dlsym` / koffi after
   *  `dlopen`'ing the compiled `.so`. */
  storageClass: "static" | "" = "static"
): string {
  const lines: string[] = [];
  const retType = fnRetType(fn);
  const nOutputs = fn.outputs.length;
  const isVoidFn = nOutputs === 0;
  const isMulti = nOutputs >= 2;
  lines.push(irFuncDocComment(fn));
  const sc = storageClass === "" ? "" : `${storageClass} `;
  lines.push(`${sc}${retType} ${fn.cName}(${fnParamList(fn)}) {`);
  const paramNames = new Set(fn.cParams);
  // Pre-declare each declared output slot (skip any that share a
  // C-name with a param — a vacuous shadowing case kept for symmetry
  // with the existing single-output code path).
  const outputCNames = new Set<string>();
  for (let i = 0; i < nOutputs; i++) {
    const cOut = fn.cOutputs[i];
    const outTy = fn.outputTypes[i];
    outputCNames.add(cOut);
    if (paramNames.has(cOut)) continue;
    if (outTy && isOwned(outTy)) {
      const h = activatedOwnedHelpers(outTy, state);
      lines.push(`  ${cTypeFor(outTy)} ${cOut} = ${h.empty}();`);
    } else if (outTy !== undefined) {
      lines.push(`  ${cTypeFor(outTy)} ${cOut} = ${defaultInitFor()};`);
    } else {
      lines.push(`  double ${cOut} = 0.0;`);
    }
  }
  // Pre-declare every body local at function top (excluding outputs
  // and params, both of which are already declared elsewhere —
  // outputs above, params in the function signature). Without the
  // paramNames filter, a body that reassigns an owned param
  // (`function f(xs); xs = xs(:); ...`) would emit a duplicate
  // `mtoc2_tensor_t xs = mtoc2_tensor_empty();` and the C compiler
  // would reject the redeclaration.
  const isLocalDeclSite = (o: { cName: string }): boolean =>
    !outputCNames.has(o.cName) && !paramNames.has(o.cName);
  const fnLocals = collectLocals(fn.body);
  const owned = fnLocals.owned.filter(isLocalDeclSite);
  for (const o of owned) {
    const h = activatedOwnedHelpers(o.ty, state);
    lines.push(`  ${cTypeFor(o.ty)} ${o.cName} = ${h.empty}();`);
  }
  for (const o of fnLocals.scalars.filter(isLocalDeclSite)) {
    lines.push(`  ${cTypeFor(o.ty)} ${o.cName} = ${defaultInitFor()};`);
  }
  // Keep each owned output alive through the body so the future-touch
  // analysis doesn't emit a stray early-free of the value we're about
  // to return (single-output) or transfer via sret (multi-output).
  // Zero-output / scalar-only-output functions contribute nothing.
  const ownedOutputsForLiveness: { cName: string; ty: Type }[] = [];
  for (let i = 0; i < nOutputs; i++) {
    const outTy = fn.outputTypes[i];
    if (outTy !== undefined && isOwned(outTy)) {
      ownedOutputsForLiveness.push({ cName: fn.cOutputs[i], ty: outTy });
    }
  }
  const futureTouches = computeFutureTouches(fn.body, ownedOutputsForLiveness);
  const fnOwnedTypes = new Map<string, Type>();
  for (const o of owned) fnOwnedTypes.set(o.cName, o.ty);
  for (let i = 0; i < nOutputs; i++) {
    const outTy = fn.outputTypes[i];
    if (outTy && isOwned(outTy)) {
      fnOwnedTypes.set(fn.cOutputs[i], outTy);
    }
  }
  // Owned-typed params: the caller wrapped them in `_copy` so the
  // callee owns its arg. Track them so early-free / scope-exit free
  // can pick the right helper. They're NOT pre-declared (the
  // function signature already declared them) but they ARE freed at
  // scope exit.
  const ownedParams: { cName: string; ty: Type }[] = [];
  for (let i = 0; i < fn.cParams.length; i++) {
    const pTy = fn.paramTypes[i];
    if (isOwned(pTy)) {
      ownedParams.push({ cName: fn.cParams[i], ty: pTy });
      fnOwnedTypes.set(fn.cParams[i], pTy);
    }
  }
  // Tail closure: sret writes (multi-output) + scope-exit frees +
  // `return [value];`. Called inline at every `ReturnFromFunction`
  // site and at the fall-through bottom. Single source of truth so
  // the two callers can't drift on cleanup order. Activate the
  // owned `_free` (and multi-output `_assign`) snippets eagerly so
  // they ship even when every path is an early return.
  for (const o of [...owned, ...ownedParams]) activateOwnedRuntime(o.ty, state);
  if (isMulti) {
    for (let i = 0; i < nOutputs; i++) {
      const outTy = fn.outputTypes[i];
      if (outTy && isOwned(outTy)) activateOwnedRuntime(outTy, state);
    }
  }
  const fnReturnTail: EmitReturnTail = indent => {
    const tail: string[] = [];
    // Multi-output: write each output through its sret pointer
    // BEFORE the scope-exit free walk so the writes can read the
    // live locals before any tensor frees would zero them.
    //   Scalar slots: bare struct copy (`*_mtoc2_o<i> = <local>`).
    //   Owned slots:  ownership transfer via the kind's `_assign`
    //                 helper. The local's scope-exit free is
    //                 suppressed below (`outputCNames` skip).
    if (isMulti) {
      for (let i = 0; i < nOutputs; i++) {
        const outTy = fn.outputTypes[i];
        if (outTy && isOwned(outTy)) {
          const h = requireOwnedHelpers(outTy);
          tail.push(`${indent}${h.assign}(_mtoc2_o${i}, ${fn.cOutputs[i]});`);
        } else {
          tail.push(`${indent}*_mtoc2_o${i} = ${fn.cOutputs[i]};`);
        }
      }
    }
    for (const o of [...owned, ...ownedParams]) {
      // Skip freeing the single-output return value — its buffer is
      // about to transfer to the caller via return-by-value.
      if (nOutputs === 1 && o.cName === fn.cOutputs[0]) continue;
      // Multi-output owned slots are moved-out by the sret writes
      // above; freeing them here would double-free the buffer the
      // caller now owns.
      if (isMulti && outputCNames.has(o.cName)) continue;
      const h = requireOwnedHelpers(o.ty);
      tail.push(`${indent}${h.free}(&${o.cName});`);
    }
    if (isVoidFn || isMulti) {
      tail.push(`${indent}return;`);
    } else {
      tail.push(`${indent}return ${fn.cOutputs[0]};`);
    }
    return tail.join("\n");
  };
  const bodyText = emitBody(
    fn.body,
    "  ",
    state,
    futureTouches,
    fnOwnedTypes,
    fnReturnTail
  );
  if (bodyText.length > 0) lines.push(bodyText);
  lines.push(fnReturnTail("  "));
  lines.push(`}`);
  return lines.join("\n");
}

/** Emit a sequence of statements with their per-stmt early-frees.
 *  After each stmt, owned C-names that aren't in the stmt's
 *  futureTouchOut (i.e. their last use was at this stmt) get a
 *  `<owned-kind>_free(&v);` call dispatched on the variable's
 *  recorded owned type. Scope-exit frees still emit unconditionally
 *  — they're no-ops for buffers already nulled by the early-free. */
export function emitBody(
  stmts: IRStmt[],
  indent: string,
  state: RuntimeState,
  futureTouches: FutureTouchMap,
  ownedTypes: ReadonlyMap<string, Type>,
  emitReturnTail: EmitReturnTail
): string {
  const out: string[] = [];
  for (const s of stmts) {
    const header = irStmtHeader(s);
    if (header !== null) {
      out.push(`${indent}/* ${header} */`);
    }
    const line = emitStmt(
      s,
      indent,
      state,
      futureTouches,
      ownedTypes,
      emitReturnTail
    );
    if (line !== null) out.push(line);
    const freeNames = earlyFreeCandidates(s, futureTouches);
    for (const v of freeNames) {
      const ty = ownedTypes.get(v);
      if (ty === undefined) {
        throw new Error(
          `emit: early-free of '${v}' but no owned type recorded`
        );
      }
      const h = activatedOwnedHelpers(ty, state);
      out.push(`${indent}${h.free}(&${v});`);
    }
  }
  return out.join("\n");
}

function emitStmt(
  s: IRStmt,
  indent: string,
  state: RuntimeState,
  futureTouches: FutureTouchMap,
  ownedTypes: ReadonlyMap<string, Type>,
  emitReturnTail: EmitReturnTail
): string | null {
  switch (s.kind) {
    case "ExprStmt":
      return `${indent}${emitExpr(s.expr, state)};`;
    case "Assign": {
      if (isOwned(s.ty)) {
        // Pure-elementwise same-shape tensor RHS goes through the
        // fused inline-loop emitter — same generated work as the
        // helper-call path for a depth-1 expression, but lets the
        // (phase-2) inliner produce deeper expressions that share a
        // single iter loop. See `emitTensorFused.ts`.
        if (s.kind === "Assign" && isFusableAssign(s)) {
          activateOwnedRuntime(s.ty, state);
          return emitTensorAssignFused(s, indent, state);
        }
        const h = activatedOwnedHelpers(s.ty, state);
        const rhs = emitOwnedRhs(s.expr, state);
        return `${indent}${h.assign}(&${s.cName}, ${rhs});`;
      }
      const rhs = emitExpr(s.expr, state);
      return `${indent}${s.cName} = ${rhs};`;
    }
    case "MemberStore": {
      const slot = [s.base.cName, ...s.fieldPath].join(".");
      if (isOwned(s.leafTy)) {
        const h = activatedOwnedHelpers(s.leafTy, state);
        const rhs = emitOwnedRhs(s.rhs, state);
        return `${indent}${h.assign}(&${slot}, ${rhs});`;
      }
      const rhs = emitExpr(s.rhs, state);
      return `${indent}${slot} = ${rhs};`;
    }
    case "IndexStore": {
      // For a member-rooted store the base Var still names the
      // OWNING root struct/class; the slot the codegen targets is
      // `<rootCName>.<fieldPath...>`. `leafTy` is the field's
      // NumericType (replaces `base.ty` for the offset / complex-lane
      // checks below).
      const slotCName =
        s.fieldPath !== undefined
          ? `${s.base.cName}.${s.fieldPath.join(".")}`
          : s.base.cName;
      const slotTy = s.leafTy ?? s.base.ty;
      const offset = emitNdScalarOffset(state, s.indices, slotCName);
      const rhs = emitExpr(s.rhs, state);
      const baseIsComplex = isNumeric(slotTy) && slotTy.isComplex;
      const rhsIsComplex = isNumeric(s.rhs.ty) && s.rhs.ty.isComplex;
      if (baseIsComplex) {
        // Both lanes must be written. The offset expression may have
        // side effects (function calls, autoinc, etc.), so hoist it to
        // a local. RHS may be real (project imag → 0) or complex
        // (split via `mtoc2_creal/cimag`).
        useRuntimeByName(state, "mtoc2_cscalar");
        if (rhsIsComplex) {
          const lines: string[] = [];
          lines.push(`${indent}{`);
          lines.push(`${indent}  long _mtoc2_off = ${offset};`);
          lines.push(`${indent}  double _Complex _mtoc2_rhs = ${rhs};`);
          lines.push(
            `${indent}  ${slotCName}.real[_mtoc2_off] = mtoc2_creal(_mtoc2_rhs);`
          );
          lines.push(
            `${indent}  ${slotCName}.imag[_mtoc2_off] = mtoc2_cimag(_mtoc2_rhs);`
          );
          lines.push(`${indent}}`);
          return lines.join("\n");
        }
        const lines: string[] = [];
        lines.push(`${indent}{`);
        lines.push(`${indent}  long _mtoc2_off = ${offset};`);
        lines.push(`${indent}  ${slotCName}.real[_mtoc2_off] = ${rhs};`);
        lines.push(`${indent}  ${slotCName}.imag[_mtoc2_off] = 0.0;`);
        lines.push(`${indent}}`);
        return lines.join("\n");
      }
      return `${indent}${slotCName}.real[${offset}] = ${rhs};`;
    }
    case "IndexSliceStore":
      return emitIndexSliceStore(s, indent, state);
    case "CellIndexStore":
      return emitCellIndexStore(s, indent, state);
    case "If": {
      const lines: string[] = [];
      lines.push(`${indent}if (${emitCondToBoolExpr(s.cond, state)}) {`);
      const thenText = emitBody(
        s.thenBody,
        indent + "  ",
        state,
        futureTouches,
        ownedTypes,
        emitReturnTail
      );
      if (thenText.length > 0) lines.push(thenText);
      if (s.elseBody.length > 0) {
        if (s.elseBody.length === 1 && s.elseBody[0].kind === "If") {
          const inner = emitStmt(
            s.elseBody[0],
            indent,
            state,
            futureTouches,
            ownedTypes,
            emitReturnTail
          );
          if (inner !== null)
            lines.push(`${indent}} else ${inner.trimStart()}`);
          else lines.push(`${indent}}`);
        } else {
          lines.push(`${indent}} else {`);
          const elseText = emitBody(
            s.elseBody,
            indent + "  ",
            state,
            futureTouches,
            ownedTypes,
            emitReturnTail
          );
          if (elseText.length > 0) lines.push(elseText);
          lines.push(`${indent}}`);
        }
      } else {
        lines.push(`${indent}}`);
      }
      return lines.join("\n");
    }
    case "While": {
      const lines: string[] = [];
      lines.push(`${indent}while (${emitCondToBoolExpr(s.cond, state)}) {`);
      const bodyText = emitBody(
        s.body,
        indent + "  ",
        state,
        futureTouches,
        ownedTypes,
        emitReturnTail
      );
      if (bodyText.length > 0) lines.push(bodyText);
      lines.push(`${indent}}`);
      return lines.join("\n");
    }
    case "For": {
      // MATLAB / numbl evaluate `start:step:end` ONCE at loop entry
      // (the range becomes a vector and the loop iterates over its
      // elements). To match: snapshot start/end into block-scoped
      // locals, derive the iteration count via `mtoc2_loop_count` so
      // it agrees with `mtoc2_tensor_make_range`'s formula, and use
      // `mtoc2_range_value` to apply the snap-to-end on the final
      // iteration. Without this, the body re-reads the live binding
      // for `end` every iteration — so e.g. `for k=1:n; n=100; end`
      // becomes a 100-iter loop instead of the MATLAB-correct 3.
      useRuntimeByName(state, "mtoc2_loop_count");
      useRuntimeByName(state, "mtoc2_range_value");
      const lines: string[] = [];
      const startC = emitExpr(s.start, state);
      const endC = emitExpr(s.end, state);
      const stepC = formatDouble(s.step);
      // Block-scope the snapshot temps so nested For loops don't
      // collide on the fixed `_mtoc2_for_*` names.
      lines.push(`${indent}{`);
      lines.push(`${indent}  const double _mtoc2_for_start = ${startC};`);
      lines.push(`${indent}  const double _mtoc2_for_end = ${endC};`);
      lines.push(
        `${indent}  const long _mtoc2_for_n = mtoc2_loop_count(_mtoc2_for_start, _mtoc2_for_end, ${stepC});`
      );
      lines.push(
        `${indent}  for (long _mtoc2_for_i = 0; _mtoc2_for_i < _mtoc2_for_n; _mtoc2_for_i++) {`
      );
      // The loop var is pre-declared at function top by
      // `collectLocals`, so the rebind here is a plain
      // assignment. This lets reads after the loop see the final
      // iteration's value (matching numbl).
      lines.push(
        `${indent}    ${s.cVar} = mtoc2_range_value(_mtoc2_for_start, ${stepC}, _mtoc2_for_end, _mtoc2_for_n, _mtoc2_for_i);`
      );
      const bodyText = emitBody(
        s.body,
        indent + "    ",
        state,
        futureTouches,
        ownedTypes,
        emitReturnTail
      );
      if (bodyText.length > 0) lines.push(bodyText);
      lines.push(`${indent}  }`);
      lines.push(`${indent}}`);
      return lines.join("\n");
    }
    case "ReturnFromFunction":
      return emitReturnTail(indent);
    case "Break":
      return `${indent}break;`;
    case "Continue":
      return `${indent}continue;`;
    case "TypeComment": {
      const lines = s.entries.map(
        e =>
          `${indent}/* type ${e.name} (${e.cName}) :: ${typeToString(e.ty)} */`
      );
      return lines.join("\n");
    }
    case "MultiAssignCall": {
      // N≥2-output user-function call OR multi-output builtin (e.g.
      // `[v, i] = sort(a)`). Named slots' `binding.cName` is pre-
      // declared at function top by `collectLocals`; ignored slots
      // become `_mtoc2_discard_<callIdx>_<i>` locals scoped to the
      // call's `{}` block.
      //
      // User-function path: callee owns its args, so owned-typed args
      // wrap in `emitOwnedRhs` (the standard `_copy` per kind).
      // Builtin path: follows the single-output builtin convention —
      // the helper reads the arg without taking ownership (no `_copy`
      // at the call site), matching every other `mtoc2_tensor_*`
      // helper that takes a tensor by value.
      const isBuiltinMA = s.isBuiltin === true;
      const builtinMA = isBuiltinMA ? lookupBuiltin(state, s.name) : undefined;
      const argStrs = s.args.map((a: IRExpr) =>
        !isBuiltinMA && isOwned(a.ty)
          ? emitOwnedRhs(a, state)
          : emitExpr(a, state)
      );
      const callIdx = state.multiAssignCallCounter++;
      const out: string[] = [];
      out.push(`${indent}{`);
      const outArgs: string[] = [];
      for (let i = 0; i < s.outputs.length; i++) {
        const slot = s.outputs[i];
        if (slot.binding === null) {
          const tmp = `_mtoc2_discard_${callIdx}_${i}`;
          if (isOwned(slot.ty)) {
            // (Unreachable in v1; kept for the future-tensor-output
            // extension.) Initialize the discard temp to an empty
            // handle so the callee's `_assign` sees a freeable slot.
            const h = activatedOwnedHelpers(slot.ty, state);
            out.push(`${indent}  ${cTypeFor(slot.ty)} ${tmp} = ${h.empty}();`);
          } else {
            out.push(
              `${indent}  ${cTypeFor(slot.ty)} ${tmp} = ${defaultInitFor()};`
            );
          }
          outArgs.push(`&${tmp}`);
        } else {
          outArgs.push(`&${slot.binding.cName}`);
        }
      }
      if (builtinMA !== undefined) {
        // Builtin path: builtin's `emitC` returns the full call string
        // including out-pointer args.
        const callStr = requireEmitC(builtinMA)({
          argsC: argStrs,
          argTypes: s.args.map((a: IRExpr) => a.ty),
          nargout: s.outputs.length,
          outArgsC: outArgs,
          useRuntime: makeEmitUseRuntime(state),
        });
        out.push(`${indent}  ${callStr};`);
      } else {
        const callArgs = [...argStrs, ...outArgs].join(", ");
        out.push(`${indent}  ${s.cName}(${callArgs});`);
      }
      // Release any owned discard temps before closing the block.
      // (Unreachable in v1; for symmetry with the future extension.)
      for (let i = 0; i < s.outputs.length; i++) {
        const slot = s.outputs[i];
        if (slot.binding === null && isOwned(slot.ty)) {
          const h = activatedOwnedHelpers(slot.ty, state);
          out.push(`${indent}  ${h.free}(&_mtoc2_discard_${callIdx}_${i});`);
        }
      }
      out.push(`${indent}}`);
      return out.join("\n");
    }
  }
}

/** Emit a `CellIndexStore` — `c{i} = rhs` or `c{i, j} = rhs`.
 *
 *  Tuple mode with every index exact: target the specific
 *  `slot_<i>` field directly. The slot's static type matches the
 *  rhs's type (the lowerer enforced this), so we use the slot's
 *  owned helpers if the rhs is owned.
 *
 *  Tuple mode with a non-exact index: emit a `switch (offset)` over
 *  every slot. Each branch uses its own owned helpers; the lowerer
 *  has already validated that every slot type unifies with the rhs
 *  type (otherwise the cell would have demoted to uniform).
 *
 *  Uniform mode: target `base.slots[<offset>]` with a runtime
 *  bounds check. The slot type is the cell's `elem` type. */
function emitCellIndexStore(
  s: Extract<IRStmt, { kind: "CellIndexStore" }>,
  indent: string,
  state: RuntimeState
): string {
  if (s.base.ty.kind !== "Cell") {
    throw new Error(
      `emit: CellIndexStore base ty is ${s.base.ty.kind}, expected Cell`
    );
  }
  const baseTy = s.base.ty;
  const baseCName = s.base.cName;
  // Resolve a linear offset from `indices`. For a 1-arg index it's
  // just the value (1-based, converted to 0-based). For 2-arg the
  // column-major formula `(j-1)*rows + (i-1)`. The lowerer has
  // already validated index arity.
  const offsetExpr = cellLinearOffsetExpr(s.indices, baseTy, baseCName, state);

  if (baseTy.mode === "uniform") {
    const elem = baseTy.elem!;
    const lines: string[] = [];
    lines.push(`${indent}{`);
    lines.push(`${indent}  size_t _mtoc2_off = ${offsetExpr};`);
    lines.push(`${indent}  if (_mtoc2_off >= ${baseCName}.nslots) {`);
    lines.push(
      `${indent}    fprintf(stderr, "mtoc2: cell index out of bounds\\n");`
    );
    lines.push(`${indent}    exit(1);`);
    lines.push(`${indent}  }`);
    if (isOwned(elem)) {
      const h = activatedOwnedHelpers(elem, state);
      const rhs = emitOwnedRhs(s.rhs, state);
      lines.push(
        `${indent}  ${h.assign}(&${baseCName}.slots[_mtoc2_off], ${rhs});`
      );
    } else {
      const rhs = emitExpr(s.rhs, state);
      lines.push(`${indent}  ${baseCName}.slots[_mtoc2_off] = ${rhs};`);
    }
    lines.push(`${indent}}`);
    return lines.join("\n");
  }

  // Tuple mode.
  const elements = baseTy.elements!;
  // Try the static-index fast path: every index is a NumLit.
  const staticOff = tryStaticCellOffset(s.indices, baseTy);
  if (staticOff !== null) {
    const slotTy = elements[staticOff];
    const slotName = `${baseCName}.slot_${staticOff}`;
    if (isOwned(slotTy)) {
      const h = activatedOwnedHelpers(slotTy, state);
      const rhs = emitOwnedRhs(s.rhs, state);
      return `${indent}${h.assign}(&${slotName}, ${rhs});`;
    }
    const rhs = emitExpr(s.rhs, state);
    return `${indent}${slotName} = ${rhs};`;
  }

  // Dynamic-index path: switch over slot index. Requires all slot
  // types unify (lowerer already enforced).
  const lines: string[] = [];
  lines.push(`${indent}{`);
  lines.push(`${indent}  size_t _mtoc2_off = ${offsetExpr};`);
  lines.push(`${indent}  switch (_mtoc2_off) {`);
  for (let i = 0; i < elements.length; i++) {
    lines.push(`${indent}    case ${i}: {`);
    if (isOwned(elements[i])) {
      const h = activatedOwnedHelpers(elements[i], state);
      const rhs = emitOwnedRhs(s.rhs, state);
      lines.push(
        `${indent}      ${h.assign}(&${baseCName}.slot_${i}, ${rhs});`
      );
    } else {
      const rhs = emitExpr(s.rhs, state);
      lines.push(`${indent}      ${baseCName}.slot_${i} = ${rhs};`);
    }
    lines.push(`${indent}      break;`);
    lines.push(`${indent}    }`);
  }
  lines.push(
    `${indent}    default: fprintf(stderr, "mtoc2: cell index out of bounds\\n"); exit(1);`
  );
  lines.push(`${indent}  }`);
  lines.push(`${indent}}`);
  return lines.join("\n");
}

/** Compute a column-major linear offset expression from cell-brace
 *  indices. The base's shape supplies the row count for the 2-D
 *  formula. Indices are 1-based in source; we subtract 1 to land
 *  in C's 0-based slot/buffer array. */
export function cellLinearOffsetExpr(
  indices: ReadonlyArray<IRExpr>,
  baseTy: Extract<Type, { kind: "Cell" }>,
  baseCName: string,
  state: RuntimeState
): string {
  if (indices.length === 1) {
    return `(size_t)((long)(${emitExpr(indices[0], state)}) - 1)`;
  }
  // For 2-D, need the row count. Tuple cells know it statically;
  // uniform cells read `base.dims[0]` at runtime.
  const rowsExpr =
    baseTy.shape !== undefined
      ? String(baseTy.shape[0])
      : `(long)${baseCName}.dims[0]`;
  const i0 = emitExpr(indices[0], state);
  const i1 = emitExpr(indices[1], state);
  return `(size_t)(((long)(${i1}) - 1) * ${rowsExpr} + ((long)(${i0}) - 1))`;
}

/** Try to compute a static slot offset from a tuple cell's indices.
 *  Returns `null` if any index isn't a NumLit. */
export function tryStaticCellOffset(
  indices: ReadonlyArray<IRExpr>,
  baseTy: Extract<Type, { kind: "Cell" }>
): number | null {
  if (baseTy.mode !== "tuple") return null;
  const shape = baseTy.shape;
  if (shape === undefined) return null;
  if (indices.length === 1) {
    if (indices[0].kind !== "NumLit") return null;
    const v = indices[0].value;
    if (!Number.isInteger(v) || v < 1) return null;
    const off = v - 1;
    if (off >= (baseTy.elements ?? []).length) return null;
    return off;
  }
  if (indices.length === 2) {
    if (indices[0].kind !== "NumLit") return null;
    if (indices[1].kind !== "NumLit") return null;
    const i = indices[0].value - 1;
    const j = indices[1].value - 1;
    if (i < 0 || j < 0) return null;
    const rows = shape[0];
    const cols = shape[1] ?? 1;
    if (i >= rows || j >= cols) return null;
    return j * rows + i;
  }
  return null;
}

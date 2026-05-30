/**
 * Elementwise-fused tensor Assign emission.
 *
 * For an Assign whose RHS is a pure elementwise expression and whose
 * multi-element operands all share the target's static shape, emit a
 * single inline iter loop instead of nested runtime-helper calls:
 *
 *     // c = a + b;     emits:
 *     {
 *       mtoc2_tensor_t _r = mtoc2_tensor_alloc_nd(a.ndim, a.dims);
 *       long _mtoc2_n = 1; for (int _mtoc2_i = 0; _mtoc2_i < _r.ndim; _mtoc2_i++) _mtoc2_n *= _r.dims[_mtoc2_i];
 *       MTOC2_OMP_PARFOR_N
 *       for (long _mtoc2_i = 0; _mtoc2_i < _mtoc2_n; _mtoc2_i++) _r.real[_mtoc2_i] = (a.real[_mtoc2_i] + b.real[_mtoc2_i]);
 *       mtoc2_tensor_assign(&c, _r);
 *     }
 *
 * The local element-count + iter variables use the `_mtoc2_` reserved
 * prefix so they can't collide with user-scope identifiers (`n`, `i`
 * are common single-letter names in MATLAB and would shadow the
 * function param / local on the enclosing block if we declared them
 * with their natural names).
 *
 * For a single-Binary RHS this is equivalent (post-`-O3`) to the
 * existing `mtoc2_tensor_assign(&c, mtoc2_tensor_plus_tt(a, b))`
 * path — the helper's body is literally the same loop. The point of
 * doing the inlining at the IR level is that the `--inline-temps`
 * pass (phase 2) can fold a chain like `_t1 = a * b; c = _t1 + d`
 * into `c = (a * b) + d`, which the fused emitter then renders as
 * one loop with no intermediate tensor allocation.
 *
 * Phase 1 (this file) only handles the *same-static-shape* case. If
 * any multi-element operand has a statically-different shape from
 * the target, the Assign falls back to the runtime-helper path
 * (`mtoc2_tensor_<op>_bcast_tt(...)`). Broadcast fusion is phase 3.
 */

import type { Assign, IRExpr } from "../lowering/ir.js";
import type { NumericType, Type } from "../lowering/types.js";
import { DIM_ONE, isMultiElement, isNumeric } from "../lowering/types.js";
import { getBuiltin } from "../builtins/index.js";
import { requireEmitC } from "../builtins/registry.js";
import {
  lookupBuiltin,
  makeEmitUseRuntime,
  useRuntimeByName,
  useRuntimeInline,
  type InlineSnippet,
  type RuntimeState,
} from "./runtime.js";
import { formatDouble } from "./cHelpers.js";

/** Runtime shape-equality probe used to guard a fused loop whose
 *  operand shapes are NOT statically proven equal (they share a static
 *  dim PATTERN but carry dynamic dims, so their runtime extents may
 *  differ — `ones(1,n) + ones(1,m)`). C-only: the JS backend never
 *  fuses (it always routes tensor ops through the shape-checking
 *  `tt_kernel`). */
const FUSED_SHAPE_EQ_SNIPPET: InlineSnippet = {
  name: "mtoc2_fused_shape_eq",
  deps: ["mtoc2_tensor_t"],
  code: `static int mtoc2_fused_shape_eq(mtoc2_tensor_t a, mtoc2_tensor_t b) {
  if (a.ndim != b.ndim) return 0;
  for (int i = 0; i < a.ndim; i++)
    if (a.dims[i] != b.dims[i]) return 0;
  return 1;
}`,
};

/** Strip multi-element shape from a type, leaving a scalar variant
 *  (same elem / isComplex / sign). Used to feed builtin `emit` the
 *  scalar-form types it expects when the framework is requesting the
 *  per-slot expression from inside a fused loop. Non-numeric and
 *  already-scalar types pass through unchanged. */
function scalarVersionOf(t: Type): Type {
  if (!isNumeric(t)) return t;
  if (!isMultiElement(t)) return t;
  return {
    kind: "Numeric",
    elem: t.elem,
    isComplex: t.isComplex,
    dims: [DIM_ONE, DIM_ONE],
    shape: [1, 1],
    sign: t.sign,
  };
}

/** True iff `e` is a pure-elementwise expression — only NumLit / Var /
 *  Binary / Unary / Call to a builtin that has a `perSlotC` hook. No
 *  owned producers (TensorBuild, TensorConcat, IndexSlice), no
 *  non-elementwise calls (mtimes, reshape, sum, …), no user-fn calls,
 *  no HandleLit, no IndexLoad/HandleCaptureLoad/MemberLoad, etc.
 *
 *  These are exactly the node kinds the per-slot renderer below
 *  knows how to walk. */
export function isPureElementwiseExpr(e: IRExpr): boolean {
  switch (e.kind) {
    case "NumLit":
    case "Var":
      return true;
    case "Binary": {
      // `Binary` covers more than elementwise ops — `mtimes` /
      // `mrdivide` are matrix ops that need a runtime helper. Gate
      // on the builtin's `elementwise` flag.
      const b = getBuiltin(e.builtin);
      if (!b || !b.elementwise) return false;
      // `mtimes`/`mrdivide` declare elementwise but only the
      // at-least-one-scalar case actually degenerates to elementwise
      // `times`/`rdivide`. Reject the both-tensor case here.
      if (e.builtin === "mtimes" || e.builtin === "mrdivide") {
        if (
          isNumeric(e.left.ty) &&
          isMultiElement(e.left.ty) &&
          isNumeric(e.right.ty) &&
          isMultiElement(e.right.ty)
        ) {
          return false;
        }
      }
      return isPureElementwiseExpr(e.left) && isPureElementwiseExpr(e.right);
    }
    case "Unary": {
      const b = getBuiltin(e.builtin);
      if (!b || !b.elementwise) return false;
      return isPureElementwiseExpr(e.operand);
    }
    case "Call": {
      const b = getBuiltin(e.name);
      if (!b || !b.elementwise) return false;
      // The fused-loop contract requires a scalar result per slot. A
      // tensor-producing Call (e.g. the shape-constructor form of
      // `nan(2,3)`) flags `elementwise` for its 0-arg scalar branch
      // but returns a fresh tensor when invoked with shape args —
      // route those through the standard helper path instead.
      if (isNumeric(e.ty) && isMultiElement(e.ty)) return false;
      return e.args.every(isPureElementwiseExpr);
    }
    default:
      return false;
  }
}

/** True iff the fused emitter can handle this Assign. Requires:
 *    - target is a real-double multi-element tensor
 *    - RHS is pure elementwise (see `isPureElementwiseExpr`)
 *    - every multi-element Var in the RHS shares the target's static
 *      shape (no broadcasting in this phase)
 *    - no complex tensor anywhere in the RHS — the per-slot renderer
 *      addresses tensors as `<cName>.real[i]` and drops the imag lane.
 *      Complex sites take the runtime-helper path. */
export function isFusableAssign(s: Assign): boolean {
  if (!isNumeric(s.ty)) return false;
  if (s.ty.isComplex) return false;
  if (!isMultiElement(s.ty)) return false;
  if (!isPureElementwiseExpr(s.expr)) return false;
  if (!everyTensorVarMatchesShape(s.expr, s.ty)) return false;
  return !rhsTouchesComplexTensor(s.expr);
}

/** Distinct cNames of the multi-element tensor Vars read in `e`, in
 *  first-appearance order. */
function multiElemOperandCNames(e: IRExpr): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  walkExpr(e, sub => {
    if (
      sub.kind === "Var" &&
      isNumeric(sub.ty) &&
      isMultiElement(sub.ty) &&
      !seen.has(sub.cName)
    ) {
      seen.add(sub.cName);
      order.push(sub.cName);
    }
  });
  return order;
}

/** A fusable Assign needs a runtime shape guard iff it reads two or more
 *  DISTINCT multi-element operands AND at least one carries a dynamic
 *  (non-exact) dim. In that case `isFusableAssign` matched the operands
 *  on a shared static dim PATTERN, but their runtime extents aren't
 *  proven equal (`ones(1,n) + ones(1,m)`), so the fused loop — which
 *  reads every operand over the shape source's linear extent — could
 *  read out of bounds. With all-exact dims the pattern match already
 *  proves equality, and a single distinct operand is its own shape
 *  source; neither needs the guard. */
export function fusedAssignNeedsShapeGuard(s: Assign): boolean {
  if (multiElemOperandCNames(s.expr).length < 2) return false;
  let anyDynamic = false;
  walkExpr(s.expr, sub => {
    if (
      sub.kind === "Var" &&
      isNumeric(sub.ty) &&
      isMultiElement(sub.ty) &&
      sub.ty.dims.some(d => d.kind !== "exact")
    ) {
      anyDynamic = true;
    }
  });
  return anyDynamic;
}

function rhsTouchesComplexTensor(e: IRExpr): boolean {
  let touched = false;
  walkExpr(e, sub => {
    if (isNumeric(sub.ty) && isMultiElement(sub.ty) && sub.ty.isComplex) {
      touched = true;
    }
  });
  return touched;
}

function everyTensorVarMatchesShape(e: IRExpr, target: NumericType): boolean {
  let ok = true;
  walkExpr(e, sub => {
    if (
      sub.kind === "Var" &&
      isNumeric(sub.ty) &&
      isMultiElement(sub.ty) &&
      !sameStaticShape(sub.ty, target)
    ) {
      ok = false;
    }
  });
  return ok;
}

function sameStaticShape(a: NumericType, b: NumericType): boolean {
  if (a.dims.length !== b.dims.length) return false;
  for (let i = 0; i < a.dims.length; i++) {
    const da = a.dims[i];
    const db = b.dims[i];
    if (da.kind !== db.kind) return false;
    if (da.kind === "exact" && db.kind === "exact" && da.value !== db.value) {
      return false;
    }
  }
  return true;
}

function walkExpr(e: IRExpr, visit: (sub: IRExpr) => void): void {
  visit(e);
  switch (e.kind) {
    case "Binary":
      walkExpr(e.left, visit);
      walkExpr(e.right, visit);
      return;
    case "Unary":
      walkExpr(e.operand, visit);
      return;
    case "Call":
      for (const a of e.args) walkExpr(a, visit);
      return;
    default:
      return;
  }
}

/** Render the per-slot C expression for `e`. Multi-element Vars
 *  become `<cName>.real[_mtoc2_i]`; scalar Vars stay as their bare
 *  cName; NumLits render as their double form; Binary/Unary/Call
 *  routes through each builtin's `perSlotC`. The fixed iter variable
 *  name `_mtoc2_i` matches the outer loop emitted by
 *  `emitTensorAssignFused`; the `_mtoc2_` reserved prefix is used so
 *  the iter doesn't shadow a user-named loop variable like `i`. */
function emitPerSlotExpr(e: IRExpr, state: RuntimeState): string {
  switch (e.kind) {
    case "NumLit":
      return formatDouble(e.value);
    case "Var":
      if (isNumeric(e.ty) && isMultiElement(e.ty)) {
        return `${e.cName}.real[_mtoc2_i]`;
      }
      return e.cName;
    case "Binary": {
      const b = lookupBuiltin(state, e.builtin);
      if (!b || !b.elementwise) {
        throw new Error(
          `emitTensorFused internal: builtin '${e.builtin}' is not elementwise`
        );
      }
      return requireEmitC(b)({
        argsC: [
          emitPerSlotExpr(e.left, state),
          emitPerSlotExpr(e.right, state),
        ],
        argTypes: [scalarVersionOf(e.left.ty), scalarVersionOf(e.right.ty)],
        nargout: 1,
        useRuntime: makeEmitUseRuntime(state),
      });
    }
    case "Unary": {
      const b = lookupBuiltin(state, e.builtin);
      if (!b || !b.elementwise) {
        throw new Error(
          `emitTensorFused internal: builtin '${e.builtin}' is not elementwise`
        );
      }
      return requireEmitC(b)({
        argsC: [emitPerSlotExpr(e.operand, state)],
        argTypes: [scalarVersionOf(e.operand.ty)],
        nargout: 1,
        useRuntime: makeEmitUseRuntime(state),
      });
    }
    case "Call": {
      const b = lookupBuiltin(state, e.name);
      if (!b || !b.elementwise) {
        throw new Error(
          `emitTensorFused internal: builtin '${e.name}' is not elementwise`
        );
      }
      return requireEmitC(b)({
        argsC: e.args.map(a => emitPerSlotExpr(a, state)),
        argTypes: e.args.map(a => scalarVersionOf(a.ty)),
        nargout: 1,
        useRuntime: makeEmitUseRuntime(state),
      });
    }
    default:
      throw new Error(
        `emitTensorFused internal: unexpected IR node '${e.kind}'`
      );
  }
}

/** Emit the fused inline iter loop. Pre-condition:
 *  `isFusableAssign(s)` returned true.
 *
 *  When `fallbackStmt` is supplied (the caller passes it iff
 *  `fusedAssignNeedsShapeGuard(s)` is true), the fast loop is wrapped in
 *  a runtime shape-equality guard: every other distinct multi-element
 *  operand must match the shape source's runtime shape, else the assign
 *  defers to `fallbackStmt` — the bcast-aware helper path, which
 *  broadcasts compatible shapes and raises "Matrix dimensions must
 *  agree" on incompatible ones. Mirrors the JS `tt_kernel` deferring to
 *  `bcast_kernel`. */
export function emitTensorAssignFused(
  s: Assign,
  indent: string,
  state: RuntimeState,
  fallbackStmt?: string
): string {
  // Find a shape source. `isFusableAssign` guarantees at least one
  // multi-element Var in the RHS (otherwise the target couldn't be
  // multi-element — the lowering type system enforces this).
  let shapeSrcCName: string | null = null;
  walkExpr(s.expr, sub => {
    if (
      shapeSrcCName === null &&
      sub.kind === "Var" &&
      isNumeric(sub.ty) &&
      isMultiElement(sub.ty)
    ) {
      shapeSrcCName = sub.cName;
    }
  });
  if (shapeSrcCName === null) {
    throw new Error(
      `emitTensorFused internal: fusable Assign '${s.cName}' has no shape-source Var`
    );
  }

  // Allocate via the N-D helper. `MTOC2_OMP_PARFOR_N` comes with the
  // tensor typedef (transitively pulled in by `mtoc2_tensor_alloc_nd`).
  useRuntimeByName(state, "mtoc2_tensor_alloc_nd");
  useRuntimeByName(state, "mtoc2_tensor_assign");

  const slot = emitPerSlotExpr(s.expr, state);
  // Use `_mtoc2_`-reserved local names so a user identifier called `n`
  // or `i` (legitimate in MATLAB — `i` is also the imaginary unit
  // fallback) doesn't get shadowed by the inlined loop's count /
  // iter. The slot expression embeds raw operand reads (e.g.
  // `_t1.real[_mtoc2_i] * n`); if `n` were a local here it would
  // bind to the loop count rather than the user's function parameter,
  // silently mis-computing every elementwise op against a user-named
  // scalar.
  const fusedBlock = (ind: string): string[] => [
    `${ind}{`,
    `${ind}  mtoc2_tensor_t _r = mtoc2_tensor_alloc_nd(${shapeSrcCName}.ndim, ${shapeSrcCName}.dims);`,
    `${ind}  long _mtoc2_n = 1;`,
    `${ind}  for (int _mtoc2_i = 0; _mtoc2_i < _r.ndim; _mtoc2_i++) _mtoc2_n *= _r.dims[_mtoc2_i];`,
    `${ind}  MTOC2_OMP_PARFOR_N`,
    `${ind}  for (long _mtoc2_i = 0; _mtoc2_i < _mtoc2_n; _mtoc2_i++) _r.real[_mtoc2_i] = ${slot};`,
    `${ind}  mtoc2_tensor_assign(&${s.cName}, _r);`,
    `${ind}}`,
  ];

  if (fallbackStmt === undefined) return fusedBlock(indent).join("\n");

  useRuntimeInline(state, FUSED_SHAPE_EQ_SNIPPET);
  const others = multiElemOperandCNames(s.expr).filter(
    c => c !== shapeSrcCName
  );
  const cond = others
    .map(c => `mtoc2_fused_shape_eq(${c}, ${shapeSrcCName})`)
    .join(" && ");
  return [
    `${indent}if (${cond}) {`,
    ...fusedBlock(indent + "  "),
    `${indent}} else {`,
    `${indent}  ${fallbackStmt}`,
    `${indent}}`,
  ].join("\n");
}

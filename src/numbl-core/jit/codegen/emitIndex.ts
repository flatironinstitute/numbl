/**
 * Index-related codegen — scalar offsets, slice slot setup, range and
 * logical-mask setup, slice producer (`IndexSlice`), and slice store
 * (`IndexSliceStore`). Shared by `emit.ts`'s `emitStmt` / `emitExpr`
 * arms for IndexLoad / IndexStore / IndexSlice / IndexSliceStore.
 *
 * Everything here ultimately emits column-major offset arithmetic
 * against an `jit_tensor_t`'s `.real` / `.imag` flat buffers, with
 * bounds checks via the `jit_idx_*` / `jit_check_*` runtime
 * helpers (per-axis, per-linear-index, or full per-range).
 */

import type { IRExpr, IRStmt, IndexSliceArg } from "../lowering/ir.js";
import {
  isColVecTy,
  isRowVecTy,
  isNumeric,
  isScalar,
  type Type,
} from "../lowering/types.js";
import { formatDouble } from "./cHelpers.js";
import { dimsProductExpr, formatNdOffset, locStringOf } from "./cFormat.js";
import { useRuntimeByName, type RuntimeState } from "./runtime.js";
import { emitExpr } from "./emitExpr.js";

/** Compute the column-major linear buffer offset for a scalar
 *  IndexLoad / IndexStore: 1-arg linear, 2-arg row/col, or N-arg N-D.
 *  Each axis index is wrapped in a runtime bounds-check call
 *  (`jit_idx_axis` for per-axis; `jit_idx_lin` for 1-arg linear)
 *  so an OOB access raises with a numbl-style "Index exceeds array
 *  bounds" message instead of silently reading/writing past the
 *  buffer. Mirrors mtoc's `emitNdScalarOffset` — one source of truth
 *  for the offset formula.
 *
 *  `grow` selects the indexed-STORE variant: the `*_grow` bounds
 *  checks bail (longjmp to the host-entry guard → host re-runs on the
 *  interpreter) instead of raising when the index exceeds the extent,
 *  because a store past the end GROWS the array in MATLAB — which the
 *  JIT can't model. Reads pass `grow = false` (OOB is a genuine
 *  error). See `grow_bail.h`. */
export function emitNdScalarOffset(
  state: RuntimeState,
  indices: ReadonlyArray<IRExpr>,
  baseCName: string,
  grow = false
): string {
  if (grow) {
    useRuntimeByName(state, "jit_grow_bail");
  } else {
    useRuntimeByName(state, "jit_oob_abort");
  }
  const linFn = grow ? "jit_idx_lin_grow" : "jit_idx_lin";
  const axisFn = grow ? "jit_idx_axis_grow" : "jit_idx_axis";
  if (indices.length === 1) {
    const loc = locStringOf(indices[0].span);
    return `${linFn}(&${baseCName}, jit_to_idx(${emitExpr(indices[0], state)}), ${loc})`;
  }
  const terms: string[] = [];
  for (let i = 0; i < indices.length; i++) {
    const loc = locStringOf(indices[i].span);
    const checked = `${axisFn}(&${baseCName}, ${i}, jit_to_idx(${emitExpr(indices[i], state)}), ${loc})`;
    if (i === 0) {
      terms.push(checked);
    } else {
      const strideParts: string[] = [];
      for (let j = 0; j < i; j++) strideParts.push(`${baseCName}.dims[${j}]`);
      terms.push(`${checked} * ${strideParts.join(" * ")}`);
    }
  }
  return terms.join(" + ");
}

/** Emit per-slot setup for a multi-slot slice (read or write): pushes
 *  `_jit_n_<i>` (per-slot iteration count) and any Range / Scalar
 *  locals into `lines`. Returns the per-slot source-index expression
 *  (in terms of `_jit_k_<i>` for Colon/Range, or the precomputed
 *  `_jit_src_<i>` local for Scalar slots).
 *
 *  `cleanups`, when provided, is populated with lines that must run
 *  after the iter loop (currently used for LogicalMask slots to free
 *  the precomputed source-index buffer). The caller is expected to
 *  emit these lines immediately after closing the slot iteration
 *  loops and before the GCC statement-expression's yield expression. */
export function emitSliceSlotSetup(
  state: RuntimeState,
  lines: string[],
  indent: string,
  slotsTyped: ReadonlyArray<IndexSliceArg>,
  baseCName: string,
  cleanups?: string[],
  grow = false
): string[] {
  const slotSrc: string[] = [];
  for (let i = 0; i < slotsTyped.length; i++) {
    const slot = slotsTyped[i];
    const kVar = `_jit_k_${i}`;
    if (slot.kind === "Colon") {
      lines.push(`${indent}long _jit_n_${i} = ${baseCName}.dims[${i}];`);
      slotSrc.push(kVar);
    } else if (slot.kind === "Scalar") {
      // Per-axis bounds check at setup time. Reads use `jit_idx_axis`,
      // which raises with a numbl-style "Index in position N exceeds array
      // bounds"; writes use the `_grow` variant, which bails to the
      // interpreter for an index past the extent (MATLAB would grow the
      // array there, which the JIT cannot model) and still raises on a
      // genuine sub-1 index. Without the grow variant a write like
      // `A(k, :) = row` could not be compiled at all unless `k` were
      // statically known — which it never is inside a loop.
      const axisFn = grow ? "jit_idx_axis_grow" : "jit_idx_axis";
      if (grow) {
        useRuntimeByName(state, "jit_grow_bail");
      } else {
        useRuntimeByName(state, "jit_oob_abort");
      }
      const scalarStr = emitExpr(slot.expr, state);
      const loc = locStringOf(slot.span);
      lines.push(`${indent}long _jit_n_${i} = 1;`);
      lines.push(
        `${indent}long _jit_src_${i} = ${axisFn}(&${baseCName}, ${i}, jit_to_idx(${scalarStr}), ${loc});`
      );
      slotSrc.push(`_jit_src_${i}`);
    } else if (slot.kind === "IndexVec") {
      // Fancy gather. The slot's tensor expression is ANF'd to a Var
      // (see `anfChildren`'s IndexSlice case), so we read its values
      // per iteration without re-evaluating. Each entry is a 1-based
      // index into the base's i-th axis; `jit_idx_axis` does the
      // bounds check and 1→0-based conversion per access.
      if (slot.expr.kind !== "Var") {
        throw new Error(
          "emit internal: IndexVec slot expr must be a Var after ANF"
        );
      }
      useRuntimeByName(state, "jit_oob_abort");
      const idxCName = slot.expr.cName;
      const idxTy = slot.expr.ty;
      lines.push(
        `${indent}long _jit_n_${i} = ${dimsProductExpr(idxCName, idxTy)};`
      );
      const loc = locStringOf(slot.span);
      // Per-iteration: read the 1-based index, bounds-check, convert to 0-based.
      slotSrc.push(
        `jit_idx_axis(&${baseCName}, ${i}, jit_to_idx(${idxCName}.real[${kVar}]), ${loc})`
      );
    } else if (slot.kind === "LogicalMask") {
      // Per-axis logical-mask gather. Scan the mask once at setup time
      // and fill a `long[]` index buffer with the 0-based source-axis
      // positions where the mask is truthy. The truthy count is the
      // per-slot iteration count; per-iter the source index is the
      // i-th entry of the buffer. Buffer is freed after the iter loop
      // via `cleanups`.
      if (slot.expr.kind !== "Var") {
        throw new Error(
          "emit internal: LogicalMask slot expr must be a Var after ANF"
        );
      }
      useRuntimeByName(state, "jit_logical_mask_indices");
      useRuntimeByName(state, "jit_alloc");
      const maskCName = slot.expr.cName;
      const maskTy = slot.expr.ty;
      const loc = locStringOf(slot.span);
      lines.push(
        `${indent}long _jit_mask_n_${i} = ${dimsProductExpr(maskCName, maskTy)};`
      );
      lines.push(
        `${indent}long *_jit_idx_${i} = (long *)jit_alloc(sizeof(long) * (_jit_mask_n_${i} > 0 ? (size_t)_jit_mask_n_${i} : 1));`
      );
      lines.push(
        `${indent}long _jit_n_${i} = jit_logical_mask_indices(${maskCName}, ${baseCName}.dims[${i}], ${i}, ${loc}, _jit_idx_${i});`
      );
      if (cleanups) {
        cleanups.push(`${indent}free(_jit_idx_${i});`);
      }
      slotSrc.push(`_jit_idx_${i}[${kVar}]`);
    } else {
      if (slot.step.kind !== "NumLit") {
        throw new Error(
          "emit internal: IndexSlice range step must be a NumLit; " +
            "should have been caught at lowering"
        );
      }
      // Range-slot bounds check: validate first and last 1-based
      // indices once at setup time. The per-iter index expression
      // doesn't need its own check — `jit_loop_count` derives `n`
      // monotonically from start/end/step, so the iteration stays
      // within `[first, last]`.
      useRuntimeByName(state, "jit_loop_count");
      useRuntimeByName(state, "jit_oob_abort");
      const startStr = emitExpr(slot.start, state);
      const endStr = emitExpr(slot.end, state);
      const stepStr = formatDouble(slot.step.value);
      const loc = locStringOf(slot.span);
      lines.push(`${indent}double _jit_start_${i} = ${startStr};`);
      lines.push(`${indent}double _jit_end_${i} = ${endStr};`);
      lines.push(
        `${indent}long _jit_n_${i} = jit_loop_count(_jit_start_${i}, _jit_end_${i}, ${stepStr});`
      );
      // Skip the bounds check on an empty range — the loop won't run
      // and validating an out-of-range start/end would reject benign
      // cases like `v(5:4)` on a 3-element vector (which yields an
      // empty slice in MATLAB).
      lines.push(`${indent}if (_jit_n_${i} > 0) {`);
      lines.push(
        `${indent}  long _jit_first_${i} = jit_to_idx(_jit_start_${i});`
      );
      lines.push(
        `${indent}  long _jit_last_${i} = jit_to_idx(_jit_start_${i} + ${stepStr} * (double)(_jit_n_${i} - 1));`
      );
      lines.push(
        `${indent}  jit_check_axis_range(&${baseCName}, ${i}, _jit_first_${i}, _jit_last_${i}, ${loc});`
      );
      lines.push(`${indent}}`);
      slotSrc.push(
        `(jit_to_idx(_jit_start_${i} + ${stepStr} * (double)${kVar}) - 1L)`
      );
    }
  }
  return slotSrc;
}

/** Emit a single-slot linear Range slice's setup block: pushes locals
 *  `_jit_start`/`_jit_end`/`_jit_n`, gates a
 *  `jit_check_linear_range` call on a non-empty range, and returns a
 *  function that maps a loop-counter expression to the corresponding
 *  source/destination buffer offset. Shared by `emitIndexSliceProducer`
 *  (read) and `emitIndexSliceStore` (write). */
function emitLinearRangeSetup(
  slot: Extract<IndexSliceArg, { kind: "Range" }>,
  baseCName: string,
  lines: string[],
  indent: string,
  state: RuntimeState,
  grow = false
): (kVar: string) => string {
  if (slot.step.kind !== "NumLit") {
    throw new Error("emit internal: index-slot Range step must be NumLit");
  }
  useRuntimeByName(state, "jit_loop_count");
  const startStr = emitExpr(slot.start, state);
  const endStr = emitExpr(slot.end, state);
  const stepStr = formatDouble(slot.step.value);
  const loc = locStringOf(slot.span);
  lines.push(`${indent}double _jit_start = ${startStr};`);
  lines.push(`${indent}double _jit_end = ${endStr};`);
  lines.push(
    `${indent}long _jit_n = jit_loop_count(_jit_start, _jit_end, ${stepStr});`
  );
  if (grow) {
    // Store path: each target index is grow-aware. An index past the
    // end grow-bails to the interpreter (which has full grow semantics
    // — the canonical `v(end+1:end+k) = ...` append); a sub-1 index
    // raises. No up-front check: a growing range must bail, not
    // raise.
    useRuntimeByName(state, "jit_grow_bail");
    return k =>
      `jit_idx_lin_grow(&${baseCName}, jit_to_idx(_jit_start + ${stepStr} * (double)${k}), ${loc})`;
  }
  useRuntimeByName(state, "jit_oob_abort");
  // Single-slot range slice indexes linearly over numel(base), not
  // against a single axis dim. Skip the check on an empty range
  // (MATLAB allows `v(5:4)` to yield 1×0).
  lines.push(`${indent}if (_jit_n > 0) {`);
  lines.push(`${indent}  long _jit_first = jit_to_idx(_jit_start);`);
  lines.push(
    `${indent}  long _jit_last = jit_to_idx(_jit_start + ${stepStr} * (double)(_jit_n - 1));`
  );
  lines.push(
    `${indent}  jit_check_linear_range(&${baseCName}, _jit_first, _jit_last, ${loc});`
  );
  lines.push(`${indent}}`);
  return k => `jit_to_idx(_jit_start + ${stepStr} * (double)${k}) - 1L`;
}

/** Emit setup for a single-slot linear logical-mask slice (read or
 *  write). Pushes the `_jit_mask_n`, `_jit_base_n`, `_jit_idx`,
 *  and `_jit_n` locals into `lines`, activates the required runtime
 *  helpers, and returns the per-iter source-index expression plus a
 *  cleanup line to emit after the iter loop. Indent is applied to the
 *  pushed lines AND to the returned cleanup so the caller can drop it
 *  straight into its line list. */
function emitLinearLogicalMaskSetup(
  slot: Extract<IndexSliceArg, { kind: "LogicalMask" }>,
  baseCName: string,
  baseTy: Type,
  lines: string[],
  indent: string,
  state: RuntimeState
): { srcIndexFor: (kVar: string) => string; cleanup: string } {
  if (slot.expr.kind !== "Var") {
    throw new Error(
      "emit internal: LogicalMask slot expr must be a Var after ANF"
    );
  }
  useRuntimeByName(state, "jit_logical_mask_indices");
  useRuntimeByName(state, "jit_alloc");
  const maskCName = slot.expr.cName;
  const loc = locStringOf(slot.span);
  lines.push(
    `${indent}long _jit_mask_n = ${dimsProductExpr(maskCName, slot.expr.ty)};`
  );
  lines.push(
    `${indent}long _jit_base_n = ${dimsProductExpr(baseCName, baseTy)};`
  );
  lines.push(
    `${indent}long *_jit_idx = (long *)jit_alloc(sizeof(long) * (_jit_mask_n > 0 ? (size_t)_jit_mask_n : 1));`
  );
  lines.push(
    `${indent}long _jit_n = jit_logical_mask_indices(${maskCName}, _jit_base_n, -1, ${loc}, _jit_idx);`
  );
  return {
    srcIndexFor: k => `_jit_idx[${k}]`,
    cleanup: `${indent}free(_jit_idx);`,
  };
}

/** Emit a "lhs/rhs element count mismatch" runtime check for a tensor
 *  RHS in a slice store. Assumes `_jit_n` is already declared as the
 *  lhs slice element count. On a mismatch the interpreter grows an empty
 *  base (`A = []; A(:, 1) = v`), so that bails; otherwise it is the
 *  interpreter's "Subscripted assignment dimension mismatch". */
function emitTensorRhsSizeCheck(
  rhs: Extract<IRExpr, { kind: "Var" }>,
  baseCName: string,
  baseTy: Type,
  lines: string[],
  indent: string
): void {
  lines.push(
    `${indent}long _jit_rhs_n = ${dimsProductExpr(rhs.cName, rhs.ty)};`
  );
  lines.push(`${indent}if (_jit_n != _jit_rhs_n) {`);
  lines.push(
    `${indent}  if (${dimsProductExpr(baseCName, baseTy)} == 0) jit_raise_bail("slice store into an empty array");`
  );
  lines.push(
    `${indent}  jit_raise_error("", "Subscripted assignment dimension mismatch");`
  );
  lines.push(`${indent}}`);
}

/** Emit an `IndexSlice` as a C statement-expression-style block that
 *  allocates the result tensor, fills it, and evaluates to the result
 *  via a comma expression. The result is consumed at an owned consume
 *  site (`jit_tensor_assign(&v, <here>)`); ANF guarantees IndexSlice
 *  appears only as the direct RHS of an Assign. */
export function emitIndexSliceProducer(
  e: Extract<IRExpr, { kind: "IndexSlice" }>,
  state: RuntimeState
): string {
  // Generate via a GCC/Clang statement-expression. This keeps the
  // IndexSlice producer self-contained at the expression site without
  // requiring out-of-line statements.
  if (e.base.kind !== "Var") {
    throw new Error(
      `emit internal: IndexSlice base must be a Var after ANF (got ${e.base.kind})`
    );
  }
  useRuntimeByName(state, "jit_tensor_t");
  const baseIsComplex = isNumeric(e.base.ty) && e.base.ty.isComplex;
  if (baseIsComplex) {
    useRuntimeByName(state, "jit_tensor_alloc_nd_complex");
  } else {
    useRuntimeByName(state, "jit_tensor_alloc_nd");
  }
  const allocFn = baseIsComplex
    ? "jit_tensor_alloc_nd_complex"
    : "jit_tensor_alloc_nd";
  const baseCName = e.base.cName;
  const lines: string[] = [];

  // Per-element copy: real lane unconditionally; imag lane only when
  // base is complex (its `imag` is non-NULL).
  const copyElem = (dstK: string, srcK: string): string =>
    baseIsComplex
      ? `_jit_t.real[${dstK}] = ${baseCName}.real[${srcK}]; _jit_t.imag[${dstK}] = ${baseCName}.imag[${srcK}];`
      : `_jit_t.real[${dstK}] = ${baseCName}.real[${srcK}];`;

  if (e.index.length === 1) {
    // Single-slot linear form.
    const slot = e.index[0];
    let count: string;
    let srcIndexFor: (kVar: string) => string;
    let resultRows: string;
    let resultCols: string;
    let linearCleanup: string | null = null;
    if (slot.kind === "Colon") {
      lines.push(`long _jit_n = ${dimsProductExpr(baseCName, e.base.ty)};`);
      count = "_jit_n";
      srcIndexFor = k => k;
      resultRows = "_jit_n";
      resultCols = "1";
    } else if (slot.kind === "Range") {
      srcIndexFor = emitLinearRangeSetup(slot, baseCName, lines, "", state);
      count = "_jit_n";
      // Single-slot range: row-vec → row, col-vec → col, matrix/N-D → row.
      const isColVec = e.base.ty.kind === "Numeric" && isColVecTy(e.base.ty);
      if (isColVec) {
        resultRows = "_jit_n";
        resultCols = "1";
      } else {
        resultRows = "1";
        resultCols = "_jit_n";
      }
    } else if (slot.kind === "LogicalMask") {
      // Single-slot linear logical-mask read: scan the mask once,
      // collect 0-based positions where it's truthy, then walk the
      // buffer. Each truthy mask position must be < numel(base);
      // `jit_logical_mask_indices` raises otherwise. Result shape
      // mirrors single-slot Range: row-vec base → row; col-vec base
      // → col; matrix / N-D base → column vector.
      const linMask = emitLinearLogicalMaskSetup(
        slot,
        baseCName,
        e.base.ty,
        lines,
        "",
        state
      );
      count = "_jit_n";
      srcIndexFor = linMask.srcIndexFor;
      const isRowBase = e.base.ty.kind === "Numeric" && isRowVecTy(e.base.ty);
      if (isRowBase) {
        resultRows = "1";
        resultCols = "_jit_n";
      } else {
        resultRows = "_jit_n";
        resultCols = "1";
      }
      linearCleanup = linMask.cleanup;
    } else {
      throw new Error(
        "emit internal: single-slot Scalar IndexSlice should have routed to IndexLoad"
      );
    }
    // A slice that is statically a single element — `v(2:2)`, or `v(:)`
    // on a 1×1 base — is typed scalar at the consume site (`double y =
    // ...`), so it must evaluate to a scalar, NOT an allocated 1×1 tensor
    // (which is a C type mismatch). Yield the element at the first source
    // index, reusing the setup already pushed above. (LogicalMask is never
    // statically scalar, so srcIndexFor here is the Colon/Range form with
    // no buffer to free.)
    if (isScalar(e.ty)) {
      const off = srcIndexFor("0L");
      if (baseIsComplex) {
        useRuntimeByName(state, "jit_cscalar");
        lines.push(`long _jit_off = ${off};`);
        lines.push(
          `jit_cmake(${baseCName}.real[_jit_off], ${baseCName}.imag[_jit_off]);`
        );
      } else {
        lines.push(`${baseCName}.real[${off}];`);
      }
      return `({ ${lines.join(" ")} })`;
    }
    lines.push(
      `jit_tensor_t _jit_t = ${allocFn}(2, (long[]){${resultRows}, ${resultCols}});`
    );
    lines.push(`for (long _jit_k = 0; _jit_k < ${count}; _jit_k++) {`);
    lines.push(`  ${copyElem("_jit_k", srcIndexFor("_jit_k"))}`);
    lines.push(`}`);
    if (linearCleanup !== null) lines.push(linearCleanup);
    lines.push(`_jit_t;`);
    return `({ ${lines.join(" ")} })`;
  }

  // Multi-slot per-axis form.
  const ndim = e.index.length;
  const cleanups: string[] = [];
  const slotSrc = emitSliceSlotSetup(
    state,
    lines,
    "",
    e.index,
    baseCName,
    cleanups
  );
  // Multi-slot slice that is statically a single element — e.g. `M(1:1, 2)`
  // — is typed scalar at the consume site, so yield the scalar element
  // rather than a 1×1 tensor. Pin every axis loop index to 0, compute the
  // one source offset, free any gathered-index buffers, then read.
  if (isScalar(e.ty)) {
    for (let i = 0; i < ndim; i++) lines.push(`long _jit_k_${i} = 0;`);
    lines.push(
      `long _jit_src_off = ${formatNdOffset(slotSrc, j => `${baseCName}.dims[${j}]`)};`
    );
    for (const c of cleanups) lines.push(c);
    if (baseIsComplex) {
      useRuntimeByName(state, "jit_cscalar");
      lines.push(
        `jit_cmake(${baseCName}.real[_jit_src_off], ${baseCName}.imag[_jit_src_off]);`
      );
    } else {
      lines.push(`${baseCName}.real[_jit_src_off];`);
    }
    return `({ ${lines.join(" ")} })`;
  }
  const resultRank =
    e.ty.kind === "Numeric" ? Math.max(2, e.ty.dims.length) : 2;
  const dimsList: string[] = [];
  for (let i = 0; i < resultRank; i++) {
    dimsList.push(i < ndim ? `_jit_n_${i}` : `1L`);
  }
  lines.push(
    `jit_tensor_t _jit_t = ${allocFn}(${resultRank}, (long[]){${dimsList.join(", ")}});`
  );
  for (let i = ndim - 1; i >= 0; i--) {
    lines.push(
      `for (long _jit_k_${i} = 0; _jit_k_${i} < _jit_n_${i}; _jit_k_${i}++) {`
    );
  }
  lines.push(
    `long _jit_src_off = ${formatNdOffset(slotSrc, j => `${baseCName}.dims[${j}]`)};`
  );
  lines.push(
    `long _jit_dst_off = ${formatNdOffset(
      Array.from({ length: ndim }, (_, i) => `_jit_k_${i}`),
      j => `_jit_n_${j}`
    )};`
  );
  lines.push(copyElem("_jit_dst_off", "_jit_src_off"));
  for (let i = ndim - 1; i >= 0; i--) {
    lines.push(`}`);
  }
  for (const c of cleanups) lines.push(c);
  lines.push(`_jit_t;`);
  return `({ ${lines.join(" ")} })`;
}

/** Emit an `IndexSliceStore` statement: mutate `base` in place.
 *  Member-rooted stores target a field slot `<rootCName>.<fieldPath>`
 *  with the field's NumericType replacing the (struct/class) base
 *  type for offset / complex-lane decisions. */
export function emitIndexSliceStore(
  s: Extract<IRStmt, { kind: "IndexSliceStore" }>,
  indent: string,
  state: RuntimeState
): string {
  const baseCName =
    s.fieldPath !== undefined
      ? `${s.base.cName}.${s.fieldPath.join(".")}`
      : s.base.cName;
  const slotTy = s.leafTy ?? s.base.ty;
  const rhsIsScalar = isScalar(s.rhs.ty);
  const baseIsComplex = isNumeric(slotTy) && slotTy.isComplex;
  const rhsIsComplex = isNumeric(s.rhs.ty) && s.rhs.ty.isComplex;
  if (baseIsComplex) {
    useRuntimeByName(state, "jit_cscalar");
  }
  // Per-element write template — handles 4 cases:
  //   real base                 → write real lane only
  //   complex base, real rhs    → real lane = src, imag lane = 0
  //   complex base, complex rhs scalar → split via creal/cimag
  //   complex base, complex rhs tensor → copy both lanes from source
  // The `src*` arguments name the per-iteration C expressions for
  // accessing the RHS (either a scalar local or `rhsTensor.real[k]`/
  // `rhsTensor.imag[k]`).
  const writeAt = (
    dstK: string,
    srcReal: string,
    srcImag: string | undefined
  ): string => {
    if (!baseIsComplex) {
      return `${baseCName}.real[${dstK}] = ${srcReal};`;
    }
    const imagExpr = srcImag !== undefined ? srcImag : "0.0";
    return (
      `${baseCName}.real[${dstK}] = ${srcReal}; ` +
      `${baseCName}.imag[${dstK}] = ${imagExpr};`
    );
  };
  const lines: string[] = [];
  lines.push(`${indent}{`);

  if (s.index.length === 1) {
    const slot = s.index[0];
    let dstOffsetFor: (kVar: string) => string;
    let linearStoreCleanup: string | null = null;
    if (slot.kind === "Colon") {
      lines.push(
        `${indent}  long _jit_n = ${dimsProductExpr(baseCName, slotTy)};`
      );
      dstOffsetFor = k => k;
    } else if (slot.kind === "Range") {
      dstOffsetFor = emitLinearRangeSetup(
        slot,
        baseCName,
        lines,
        `${indent}  `,
        state,
        /*grow=*/ true
      );
    } else if (slot.kind === "LogicalMask") {
      // Single-slot linear logical-mask write: precompute the buffer of
      // 0-based truthy positions, then walk it.
      const linMask = emitLinearLogicalMaskSetup(
        slot,
        baseCName,
        slotTy,
        lines,
        `${indent}  `,
        state
      );
      dstOffsetFor = linMask.srcIndexFor;
      linearStoreCleanup = linMask.cleanup;
    } else {
      throw new Error(
        "emit internal: single-slot Scalar IndexSliceStore should have routed to IndexStore"
      );
    }

    if (rhsIsScalar) {
      const rhsExpr = emitExpr(s.rhs, state);
      let srcReal: string;
      let srcImag: string | undefined;
      if (rhsIsComplex) {
        lines.push(`${indent}  double _Complex _jit_rhs = ${rhsExpr};`);
        srcReal = "jit_creal(_jit_rhs)";
        srcImag = "jit_cimag(_jit_rhs)";
      } else {
        lines.push(`${indent}  double _jit_rhs = ${rhsExpr};`);
        srcReal = "_jit_rhs";
        srcImag = undefined;
      }
      lines.push(
        `${indent}  for (long _jit_k = 0; _jit_k < _jit_n; _jit_k++) {`
      );
      lines.push(`${indent}    long _jit_dst = ${dstOffsetFor("_jit_k")};`);
      lines.push(`${indent}    ${writeAt("_jit_dst", srcReal, srcImag)}`);
      lines.push(`${indent}  }`);
    } else {
      if (s.rhs.kind !== "Var") {
        throw new Error(
          `emit internal: IndexSliceStore tensor RHS must be a Var (got ${s.rhs.kind})`
        );
      }
      emitTensorRhsSizeCheck(s.rhs, baseCName, slotTy, lines, `${indent}  `);
      lines.push(
        `${indent}  for (long _jit_k = 0; _jit_k < _jit_n; _jit_k++) {`
      );
      lines.push(`${indent}    long _jit_dst = ${dstOffsetFor("_jit_k")};`);
      // RHS imag lane: prefer the source's imag when it has one;
      // otherwise write 0 (real-typed tensor RHS into complex base).
      const srcImag =
        baseIsComplex && rhsIsComplex
          ? `${s.rhs.cName}.imag[_jit_k]`
          : undefined;
      lines.push(
        `${indent}    ${writeAt("_jit_dst", `${s.rhs.cName}.real[_jit_k]`, srcImag)}`
      );
      lines.push(`${indent}  }`);
    }
    if (linearStoreCleanup !== null) lines.push(linearStoreCleanup);
    lines.push(`${indent}}`);
    return lines.join("\n");
  }

  // Multi-slot.
  const ndim = s.index.length;
  const slotDst = emitSliceSlotSetup(
    state,
    lines,
    `${indent}  `,
    s.index,
    baseCName,
    undefined,
    /*grow=*/ true
  );
  const totalParts: string[] = [];
  for (let i = 0; i < ndim; i++) totalParts.push(`_jit_n_${i}`);
  lines.push(`${indent}  long _jit_n = ${totalParts.join(" * ")};`);

  let scalarSrcReal: string | undefined;
  let scalarSrcImag: string | undefined;
  if (rhsIsScalar) {
    const rhsExpr = emitExpr(s.rhs, state);
    if (rhsIsComplex) {
      lines.push(`${indent}  double _Complex _jit_rhs = ${rhsExpr};`);
      scalarSrcReal = "jit_creal(_jit_rhs)";
      scalarSrcImag = "jit_cimag(_jit_rhs)";
    } else {
      lines.push(`${indent}  double _jit_rhs = ${rhsExpr};`);
      scalarSrcReal = "_jit_rhs";
    }
  } else {
    if (s.rhs.kind !== "Var") {
      throw new Error(
        `emit internal: IndexSliceStore tensor RHS must be a Var (got ${s.rhs.kind})`
      );
    }
    emitTensorRhsSizeCheck(s.rhs, baseCName, slotTy, lines, `${indent}  `);
  }

  for (let i = ndim - 1; i >= 0; i--) {
    const ind = "  ".repeat(ndim - 1 - i);
    lines.push(
      `${indent}  ${ind}for (long _jit_k_${i} = 0; _jit_k_${i} < _jit_n_${i}; _jit_k_${i}++) {`
    );
  }
  const innerInd = "  ".repeat(ndim);
  lines.push(
    `${indent}  ${innerInd}long _jit_dst = ${formatNdOffset(slotDst, j => `${baseCName}.dims[${j}]`)};`
  );
  if (rhsIsScalar) {
    lines.push(
      `${indent}  ${innerInd}${writeAt("_jit_dst", scalarSrcReal!, scalarSrcImag)}`
    );
  } else {
    const rhs = s.rhs as Extract<IRExpr, { kind: "Var" }>;
    lines.push(
      `${indent}  ${innerInd}long _jit_k = ${formatNdOffset(
        Array.from({ length: ndim }, (_, i) => `_jit_k_${i}`),
        j => `_jit_n_${j}`
      )};`
    );
    const srcImag =
      baseIsComplex && rhsIsComplex ? `${rhs.cName}.imag[_jit_k]` : undefined;
    lines.push(
      `${indent}  ${innerInd}${writeAt("_jit_dst", `${rhs.cName}.real[_jit_k]`, srcImag)}`
    );
  }
  for (let i = ndim - 1; i >= 0; i--) {
    const ind = "  ".repeat(ndim - 1 - i);
    lines.push(`${indent}  ${ind}}`);
  }
  lines.push(`${indent}}`);
  return lines.join("\n");
}

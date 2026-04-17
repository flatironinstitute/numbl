/**
 * Shared fusible-operation name sets for JIT fusion analysis.
 *
 * Both the C-JIT and JS-JIT fusion paths use these to determine which
 * tensor Call nodes are fusible element-wise unary ops or absorbable
 * trailing reductions. The numeric op codes live in their respective
 * backend files (cFeasibility.ts for C, jitHelpersTensor.ts for JS).
 */

/** Tensor unary builtins fusible into per-element loops. */
export const FUSIBLE_TENSOR_UNARY_OPS: ReadonlySet<string> = new Set([
  "exp",
  "abs",
  "floor",
  "ceil",
  "round",
  "fix",
  "sin",
  "cos",
  "tan",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "sign",
]);

/**
 * JS-JIT-safe subset: excludes transcendentals (exp, sin, cos, tan, etc.)
 * which V8 can't SIMD-vectorize. Fusing these into a scalar per-element
 * loop is slower than calling libnumbl_ops per-op (which uses -fopenmp-simd).
 * The C-JIT uses the full set because GCC/Clang vectorize via #pragma omp simd.
 */
export const FUSIBLE_TENSOR_UNARY_OPS_JS: ReadonlySet<string> = new Set([
  "abs",
  "floor",
  "ceil",
  "round",
  "fix",
  "sign",
]);

/**
 * Two-argument tensor element-wise builtins fusible into per-element loops.
 * These are parsed as Call nodes (not Binary nodes) and need separate
 * recognition in isPureElementwise / emitScalarExpr.
 */
export const FUSIBLE_TENSOR_BINARY_OPS: ReadonlySet<string> = new Set([
  "max",
  "min",
  "mod",
  "rem",
  "atan2",
  "hypot",
]);

/** Tensor reduction builtins absorbable as trailing reductions. */
export const FUSIBLE_TENSOR_REDUCTION_OPS: ReadonlySet<string> = new Set([
  "sum",
  "prod",
  "max",
  "min",
  "any",
  "all",
  "mean",
]);

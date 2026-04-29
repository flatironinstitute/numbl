/**
 * C-JIT builtin registry — the single source of truth for which
 * builtins each C-JIT context can emit.
 *
 * Two contexts:
 *
 *   - **Elementwise** (`c-jit-fuse`, `c-jit-chain`): unary real-math
 *     builtins that compile to a single C function call inside an
 *     element-wise loop. Real-only — no complex args.
 *
 *   - **Scalar loop** (`c-jit-loop`): a superset that adds `atan2`
 *     (binary) and the complex projection builtins (`real`, `imag`,
 *     `conj`, which accept either real or complex args).
 *
 * Adding a new builtin: extend the relevant set here, make sure the
 * C name is right (override `cBuiltinName` if it differs), and add a
 * test or benchmark that exercises it. See
 * `docs/developer_reference/jit/cjit-substrate.md` for the broader
 * substrate contract.
 */

/** Real-only unary math builtins emittable in an element-wise loop. */
export const ELEMWISE_REAL_BUILTINS: ReadonlySet<string> = new Set([
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "exp",
  "log",
  "log2",
  "log10",
  "sqrt",
  "abs",
  "floor",
  "ceil",
  "round",
]);

/** Real-only math builtins emittable in a scalar loop body.
 *  Superset of the elementwise set; adds `atan2` (binary). */
export const LOOP_REAL_MATH_BUILTINS: ReadonlySet<string> = new Set([
  ...ELEMWISE_REAL_BUILTINS,
  "atan2",
]);

/** Projection builtins that accept real OR complex args. */
export const LOOP_COMPLEX_PROJECTION_BUILTINS: ReadonlySet<string> = new Set([
  "real",
  "imag",
  "conj",
]);

/** Map a numbl/MATLAB builtin name to its math.h C name. Identity for
 *  most; `abs` maps to `fabs` because C's `abs` is integer-typed. */
export function cBuiltinName(name: string): string {
  if (name === "abs") return "fabs";
  return name;
}

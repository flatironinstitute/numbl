/**
 * Op-code enums for the tensor-ops layer.
 *
 * SINGLE SOURCE OF TRUTH on the TS side. Mirrors native/ops/numbl_ops.h.
 * Drift detection: src/__tests__/op-codes-sync.test.ts compares this file's
 * values against numbl_dump_op_codes() at CI time.
 */

/**
 * Binary element-wise ops, one numbering for real and complex operands.
 * Codes below `NATIVE_BIN_OP_COUNT` have native kernels (real and complex,
 * `NUMBL_REAL_BIN_*` / `NUMBL_COMPLEX_BIN_*` in numbl_ops.h); the rest run
 * only in the TypeScript kernel layer (`ops/elementwise.ts`).
 */
export const OpBin = {
  ADD: 0,
  SUB: 1,
  MUL: 2,
  DIV: 3,
  POW: 4,
  MIN: 5,
  MAX: 6,
  MOD: 7,
  REM: 8,
  ATAN2: 9,
  HYPOT: 10,
} as const;

/** Binary ops with a native (and TS bridge) kernel: ADD, SUB, MUL, DIV. */
export const NATIVE_BIN_OP_COUNT = 4;

/** Flat reduction ops. */
export const OpReduce = {
  SUM: 0,
  PROD: 1,
  MAX: 2,
  MIN: 3,
  ANY: 4,
  ALL: 5,
  MEAN: 6,
} as const;

/**
 * Logical-result binary ops. EQ..GE have native kernels (real and complex);
 * AND, OR and XOR run only in the TypeScript kernel layer.
 */
export const OpCmp = {
  EQ: 0,
  NE: 1,
  LT: 2,
  LE: 3,
  GT: 4,
  GE: 5,
  AND: 6,
  OR: 7,
  XOR: 8,
} as const;

/** Logical-result ops with a native kernel: EQ..GE. */
export const NATIVE_CMP_OP_COUNT = 6;

/** Bessel function ops (real input). H-variants are in numbl_bessel_h (complex output). */
export const OpBessel = {
  J: 0,
  Y: 1,
  I: 2,
  K: 3,
} as const;

/** Unary element-wise ops (shared numbering for real + complex). */
export const OpUnary = {
  EXP: 0,
  LOG: 1,
  LOG2: 2,
  LOG10: 3,
  SQRT: 4,
  ABS: 5,
  FLOOR: 6,
  CEIL: 7,
  ROUND: 8,
  TRUNC: 9,
  SIN: 10,
  COS: 11,
  TAN: 12,
  ASIN: 13,
  ACOS: 14,
  ATAN: 15,
  SINH: 16,
  COSH: 17,
  TANH: 18,
  SIGN: 19,
} as const;

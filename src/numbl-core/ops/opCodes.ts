/**
 * Op-code enums for the tensor-ops layer.
 *
 * SINGLE SOURCE OF TRUTH on the TS side. Mirrors native/ops/numbl_ops.h.
 * Drift detection: src/__tests__/op-codes-sync.test.ts compares this file's
 * values against numbl_dump_op_codes() at CI time.
 */

/** Real binary element-wise ops. */
export const OpRealBin = {
  ADD: 0,
  SUB: 1,
  MUL: 2,
  DIV: 3,
} as const;

/** Complex binary element-wise ops. */
export const OpComplexBin = {
  ADD: 0,
  SUB: 1,
  MUL: 2,
  DIV: 3,
} as const;

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

/** Comparison ops (logical output). */
export const OpCmp = {
  EQ: 0,
  NE: 1,
  LT: 2,
  LE: 3,
  GT: 4,
  GE: 5,
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

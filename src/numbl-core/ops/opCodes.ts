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

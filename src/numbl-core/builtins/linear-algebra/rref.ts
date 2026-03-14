/**
 * rref (reduced row echelon form) builtin function
 *
 * R = rref(A)           — reduced row echelon form
 * [R, pivots] = rref(A) — also return pivot column indices
 * R = rref(A, tol)      — with custom tolerance
 *
 * Uses Gauss-Jordan elimination with partial pivoting.
 */

import {
  colMajorIndex,
  RTV,
  RuntimeError,
  tensorSize2D,
  toNumber,
} from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register } from "../registry.js";
import { isNum, isTensor, isFullyUnknown } from "../../lowering/itemTypes.js";
import { out, unknownMatrix } from "./check-helpers.js";

export function registerRref(): void {
  register("rref", [
    {
      check: (argTypes, nargout) => {
        if (nargout < 1 || nargout > 2) return null;
        if (argTypes.length < 1 || argTypes.length > 2) return null;

        const A = argTypes[0];
        if (nargout === 1) {
          if (isFullyUnknown(A)) return out(unknownMatrix());
          if (isNum(A) === true) return out(unknownMatrix());
          if (isTensor(A) !== true) return null;
          return out(unknownMatrix());
        }
        // nargout === 2: [R, pivots]
        if (isFullyUnknown(A)) return out(unknownMatrix(), unknownMatrix());
        if (isNum(A) === true) return out(unknownMatrix(), unknownMatrix());
        if (isTensor(A) !== true) return null;
        return out(unknownMatrix(), unknownMatrix());
      },
      apply: (args, nargout) => {
        if (args.length < 1 || args.length > 2)
          throw new RuntimeError("rref requires 1 or 2 arguments");

        const A = args[0];

        // Scalar case
        if (isRuntimeNumber(A)) {
          const val = A as number;
          if (val === 0) {
            if (nargout === 2) {
              return [
                RTV.tensor(new FloatXArray([0]), [1, 1]),
                RTV.tensor(new FloatXArray(0), [1, 0]),
              ];
            }
            return RTV.tensor(new FloatXArray([0]), [1, 1]);
          }
          if (nargout === 2) {
            return [
              RTV.tensor(new FloatXArray([1]), [1, 1]),
              RTV.tensor(new FloatXArray([1]), [1, 1]),
            ];
          }
          return RTV.tensor(new FloatXArray([1]), [1, 1]);
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("rref: argument must be numeric");

        const [m, n] = tensorSize2D(A);

        // Copy data to working array (column-major)
        const R = new Float64Array(m * n);
        for (let i = 0; i < m * n; i++) {
          R[i] = A.data[i];
        }

        // Determine tolerance
        let tol: number;
        if (args.length >= 2) {
          tol = toNumber(args[1]);
        } else {
          // Default tolerance: max(m, n) * eps * max(abs(A))
          let maxAbs = 0;
          for (let i = 0; i < R.length; i++) {
            const v = Math.abs(R[i]);
            if (v > maxAbs) maxAbs = v;
          }
          tol =
            Math.max(m, n) *
            Math.pow(
              2,
              Math.floor(Math.log2(Math.max(maxAbs, Number.MIN_VALUE))) - 52
            ) *
            maxAbs;
        }

        // Gauss-Jordan elimination with partial pivoting
        const pivotCols: number[] = [];
        let pivotRow = 0;

        for (let col = 0; col < n && pivotRow < m; col++) {
          // Find the row with the largest absolute value in this column
          let maxVal = 0;
          let maxRow = -1;
          for (let row = pivotRow; row < m; row++) {
            const v = Math.abs(R[colMajorIndex(row, col, m)]);
            if (v > maxVal) {
              maxVal = v;
              maxRow = row;
            }
          }

          if (maxVal <= tol) {
            // Zero out the column below tolerance
            for (let row = pivotRow; row < m; row++) {
              R[colMajorIndex(row, col, m)] = 0;
            }
            continue;
          }

          // Record pivot column (1-based for MATLAB compatibility)
          pivotCols.push(col + 1);

          // Swap rows
          if (maxRow !== pivotRow) {
            for (let j = 0; j < n; j++) {
              const tmp = R[colMajorIndex(pivotRow, j, m)];
              R[colMajorIndex(pivotRow, j, m)] = R[colMajorIndex(maxRow, j, m)];
              R[colMajorIndex(maxRow, j, m)] = tmp;
            }
          }

          // Scale the pivot row
          const pivot = R[colMajorIndex(pivotRow, col, m)];
          for (let j = 0; j < n; j++) {
            R[colMajorIndex(pivotRow, j, m)] /= pivot;
          }

          // Eliminate all other rows (both above and below — this is RREF)
          for (let row = 0; row < m; row++) {
            if (row === pivotRow) continue;
            const factor = R[colMajorIndex(row, col, m)];
            if (factor === 0) continue;
            for (let j = 0; j < n; j++) {
              R[colMajorIndex(row, j, m)] -=
                factor * R[colMajorIndex(pivotRow, j, m)];
            }
            // Ensure the eliminated entry is exactly zero
            R[colMajorIndex(row, col, m)] = 0;
          }

          pivotRow++;
        }

        const resultTensor = RTV.tensor(new FloatXArray(R), [m, n]);

        if (nargout === 2) {
          const pivotsData = new FloatXArray(pivotCols.length);
          for (let i = 0; i < pivotCols.length; i++) {
            pivotsData[i] = pivotCols[i];
          }
          return [resultTensor, RTV.tensor(pivotsData, [1, pivotCols.length])];
        }

        return resultTensor;
      },
    },
  ]);
}

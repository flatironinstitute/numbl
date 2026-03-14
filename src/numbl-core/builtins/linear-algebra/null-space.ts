/**
 * null (null space / kernel) builtin function
 *
 * Z = null(A)       — orthonormal basis for the null space of A
 * Z = null(A, 'r')  — rational basis via rref (not yet implemented)
 *
 * Uses SVD: columns of V corresponding to zero singular values.
 */

import {
  colMajorIndex,
  RTV,
  RuntimeError,
  RuntimeValue,
  tensorSize2D,
} from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import { getBuiltin } from "../registry.js";

/** Compute eps(x) — the distance from |x| to the next larger double */
function epsOf(x: number): number {
  if (!isFinite(x) || x === 0) return Number.EPSILON;
  const ax = Math.abs(x);
  return Math.pow(2, Math.floor(Math.log2(ax)) - 52);
}

export function registerNull(): void {
  register(
    "null",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("null requires 1 or 2 arguments");

      const A = args[0];

      // Scalar case: null(0) = 1, null(nonzero) = zeros(1,0)
      if (isRuntimeNumber(A)) {
        if (A === 0) {
          return RTV.tensor(new FloatXArray([1]), [1, 1]);
        }
        return RTV.tensor(new FloatXArray(0), [1, 0]);
      }

      if (!isRuntimeTensor(A))
        throw new RuntimeError("null: argument must be numeric");

      const [m, n] = tensorSize2D(A);

      // Get full SVD: [U, S, V] = svd(A)
      const svdBranches = getBuiltin("svd");
      if (!svdBranches) throw new RuntimeError("null: svd builtin not found");
      const svdResult = svdBranches[0].apply([A], 3);
      if (!Array.isArray(svdResult) || svdResult.length < 3)
        throw new RuntimeError("null: unexpected svd result");

      const S = svdResult[1] as RuntimeValue;
      const V = svdResult[2] as RuntimeValue;

      if (!isRuntimeTensor(S) || !isRuntimeTensor(V))
        throw new RuntimeError("null: unexpected svd result types");

      // Extract singular values from the diagonal of S
      const k = Math.min(m, n);
      const sVals = new Float64Array(k);
      const sRows = S.shape[0];
      for (let i = 0; i < k; i++) {
        sVals[i] = S.data[colMajorIndex(i, i, sRows)];
      }

      // Determine tolerance: max(size(A)) * eps(max(s))
      let sMax = 0;
      for (let i = 0; i < k; i++) {
        if (sVals[i] > sMax) sMax = sVals[i];
      }
      const tol = Math.max(m, n) * epsOf(sMax);

      // Count rank (number of singular values > tol)
      let r = 0;
      for (let i = 0; i < k; i++) {
        if (sVals[i] > tol) r++;
      }

      // Null space dimension
      const nullDim = n - r;

      if (nullDim === 0) {
        // Empty null space
        return RTV.tensor(new FloatXArray(0), [n, 0]);
      }

      // Extract the last nullDim columns of V (columns r..n-1)
      const Z_data = new FloatXArray(n * nullDim);
      for (let j = 0; j < nullDim; j++) {
        for (let i = 0; i < n; i++) {
          Z_data[colMajorIndex(i, j, n)] = V.data[colMajorIndex(i, r + j, n)];
        }
      }

      return RTV.tensor(Z_data, [n, nullDim]);
    })
  );
}

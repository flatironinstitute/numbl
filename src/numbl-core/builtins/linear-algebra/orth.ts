/**
 * orth (orthonormal basis for column space) builtin function
 *
 * Q = orth(A)       — orthonormal basis for range(A)
 * Q = orth(A, tol)  — with custom tolerance
 *
 * Uses SVD: columns of U corresponding to nonzero singular values.
 */

import {
  colMajorIndex,
  RTV,
  RuntimeError,
  RuntimeValue,
  tensorSize2D,
  toNumber,
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

export function registerOrth(): void {
  register(
    "orth",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("orth requires 1 or 2 arguments");

      const A = args[0];

      // Scalar case: orth(0) = zeros(1,0), orth(nonzero) = 1
      if (isRuntimeNumber(A)) {
        if (A === 0) {
          return RTV.tensor(new FloatXArray(0), [1, 0]);
        }
        return RTV.tensor(new FloatXArray([1]), [1, 1]);
      }

      if (!isRuntimeTensor(A))
        throw new RuntimeError("orth: argument must be numeric");

      const [m, n] = tensorSize2D(A);

      // Get full SVD: [U, S, V] = svd(A)
      const svdBranches = getBuiltin("svd");
      if (!svdBranches) throw new RuntimeError("orth: svd builtin not found");
      const svdResult = svdBranches[0].apply([A], 3);
      if (!Array.isArray(svdResult) || svdResult.length < 3)
        throw new RuntimeError("orth: unexpected svd result");

      const U = svdResult[0] as RuntimeValue;
      const S = svdResult[1] as RuntimeValue;

      if (!isRuntimeTensor(U) || !isRuntimeTensor(S))
        throw new RuntimeError("orth: unexpected svd result types");

      // Extract singular values from the diagonal of S
      const k = Math.min(m, n);
      const sVals = new Float64Array(k);
      const sRows = S.shape[0];
      for (let i = 0; i < k; i++) {
        sVals[i] = S.data[colMajorIndex(i, i, sRows)];
      }

      // Determine tolerance
      let tol: number;
      if (args.length >= 2) {
        tol = toNumber(args[1]);
      } else {
        // Default: max(size(A)) * eps(max(s))
        let sMax = 0;
        for (let i = 0; i < k; i++) {
          if (sVals[i] > sMax) sMax = sVals[i];
        }
        tol = Math.max(m, n) * epsOf(sMax);
      }

      // Count rank (number of singular values > tol)
      let r = 0;
      for (let i = 0; i < k; i++) {
        if (sVals[i] > tol) r++;
      }

      if (r === 0) {
        // Zero matrix — empty column space
        return RTV.tensor(new FloatXArray(0), [m, 0]);
      }

      // Extract the first r columns of U
      const Q_data = new FloatXArray(m * r);
      for (let j = 0; j < r; j++) {
        for (let i = 0; i < m; i++) {
          Q_data[colMajorIndex(i, j, m)] = U.data[colMajorIndex(i, j, m)];
        }
      }

      return RTV.tensor(Q_data, [m, r]);
    })
  );
}

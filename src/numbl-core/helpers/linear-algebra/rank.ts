/**
 * Matrix rank builtin function
 *
 * k = rank(A)       — number of singular values > max(size(A)) * eps(norm(A))
 * k = rank(A, tol)  — number of singular values > tol
 */

import {
  RTV,
  RuntimeError,
  RuntimeValue,
  tensorSize2D,
  toNumber,
} from "../../runtime/index.js";
import {
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import { applyBuiltin as _applyBuiltin } from "../check-helpers.js";

function applyBuiltin(
  name: string,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue {
  return _applyBuiltin("rank", name, args, nargout);
}

/** Compute eps(x) — the distance from |x| to the next larger double */
function epsOf(x: number): number {
  if (!isFinite(x) || x === 0) return Number.EPSILON;
  const ax = Math.abs(x);
  // 2^(exponent - 52) where exponent = floor(log2(ax))
  return Math.pow(2, Math.floor(Math.log2(ax)) - 52);
}

export function registerRank(): void {
  register(
    "rank",
    builtinSingle(
      args => {
        if (args.length < 1 || args.length > 2)
          throw new RuntimeError("rank requires 1 or 2 arguments");

        const A = args[0];

        // Scalar case
        if (isRuntimeNumber(A)) {
          return RTV.num(A === 0 ? 0 : 1);
        }
        if (isRuntimeComplexNumber(A)) {
          return RTV.num(A.re === 0 && A.im === 0 ? 0 : 1);
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("rank: argument must be numeric");

        const [rows, cols] = tensorSize2D(A);

        // Get singular values via SVD
        const sVec = applyBuiltin("svd", [A], 1);
        if (!isRuntimeTensor(sVec))
          throw new RuntimeError("rank: unexpected svd result");
        if (sVec.imag)
          throw new RuntimeError("rank: singular values must be real");
        const s = sVec.data;

        // Determine tolerance
        let tol: number;
        if (args.length >= 2) {
          tol = toNumber(args[1]);
        } else {
          // Default: max(size(A)) * eps(norm(A))
          // norm(A) for rank uses the 2-norm = max singular value
          let sMax = 0;
          for (let i = 0; i < s.length; i++) {
            if (s[i] > sMax) sMax = s[i];
          }
          tol = Math.max(rows, cols) * epsOf(sMax);
        }

        // Count singular values larger than tol
        let k = 0;
        for (let i = 0; i < s.length; i++) {
          if (s[i] > tol) k++;
        }

        return RTV.num(k);
      },
      { outputType: { kind: "Number" } }
    )
  );
}

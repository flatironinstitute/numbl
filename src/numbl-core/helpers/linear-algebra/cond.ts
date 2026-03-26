/**
 * Condition number builtin function
 *
 * C = cond(A)     — 2-norm condition number (ratio of largest to smallest singular value)
 * C = cond(A, p)  — p-norm condition number, where p can be 1, 2, Inf, or 'fro'
 */

import {
  RTV,
  RuntimeError,
  RuntimeValue,
  tensorSize2D,
  toNumber,
} from "../../runtime/index.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import {
  applyBuiltin as _applyBuiltin,
  parseStringArgLower,
} from "../check-helpers.js";

function applyBuiltin(
  name: string,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue {
  return _applyBuiltin("cond", name, args, nargout);
}

export function registerCond(): void {
  register(
    "cond",
    builtinSingle(
      args => {
        if (args.length < 1 || args.length > 2)
          throw new RuntimeError("cond requires 1 or 2 arguments");

        const A = args[0];

        // Scalar case
        if (isRuntimeNumber(A)) {
          return RTV.num(A === 0 ? Infinity : 1);
        }
        if (isRuntimeComplexNumber(A)) {
          return RTV.num(A.re === 0 && A.im === 0 ? Infinity : 1);
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("cond: argument must be numeric");

        // Determine p
        let p: number | string = 2;
        if (args.length >= 2) {
          const pArg = args[1];
          if (isRuntimeString(pArg) || isRuntimeChar(pArg)) {
            const pStr = parseStringArgLower(pArg);
            if (pStr === "fro") {
              p = "fro";
            } else {
              throw new RuntimeError("cond: string argument must be 'fro'");
            }
          } else {
            p = toNumber(pArg);
          }
        }

        if (p === 2) {
          // 2-norm condition number via SVD: max(s) / min(s)
          const sVec = applyBuiltin("svd", [A], 1);
          if (!isRuntimeTensor(sVec))
            throw new RuntimeError("cond: unexpected svd result");
          if (sVec.imag)
            throw new RuntimeError("cond: singular values must be real");
          const s = sVec.data;
          let sMax = -Infinity;
          let sMin = Infinity;
          for (let i = 0; i < s.length; i++) {
            if (s[i] > sMax) sMax = s[i];
            if (s[i] < sMin) sMin = s[i];
          }
          if (sMin === 0) return RTV.num(Infinity);
          return RTV.num(sMax / sMin);
        }

        // For other norms: cond(A, p) = norm(A, p) * norm(inv(A), p)
        const [rows, cols] = tensorSize2D(A);
        if (rows !== cols)
          throw new RuntimeError(
            "cond: matrix must be square for non-2 condition number"
          );

        const normArg: RuntimeValue =
          p === "fro" ? RTV.string("fro") : RTV.num(p as number);

        const normA = toNumber(applyBuiltin("norm", [A, normArg], 1));
        const invA = applyBuiltin("inv", [A], 1);
        const normInvA = toNumber(applyBuiltin("norm", [invA, normArg], 1));

        return RTV.num(normA * normInvA);
      },
      { outputType: { kind: "Number" } }
    )
  );
}

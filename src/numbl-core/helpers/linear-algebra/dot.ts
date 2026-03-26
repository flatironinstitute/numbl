/**
 * Dot product builtin function
 */

import { RTV, RuntimeError } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register } from "../registry.js";
import { type ItemType, IType, isComplex } from "../../lowering/itemTypes.js";

export function registerDot(): void {
  register("dot", [
    {
      check: (argTypes: ItemType[], nargout: number) => {
        if (nargout !== 1) return null;
        // For tensor inputs, result could be scalar (vectors) or tensor
        // (matrices), so return Unknown.
        const hasTensor = argTypes.some(t => t.kind === "Tensor");
        if (hasTensor) return { outputTypes: [{ kind: "Unknown" }] };
        const inputIsComplex = argTypes.some(t => isComplex(t) === true);
        const outType: ItemType = inputIsComplex
          ? { ...IType.Complex }
          : IType.num();
        return { outputTypes: [outType] };
      },
      apply: args => {
        if (args.length !== 2)
          throw new RuntimeError("dot requires 2 arguments");
        const a = args[0],
          b = args[1];

        // Extract real/imag data and shape from each argument
        let aRe: FloatXArrayType | null = null;
        let aIm: FloatXArrayType | null = null;
        let aShape: number[] = [1, 1];
        let bRe: FloatXArrayType | null = null;
        let bIm: FloatXArrayType | null = null;

        if (isRuntimeTensor(a)) {
          aRe = a.data;
          aIm = a.imag ?? null;
          aShape = a.shape;
        } else if (isRuntimeNumber(a)) {
          aRe = new FloatXArray([a]);
        } else if (isRuntimeComplexNumber(a)) {
          aRe = new FloatXArray([a.re]);
          aIm = new FloatXArray([a.im]);
        }

        if (isRuntimeTensor(b)) {
          bRe = b.data;
          bIm = b.imag ?? null;
        } else if (isRuntimeNumber(b)) {
          bRe = new FloatXArray([b]);
        } else if (isRuntimeComplexNumber(b)) {
          bRe = new FloatXArray([b.re]);
          bIm = new FloatXArray([b.im]);
        }

        if (!aRe || !bRe)
          throw new RuntimeError("dot: arguments must be numeric");
        if (aRe.length !== bRe.length)
          throw new RuntimeError("dot: vectors must be same length");

        const hasComplex = aIm !== null || bIm !== null;

        // Determine if we should operate column-wise (matrix inputs)
        const rows = aShape[0];
        const cols = aShape.length >= 2 ? aShape[1] : 1;
        const isMatrix = rows > 1 && cols > 1;

        if (isMatrix) {
          // Column-wise dot products: result is 1 x cols
          const resultRe = new FloatXArray(cols);
          const resultIm = hasComplex ? new FloatXArray(cols) : null;
          for (let c = 0; c < cols; c++) {
            let sRe = 0,
              sIm = 0;
            for (let r = 0; r < rows; r++) {
              const idx = c * rows + r; // column-major
              const aRei = aRe[idx];
              const aImi = aIm ? aIm[idx] : 0;
              const bRei = bRe[idx];
              const bImi = bIm ? bIm[idx] : 0;
              // conj(a) .* b
              sRe += aRei * bRei + aImi * bImi;
              sIm += aRei * bImi - aImi * bRei;
            }
            resultRe[c] = sRe;
            if (resultIm) resultIm[c] = sIm;
          }
          return RTV.tensor(resultRe, [1, cols], resultIm ?? undefined);
        }

        // Vector or scalar case: return scalar
        if (!hasComplex) {
          let s = 0;
          for (let i = 0; i < aRe.length; i++) s += aRe[i] * bRe[i];
          return RTV.num(s);
        }

        let sRe = 0,
          sIm = 0;
        for (let i = 0; i < aRe.length; i++) {
          const aRei = aRe[i];
          const aImi = aIm ? aIm[i] : 0;
          const bRei = bRe[i];
          const bImi = bIm ? bIm[i] : 0;
          sRe += aRei * bRei + aImi * bImi;
          sIm += aRei * bImi - aImi * bRei;
        }
        if (sIm === 0) return RTV.num(sRe);
        return RTV.complex(sRe, sIm);
      },
    },
  ]);
}

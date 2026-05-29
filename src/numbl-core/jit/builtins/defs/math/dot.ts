/**
 * `dot(a, b)` — real or complex dot product.
 *
 * Numbl semantics
 * (`numbl-core/interpreter/builtins/linear-algebra.ts`):
 *   - Two same-length 1-D vectors (any combination of row / column /
 *     scalar) → scalar `sum_i a_i * b_i` (real) or
 *     `sum_i conj(a_i) * b_i` (complex).
 *   - Two matrices of the **same** shape M×N → column-wise dot,
 *     returned as a 1×N row vector.
 *   - Length / shape mismatch → runtime error.
 *   - Any complex operand promotes the result to complex (real
 *     tensors flow through with `imag = 0`).
 *
 * Real folding: when both inputs are exact and small enough, the
 * result is computed at type-check time and lands as `exact` on the
 * output type, so call sites used in `if`-conds get the static fold.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  isMultiElement,
  isNumeric,
  isScalar,
  scalarComplex,
  scalarDouble,
  shapeNumel,
  signFromNumber,
  tensorComplex,
  tensorDouble,
  type NumericType,
  type Type,
  typeToString,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  exactComplexArray,
  exactDouble,
  exactRealArray,
  exactScalarAsComplex,
} from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { isComplexValue } from "../../../runtime/value.js";
import {
  mtoc2_dot_real as jsDotReal,
  mtoc2_dot_real_matrix as jsDotRealMatrix,
  mtoc2_dot_complex as jsDotComplex,
  mtoc2_dot_complex_matrix as jsDotComplexMatrix,
} from "../../runtime/snippets.gen.js";

function requireNumeric(t: Type, what: string): NumericType {
  if (!isNumeric(t)) {
    throw new TypeError(`${what} must be a numeric (got ${typeToString(t)})`);
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(`${what} must be double or logical (got ${t.elem})`);
  }
  return t;
}

function isVectorLike(t: NumericType): boolean {
  if (isScalar(t)) return true;
  // 1-D vector means: 2-D with one axis exactly 1.
  if (t.dims.length !== 2) return false;
  const aOne = t.dims[0].kind === "exact" && t.dims[0].value === 1;
  const bOne = t.dims[1].kind === "exact" && t.dims[1].value === 1;
  return aOne || bOne;
}

function knownNumel(t: NumericType): number | undefined {
  return t.shape !== undefined ? shapeNumel(t.shape) : undefined;
}

function matrixCols(t: NumericType): number | undefined {
  if (t.shape === undefined || t.shape.length !== 2) return undefined;
  return t.shape[1];
}

function realScalar(re: number): NumericType {
  return Number.isFinite(re)
    ? scalarDouble(signFromNumber(re), re)
    : scalarDouble();
}

/** Compute `sum_i conj(a_i) * b_i` over two same-length complex
 *  buffers, returning `{re, im}`. Both lanes treated as zero when the
 *  corresponding array is undefined (a real tensor flowing through). */
function complexDotExact(
  aRe: Float64Array,
  aIm: Float64Array | undefined,
  bRe: Float64Array,
  bIm: Float64Array | undefined
): { re: number; im: number } {
  let accRe = 0;
  let accIm = 0;
  for (let i = 0; i < aRe.length; i++) {
    const aR = aRe[i];
    const aI = aIm !== undefined ? aIm[i] : 0;
    const bR = bRe[i];
    const bI = bIm !== undefined ? bIm[i] : 0;
    accRe += aR * bR + aI * bI;
    accIm += aR * bI - aI * bR;
  }
  return { re: accRe, im: accIm };
}

export const dot: Builtin = {
  name: "dot",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'dot' expects 2 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'dot' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = requireNumeric(argTypes[0], `'dot' arg 1`);
    const b = requireNumeric(argTypes[1], `'dot' arg 2`);
    const anyComplex = a.isComplex || b.isComplex;

    // Both scalar.
    if (isScalar(a) && isScalar(b)) {
      if (anyComplex) {
        const ax = exactScalarAsComplex(a);
        const bx = exactScalarAsComplex(b);
        if (ax !== undefined && bx !== undefined) {
          // conj(a) * b
          return [
            scalarComplex({
              re: ax.re * bx.re + ax.im * bx.im,
              im: ax.re * bx.im - ax.im * bx.re,
            }),
          ];
        }
        return [scalarComplex()];
      }
      const xa = exactDouble(a);
      const xb = exactDouble(b);
      if (xa !== undefined && xb !== undefined) {
        return [realScalar(xa * xb)];
      }
      return [scalarDouble()];
    }

    const aVec = isVectorLike(a);
    const bVec = isVectorLike(b);
    if (aVec && bVec) {
      const na = knownNumel(a);
      const nb = knownNumel(b);
      if (na !== undefined && nb !== undefined && na !== nb) {
        throw new TypeError(
          `'dot' vectors must be same length (got ${na} and ${nb})`
        );
      }
      if (anyComplex) {
        const aReArr = a.isComplex
          ? exactComplexArray(a)?.re
          : exactRealArray(a);
        const aImArr = a.isComplex ? exactComplexArray(a)?.im : undefined;
        const bReArr = b.isComplex
          ? exactComplexArray(b)?.re
          : exactRealArray(b);
        const bImArr = b.isComplex ? exactComplexArray(b)?.im : undefined;
        if (
          aReArr !== undefined &&
          bReArr !== undefined &&
          aReArr.length === bReArr.length &&
          aReArr.length <= EXACT_ARRAY_MAX_ELEMENTS
        ) {
          const cx = complexDotExact(aReArr, aImArr, bReArr, bImArr);
          return [scalarComplex(cx)];
        }
        return [scalarComplex()];
      }
      const arrA = exactRealArray(a);
      const arrB = exactRealArray(b);
      if (
        arrA !== undefined &&
        arrB !== undefined &&
        arrA.length === arrB.length &&
        arrA.length <= EXACT_ARRAY_MAX_ELEMENTS
      ) {
        let acc = 0;
        for (let i = 0; i < arrA.length; i++) acc += arrA[i] * arrB[i];
        return [realScalar(acc)];
      }
      return [scalarDouble()];
    }

    // Matrix form: both args are full matrices of the same shape.
    if (
      !aVec &&
      !bVec &&
      a.shape !== undefined &&
      b.shape !== undefined &&
      a.shape.length === 2 &&
      b.shape.length === 2 &&
      a.shape[0] === b.shape[0] &&
      a.shape[1] === b.shape[1]
    ) {
      const cols = matrixCols(a) ?? 0;
      if (anyComplex) {
        const aReArr = a.isComplex
          ? exactComplexArray(a)?.re
          : exactRealArray(a);
        const aImArr = a.isComplex ? exactComplexArray(a)?.im : undefined;
        const bReArr = b.isComplex
          ? exactComplexArray(b)?.re
          : exactRealArray(b);
        const bImArr = b.isComplex ? exactComplexArray(b)?.im : undefined;
        if (
          aReArr !== undefined &&
          bReArr !== undefined &&
          aReArr.length === bReArr.length &&
          aReArr.length <= EXACT_ARRAY_MAX_ELEMENTS
        ) {
          const rows = a.shape[0];
          const outRe = new Float64Array(cols);
          const outIm = new Float64Array(cols);
          for (let j = 0; j < cols; j++) {
            let accRe = 0;
            let accIm = 0;
            for (let i = 0; i < rows; i++) {
              const off = j * rows + i;
              const aR = aReArr[off];
              const aI = aImArr !== undefined ? aImArr[off] : 0;
              const bR = bReArr[off];
              const bI = bImArr !== undefined ? bImArr[off] : 0;
              accRe += aR * bR + aI * bI;
              accIm += aR * bI - aI * bR;
            }
            outRe[j] = accRe;
            outIm[j] = accIm;
          }
          return [tensorComplex([1, cols], { re: outRe, im: outIm })];
        }
        return [tensorComplex([1, cols])];
      }
      const arrA = exactRealArray(a);
      const arrB = exactRealArray(b);
      if (
        arrA !== undefined &&
        arrB !== undefined &&
        arrA.length === arrB.length &&
        arrA.length <= EXACT_ARRAY_MAX_ELEMENTS
      ) {
        const rows = a.shape[0];
        const out = new Float64Array(cols);
        for (let j = 0; j < cols; j++) {
          let acc = 0;
          for (let i = 0; i < rows; i++) {
            const off = j * rows + i;
            acc += arrA[off] * arrB[off];
          }
          out[j] = acc;
        }
        return [tensorDouble([1, cols], out)];
      }
      return [tensorDouble([1, cols])];
    }

    throw new UnsupportedConstruct(
      `'dot' supports two vectors of the same length, or two matrices of the same shape ` +
        `(got ${typeToString(a)} and ${typeToString(b)})`
    );
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    const anyComplex = a.isComplex || b.isComplex;
    if (isMultiElement(a) && isMultiElement(b)) {
      useRuntime("mtoc2_dot_real");
      if (anyComplex) useRuntime("mtoc2_cscalar");
      const helper = anyComplex
        ? isVectorLike(a) || isVectorLike(b)
          ? "mtoc2_dot_complex"
          : "mtoc2_dot_complex_matrix"
        : isVectorLike(a) || isVectorLike(b)
          ? "mtoc2_dot_real"
          : "mtoc2_dot_real_matrix";
      return `${helper}(${argsC[0]}, ${argsC[1]})`;
    }
    // Scalar/scalar. For complex, emit `conj(a) * b` directly via
    // mtoc2_cmul + mtoc2_cconj; for the real path, plain `a * b`.
    if (anyComplex) {
      useRuntime("mtoc2_cscalar");
      const aC = a.isComplex
        ? argsC[0]
        : `mtoc2_cmake((double)(${argsC[0]}), 0.0)`;
      const bC = b.isComplex
        ? argsC[1]
        : `mtoc2_cmake((double)(${argsC[1]}), 0.0)`;
      return `mtoc2_cmul(mtoc2_cconj(${aC}), ${bC})`;
    }
    return `((${argsC[0]}) * (${argsC[1]}))`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    const anyComplex = a.isComplex || b.isComplex;
    if (isMultiElement(a) && isMultiElement(b)) {
      useRuntime("mtoc2_dot_real");
      const helper = anyComplex
        ? isVectorLike(a) || isVectorLike(b)
          ? "mtoc2_dot_complex"
          : "mtoc2_dot_complex_matrix"
        : isVectorLike(a) || isVectorLike(b)
          ? "mtoc2_dot_real"
          : "mtoc2_dot_real_matrix";
      return `${helper}(${argsJs[0]}, ${argsJs[1]})`;
    }
    if (anyComplex) {
      useRuntime("mtoc2_cscalar");
      const aJs = a.isComplex ? argsJs[0] : `mtoc2_cmake(${argsJs[0]}, 0)`;
      const bJs = b.isComplex ? argsJs[1] : `mtoc2_cmake(${argsJs[1]}, 0)`;
      return `mtoc2_cmul(mtoc2_cconj(${aJs}), ${bJs})`;
    }
    return `((${argsJs[0]}) * (${argsJs[1]}))`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    const anyComplex = a.isComplex || b.isComplex;
    if (isMultiElement(a) && isMultiElement(b)) {
      if (anyComplex) {
        if (isVectorLike(a) || isVectorLike(b)) {
          return [
            jsDotComplex(
              args[0] as RuntimeTensor,
              args[1] as RuntimeTensor
            ) as unknown as { re: number; im: number },
          ];
        }
        return [
          jsDotComplexMatrix(
            args[0] as RuntimeTensor,
            args[1] as RuntimeTensor
          ) as unknown as RuntimeTensor,
        ];
      }
      if (isVectorLike(a) || isVectorLike(b)) {
        return [
          jsDotReal(
            args[0] as RuntimeTensor,
            args[1] as RuntimeTensor
          ) as number,
        ];
      }
      return [
        jsDotRealMatrix(
          args[0] as RuntimeTensor,
          args[1] as RuntimeTensor
        ) as unknown as RuntimeTensor,
      ];
    }
    // Scalar / scalar (including a scalar vs length-1 tensor).
    const va = args[0];
    const vb = args[1];
    if (anyComplex) {
      const toCx = (v: unknown): { re: number; im: number } => {
        if (isComplexValue(v as never)) return v as { re: number; im: number };
        const re = typeof v === "number" ? v : Number(v);
        return { re, im: 0 };
      };
      const ax = toCx(va);
      const bx = toCx(vb);
      return [
        {
          re: ax.re * bx.re + ax.im * bx.im,
          im: ax.re * bx.im - ax.im * bx.re,
        },
      ];
    }
    const xa = typeof va === "number" ? va : Number(va);
    const xb = typeof vb === "number" ? vb : Number(vb);
    return [xa * xb];
  },
};

/**
 * `isinf(x)` / `isfinite(x)` — elementwise tests returning logical
 * 1 / 0. Real and complex inputs both supported; for complex, `isinf`
 * is true if either component is infinite and `isfinite` requires
 * both components to be finite (matches numbl / MATLAB).
 */
import { defineUnaryPred } from "./_unary_pred.js";

const isInfNumber = (x: number): boolean => x === Infinity || x === -Infinity;

export const isinf = defineUnaryPred({
  name: "isinf",
  // C99 `isinf(x)` returns nonzero for ±Infinity but the sign of
  // the return value is implementation-defined (glibc returns -1 for
  // -Inf). Normalize to 0/1 with `!= 0`.
  cScalar: arg => `(isinf(${arg}) != 0)`,
  jsScalar: arg => `(Math.abs(${arg}) === Infinity)`,
  jsFn: isInfNumber,
  tensorHelper: "mtoc2_tensor_predicate",
  complex: {
    cScalarComplex: (re, im) => `isinf(${re}) || isinf(${im})`,
    jsScalarComplex: (re, im) =>
      `Math.abs(${re}) === Infinity || Math.abs(${im}) === Infinity`,
    jsFnComplex: (re, im) => isInfNumber(re) || isInfNumber(im),
    tensorHelperComplex: "mtoc2_tensor_isinf_complex",
  },
});

export const isfinite = defineUnaryPred({
  name: "isfinite",
  cScalar: arg => `isfinite(${arg})`,
  jsScalar: arg => `Number.isFinite(${arg})`,
  jsFn: Number.isFinite,
  tensorHelper: "mtoc2_tensor_predicate",
  complex: {
    cScalarComplex: (re, im) => `isfinite(${re}) && isfinite(${im})`,
    jsScalarComplex: (re, im) =>
      `Number.isFinite(${re}) && Number.isFinite(${im})`,
    jsFnComplex: (re, im) => Number.isFinite(re) && Number.isFinite(im),
    tensorHelperComplex: "mtoc2_tensor_isfinite_complex",
  },
});

/**
 * Complex number builtins: real, imag, conj, angle.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
} from "../../runtime/types.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { builtin } from "./builtin.js";
import { numericArg } from "./args.js";
import { unaryRealResult, mkc, makeTensor } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

// ── real ────────────────────────────────────────────────────────────────

builtin({
  name: "real",
  nargin: 1,
  apply: unaryRealResult(
    x => x,
    re => re,
    "real"
  ),
});

// ── imag ────────────────────────────────────────────────────────────────

builtin({
  name: "imag",
  nargin: 1,
  apply: unaryRealResult(
    () => 0,
    (_re, im) => im,
    "imag"
  ),
});

// ── conj ────────────────────────────────────────────────────────────────

builtin({
  name: "conj",
  nargin: 1,
  apply: args => {
    let v = numericArg(args[0]);
    if (typeof v === "number") return v;
    if (isRuntimeComplexNumber(v)) return mkc(v.re, -v.im);
    if (isRuntimeSparseMatrix(v)) v = sparseToDense(v);
    const n = v.data.length;
    const outR = allocFloat64Array(n);
    outR.set(v.data);
    if (!v.imag) return makeTensor(outR, undefined, v.shape.slice());
    const outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) outI[i] = -v.imag[i];
    return makeTensor(outR, outI, v.shape.slice());
  },
});

// ── angle ───────────────────────────────────────────────────────────────

builtin({
  name: "angle",
  nargin: 1,
  mfile: true,
  apply: unaryRealResult(
    x => (isNaN(x) ? NaN : x >= 0 ? 0 : Math.PI),
    (re, im) => Math.atan2(im, re),
    "angle"
  ),
});

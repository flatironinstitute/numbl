/**
 * Complex number builtins: real, imag, conj, angle.
 */

import {
  FloatXArray,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import {
  defineBuiltin,
  unaryRealResultCases,
  type BuiltinCase,
  mkc,
  makeTensor,
  scalarConstantJitEmitC,
  scalarIdentityJitEmitC,
} from "./types.js";

// ── real ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "real",
  cases: unaryRealResultCases(
    x => x,
    re => re,
    "real"
  ),
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    if (k === "number" || k === "boolean") return `(+${argCode[0]})`;
    if (k === "complex_or_number") return `$h.re(${argCode[0]})`;
    return null;
  },
  jitEmitC: scalarIdentityJitEmitC(),
});

// ── imag ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "imag",
  cases: unaryRealResultCases(
    () => 0,
    (_re, im) => im,
    "imag"
  ),
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    if (k === "number" || k === "boolean") return "0";
    if (k === "complex_or_number") return `$h.im(${argCode[0]})`;
    return null;
  },
  jitEmitC: scalarConstantJitEmitC({ number: "0.0", boolean: "0.0" }),
});

// ── conj ────────────────────────────────────────────────────────────────

const conjCases: BuiltinCase[] = [
  {
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      const a = argTypes[0];
      if (a.kind === "number" || a.kind === "boolean")
        return [{ kind: "number" }];
      return null;
    },
    apply: args => {
      const v = args[0];
      if (typeof v === "boolean") return v ? 1 : 0;
      return v;
    },
  },
  {
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      if (argTypes[0].kind !== "complex_or_number") return null;
      return [{ kind: "complex_or_number" }];
    },
    apply: args => {
      const v = args[0];
      if (typeof v === "number") return v;
      if (isRuntimeComplexNumber(v)) return mkc(v.re, -v.im);
      throw new Error("conj: unsupported argument type");
    },
  },
  {
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      const a = argTypes[0];
      if (a.kind !== "tensor") return null;
      return [
        {
          kind: "tensor",
          isComplex: a.isComplex,
          shape: a.shape,
          ndim: a.ndim,
        },
      ];
    },
    apply: args => {
      const v = args[0];
      if (!isRuntimeTensor(v))
        throw new Error("conj: unsupported argument type");
      const n = v.data.length;
      const outR = new FloatXArray(n);
      outR.set(v.data);
      if (!v.imag) return makeTensor(outR, undefined, v.shape.slice());
      const outI = new FloatXArray(n);
      for (let i = 0; i < n; i++) outI[i] = -v.imag[i];
      return makeTensor(outR, outI, v.shape.slice());
    },
  },
  {
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      const a = argTypes[0];
      if (a.kind !== "sparse_matrix") return null;
      const shape =
        a.m !== undefined && a.n !== undefined ? [a.m, a.n] : undefined;
      return [{ kind: "tensor", isComplex: a.isComplex, shape }];
    },
    apply: args => {
      const v = args[0];
      if (isRuntimeSparseMatrix(v))
        return conjCases[2].apply([sparseToDense(v)], 1);
      throw new Error("conj: unsupported argument type");
    },
  },
];

defineBuiltin({
  name: "conj",
  cases: conjCases,
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    if (k === "number" || k === "boolean") return argCode[0];
    if (k === "complex_or_number") return `$h.cConj(${argCode[0]})`;
    return null;
  },
  jitEmitC: scalarIdentityJitEmitC(),
});

// ── angle ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "angle",
  cases: unaryRealResultCases(
    x => (isNaN(x) ? NaN : x >= 0 ? 0 : Math.PI),
    (re, im) => Math.atan2(im, re),
    "angle"
  ),
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    if (k === "number")
      return `(Number.isNaN(${argCode[0]}) ? NaN : ${argCode[0]} >= 0 ? 0 : Math.PI)`;
    if (k === "boolean") return "0";
    if (k === "complex_or_number") return `$h.cAngle(${argCode[0]})`;
    return null;
  },
});

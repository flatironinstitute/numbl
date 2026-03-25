/**
 * Unary math builtins: trig, hyperbolic, exp/log, rounding, abs, sqrt, sign.
 */

import { FloatXArray, isRuntimeTensor } from "../../runtime/types.js";
import {
  defineBuiltin,
  unaryElemwiseCases,
  unaryRealResultCases,
  applyUnaryElemwise,
  applyBinaryScalar,
  binaryNumberOnly,
  makeTensor,
  unaryMathJitEmit,
  binaryMathJitEmit,
} from "./types.js";
import { type JitType, isNonneg, type SignCategory } from "../jit/jitTypes.js";
import {
  erfScalar,
  erfcScalar,
  erfinvScalar,
  erfcinvScalar,
  erfcxScalar,
} from "../../builtins/erf.js";

// ── Simple unary registration helper ────────────────────────────────────

function registerUnary(
  name: string,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null
): void {
  defineBuiltin({
    name,
    cases: unaryElemwiseCases({ realFn, complexFn }, name),
    jitEmit,
  });
}

// ── Trig ────────────────────────────────────────────────────────────────

registerUnary(
  "sin",
  Math.sin,
  (re, im) => ({
    re: Math.sin(re) * Math.cosh(im),
    im: Math.cos(re) * Math.sinh(im),
  }),
  unaryMathJitEmit("Math.sin", "tSin")
);

registerUnary(
  "cos",
  Math.cos,
  (re, im) => ({
    re: Math.cos(re) * Math.cosh(im),
    im: -Math.sin(re) * Math.sinh(im),
  }),
  unaryMathJitEmit("Math.cos", "tCos")
);

registerUnary(
  "tan",
  Math.tan,
  (re, im) => {
    const denom = Math.cos(2 * re) + Math.cosh(2 * im);
    return { re: Math.sin(2 * re) / denom, im: Math.sinh(2 * im) / denom };
  },
  unaryMathJitEmit("Math.tan", "tTan")
);

// Inverse trig (complex formulas from builtins/math.ts)

function cAsin(re: number, im: number): { re: number; im: number } {
  const w1re = -im,
    w1im = re;
  const w2re = 1 - re * re + im * im,
    w2im = -2 * re * im;
  const w2r = Math.sqrt(w2re * w2re + w2im * w2im);
  const w3re = Math.sqrt((w2r + w2re) / 2);
  const w3im =
    w2im >= 0 ? Math.sqrt((w2r - w2re) / 2) : -Math.sqrt((w2r - w2re) / 2);
  const w4re = w1re + w3re,
    w4im = w1im + w3im;
  const w5re = Math.log(Math.sqrt(w4re * w4re + w4im * w4im));
  const w5im = Math.atan2(w4im, w4re);
  return { re: w5im, im: -w5re };
}

function cAcos(re: number, im: number): { re: number; im: number } {
  const w2re = 1 - re * re + im * im,
    w2im = -2 * re * im;
  const w2r = Math.sqrt(w2re * w2re + w2im * w2im);
  const w3re = Math.sqrt((w2r + w2re) / 2);
  const w3im =
    w2im >= 0 ? Math.sqrt((w2r - w2re) / 2) : -Math.sqrt((w2r - w2re) / 2);
  const w4re = re - w3im,
    w4im = im + w3re;
  const w5re = Math.log(Math.sqrt(w4re * w4re + w4im * w4im));
  const w5im = Math.atan2(w4im, w4re);
  return { re: w5im, im: -w5re };
}

// asin/acos: use maybe-complex variant so out-of-domain real inputs produce complex
defineBuiltin({
  name: "asin",
  cases: unaryElemwiseCases(
    { realFn: Math.asin, complexFn: cAsin, maybeComplex: true },
    "asin"
  ),
  jitEmit: unaryMathJitEmit("Math.asin", "tAsin"),
});
defineBuiltin({
  name: "acos",
  cases: unaryElemwiseCases(
    { realFn: Math.acos, complexFn: cAcos, maybeComplex: true },
    "acos"
  ),
  jitEmit: unaryMathJitEmit("Math.acos", "tAcos"),
});
registerUnary(
  "atan",
  Math.atan,
  (re, im) => {
    const w1re = 1 + im,
      w1im = -re;
    const w2re = 1 - im,
      w2im = re;
    const denom = w2re * w2re + w2im * w2im;
    const w3re = (w1re * w2re + w1im * w2im) / denom;
    const w3im = (w1im * w2re - w1re * w2im) / denom;
    const w4re = Math.log(Math.sqrt(w3re * w3re + w3im * w3im));
    const w4im = Math.atan2(w3im, w3re);
    return { re: -w4im / 2, im: w4re / 2 };
  },
  unaryMathJitEmit("Math.atan", "tAtan")
);

// ── Hyperbolic ──────────────────────────────────────────────────────────

registerUnary(
  "sinh",
  Math.sinh,
  (re, im) => ({
    re: Math.sinh(re) * Math.cos(im),
    im: Math.cosh(re) * Math.sin(im),
  }),
  unaryMathJitEmit("Math.sinh", "tSinh")
);

registerUnary(
  "cosh",
  Math.cosh,
  (re, im) => ({
    re: Math.cosh(re) * Math.cos(im),
    im: Math.sinh(re) * Math.sin(im),
  }),
  unaryMathJitEmit("Math.cosh", "tCosh")
);

registerUnary(
  "tanh",
  Math.tanh,
  (re, im) => {
    const denom = Math.cosh(2 * re) + Math.cos(2 * im);
    return { re: Math.sinh(2 * re) / denom, im: Math.sin(2 * im) / denom };
  },
  unaryMathJitEmit("Math.tanh", "tTanh")
);

// ── Exp / Log ───────────────────────────────────────────────────────────

const complexExp = (re: number, im: number) => ({
  re: Math.exp(re) * Math.cos(im),
  im: Math.exp(re) * Math.sin(im),
});

defineBuiltin({
  name: "exp",
  cases: unaryElemwiseCases(
    {
      numberType: () => ({ kind: "number", sign: "positive" }),
      booleanType: () => ({ kind: "number", sign: "positive" }),
      tensorType: t => ({
        kind: "tensor",
        isComplex: t.isComplex,
        shape: t.shape,
        ndim: t.ndim,
        ...(t.isComplex ? {} : { nonneg: true }),
      }),
      realFn: Math.exp,
      complexFn: complexExp,
    },
    "exp"
  ),
  jitEmit: unaryMathJitEmit("Math.exp", "tExp"),
});

function complexLog(re: number, im: number): { re: number; im: number } {
  return {
    re: Math.log(Math.sqrt(re * re + im * im)),
    im: Math.atan2(im, re),
  };
}

defineBuiltin({
  name: "log",
  cases: unaryElemwiseCases(
    { realFn: Math.log, complexFn: complexLog, maybeComplex: true },
    "log"
  ),
  jitEmit: unaryMathJitEmit("Math.log", "tLog", true),
});

const complexLog2 = (re: number, im: number) => ({
  re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN2,
  im: Math.atan2(im, re) / Math.LN2,
});

// log2: has special two-output frexp form, so uses defineBuiltin with custom cases
defineBuiltin({
  name: "log2",
  cases: [
    // [f, e] = log2(x) — frexp form (nargout > 1)
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout <= 1) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "boolean")
          return [{ kind: "number" }, { kind: "number" }];
        if (a.kind === "tensor" && a.isComplex !== true)
          return [
            { kind: "tensor", isComplex: false, shape: a.shape, ndim: a.ndim },
            { kind: "tensor", isComplex: false, shape: a.shape, ndim: a.ndim },
          ];
        return null;
      },
      apply: args => {
        function frexpScalar(x: number): { f: number; e: number } {
          if (x === 0) return { f: 0, e: 0 };
          if (!isFinite(x)) return { f: x, e: 0 };
          const e = Math.floor(Math.log2(Math.abs(x))) + 1;
          return { f: x / Math.pow(2, e), e };
        }
        const v = args[0];
        if (typeof v === "boolean" || typeof v === "number") {
          const { f, e } = frexpScalar(typeof v === "boolean" ? +v : v);
          return [f, e];
        }
        if (isRuntimeTensor(v) && !v.imag) {
          const n = v.data.length;
          const fData = new FloatXArray(n);
          const eData = new FloatXArray(n);
          for (let i = 0; i < n; i++) {
            const { f, e } = frexpScalar(v.data[i]);
            fData[i] = f;
            eData[i] = e;
          }
          return [
            makeTensor(fData, undefined, v.shape.slice()),
            makeTensor(eData, undefined, v.shape.slice()),
          ];
        }
        throw new Error("log2: frexp form only supports real inputs");
      },
    },
    // Single-output log2 (standard element-wise)
    ...unaryElemwiseCases(
      { realFn: Math.log2, complexFn: complexLog2, maybeComplex: true },
      "log2"
    ),
  ],
  jitEmit: unaryMathJitEmit("Math.log2", "tLog2", true),
});

const complexLog10 = (re: number, im: number) => ({
  re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN10,
  im: Math.atan2(im, re) / Math.LN10,
});

defineBuiltin({
  name: "log10",
  cases: unaryElemwiseCases(
    { realFn: Math.log10, complexFn: complexLog10, maybeComplex: true },
    "log10"
  ),
  jitEmit: unaryMathJitEmit("Math.log10", "tLog10", true),
});

// ── Abs ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "abs",
  cases: unaryRealResultCases(
    Math.abs,
    (re, im) => Math.sqrt(re * re + im * im),
    "abs"
  ),
  jitEmit: unaryMathJitEmit("Math.abs", "tAbs"),
});

// ── Sqrt ────────────────────────────────────────────────────────────────

function complexSqrt(re: number, im: number): { re: number; im: number } {
  const mag = Math.sqrt(re * re + im * im);
  if (mag === 0) return { re: 0, im: 0 };
  const r = Math.sqrt(mag);
  const angle = Math.atan2(im, re) / 2;
  return { re: r * Math.cos(angle), im: r * Math.sin(angle) };
}

defineBuiltin({
  name: "sqrt",
  cases: unaryElemwiseCases(
    {
      numberType: a => {
        if (isNonneg(a)) {
          const outSign: SignCategory | undefined =
            a.sign === "positive" ? "positive" : "nonneg";
          return { kind: "number", sign: outSign };
        }
        return { kind: "complex_or_number" };
      },
      booleanType: () => ({ kind: "number", sign: "nonneg" }),
      tensorType: t => {
        if (t.isComplex === true)
          return {
            kind: "tensor",
            isComplex: true,
            shape: t.shape,
            ndim: t.ndim,
          };
        if (t.nonneg)
          return {
            kind: "tensor",
            isComplex: false,
            shape: t.shape,
            ndim: t.ndim,
            nonneg: true,
          };
        return {
          kind: "tensor",
          isComplex: true,
          shape: t.shape,
          ndim: t.ndim,
        };
      },
      realFn: x => (x >= 0 ? Math.sqrt(x) : NaN),
      complexFn: complexSqrt,
      maybeComplex: true,
    },
    "sqrt"
  ),
  jitEmit: unaryMathJitEmit("Math.sqrt", "tSqrt", true),
});

// ── Sign ────────────────────────────────────────────────────────────────

registerUnary(
  "sign",
  Math.sign,
  (re, im) => {
    const mag = Math.sqrt(re * re + im * im);
    if (mag === 0) return { re: 0, im: 0 };
    return { re: re / mag, im: im / mag };
  },
  unaryMathJitEmit("Math.sign", "tSign")
);

// ── Rounding ────────────────────────────────────────────────────────────

function registerRounding(
  name: string,
  fn: (x: number) => number,
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null
): void {
  defineBuiltin({
    name,
    cases: unaryElemwiseCases(
      {
        numberType: a => ({
          kind: "number",
          ...(a.sign ? { sign: a.sign } : {}),
        }),
        booleanType: () => ({ kind: "number", sign: "nonneg" }),
        tensorType: t => ({
          kind: "tensor",
          isComplex: t.isComplex,
          shape: t.shape,
          ndim: t.ndim,
          ...(t.nonneg ? { nonneg: true } : {}),
        }),
        realFn: fn,
        complexFn: (re, im) => ({ re: fn(re), im: fn(im) }),
      },
      name
    ),
    jitEmit,
  });
}

registerRounding("floor", Math.floor, unaryMathJitEmit("Math.floor", "tFloor"));
registerRounding("ceil", Math.ceil, unaryMathJitEmit("Math.ceil", "tCeil"));
registerRounding("fix", Math.trunc, unaryMathJitEmit("Math.trunc", "tFix"));

// MATLAB round: half away from zero (not JS half-toward-+inf)
function matlabRound(x: number): number {
  return Math.sign(x) * Math.round(Math.abs(x));
}

defineBuiltin({
  name: "round",
  cases: [
    // Two-arg form: round(x, n) — must come before the single-arg cases
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0];
        const b = argTypes[1];
        if (b.kind !== "number" && b.kind !== "boolean") return null;
        switch (a.kind) {
          case "number":
            return [{ kind: "number", ...(a.sign ? { sign: a.sign } : {}) }];
          case "boolean":
            return [{ kind: "number", sign: "nonneg" }];
          case "complex_or_number":
            return [{ kind: "complex_or_number" }];
          case "tensor":
            return [
              {
                kind: "tensor",
                isComplex: a.isComplex,
                shape: a.shape,
                ndim: a.ndim,
                ...(a.nonneg ? { nonneg: true } : {}),
              },
            ];
          default:
            return null;
        }
      },
      apply: args => {
        const nArg = args[1];
        const n = typeof nArg === "number" ? nArg : 0;
        const scale = Math.pow(10, n);
        function roundN(x: number): number {
          return matlabRound(x * scale) / scale;
        }
        return applyUnaryElemwise(
          args[0],
          roundN,
          (re, im) => ({ re: roundN(re), im: roundN(im) }),
          "round"
        );
      },
    },
    // Single-arg form
    ...unaryElemwiseCases(
      {
        numberType: a => ({
          kind: "number",
          ...(a.sign ? { sign: a.sign } : {}),
        }),
        booleanType: () => ({ kind: "number", sign: "nonneg" }),
        tensorType: t => ({
          kind: "tensor",
          isComplex: t.isComplex,
          shape: t.shape,
          ndim: t.ndim,
          ...(t.nonneg ? { nonneg: true } : {}),
        }),
        realFn: matlabRound,
        complexFn: (re, im) => ({ re: matlabRound(re), im: matlabRound(im) }),
      },
      "round"
    ),
  ],
});

// ── Precision math: expm1, log1p ─────────────────────────────────────────

registerUnary(
  "expm1",
  Math.expm1,
  (re, im) => {
    // expm1(z) = exp(z) - 1
    const r = complexExp(re, im);
    return { re: r.re - 1, im: r.im };
  },
  unaryMathJitEmit("Math.expm1", "tExpm1")
);

function complexLog1p(re: number, im: number): { re: number; im: number } {
  // log1p(z) = log(1 + z)
  return complexLog(1 + re, im);
}

defineBuiltin({
  name: "log1p",
  cases: unaryElemwiseCases(
    { realFn: Math.log1p, complexFn: complexLog1p, maybeComplex: true },
    "log1p"
  ),
  jitEmit: unaryMathJitEmit("Math.log1p", "tLog1p"),
});

// ── Hypot ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "hypot",
  cases: [
    {
      match: argTypes => binaryNumberOnly(argTypes),
      apply: args => applyBinaryScalar(args, Math.hypot, "hypot"),
    },
  ],
  jitEmit: binaryMathJitEmit("Math.hypot"),
});

// ── Error functions ──────────────────────────────────────────────────────

const noComplexFn = () => ({ re: NaN, im: NaN });

registerUnary("erf", erfScalar, noComplexFn);
registerUnary("erfc", erfcScalar, noComplexFn);
registerUnary("erfinv", erfinvScalar, noComplexFn);
registerUnary("erfcinv", erfcinvScalar, noComplexFn);
registerUnary("erfcx", erfcxScalar, noComplexFn);

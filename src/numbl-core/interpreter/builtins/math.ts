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
  applyBinaryElemwise,
  binaryNumberOnly,
  binaryElemwiseMatch,
  makeTensor,
  unaryMathJitEmit,
  unaryMathJitEmitC,
  binaryMathJitEmit,
  binaryMathJitEmitC,
  type BuiltinHelp,
} from "./types.js";
import {
  type JitType,
  isNonneg,
  type SignCategory,
} from "../../jit/jitTypes.js";
import {
  erfScalar,
  erfcScalar,
  erfinvScalar,
  erfcinvScalar,
  erfcxScalar,
} from "../../helpers/erf.js";
import { lanczosGamma } from "../../helpers/bessel.js";

// ── Simple unary registration helper ────────────────────────────────────

function registerUnary(
  name: string,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null,
  help?: BuiltinHelp,
  jitEmitC?: (argCode: string[], argTypes: JitType[]) => string | null
): void {
  defineBuiltin({
    name,
    help,
    cases: unaryElemwiseCases({ realFn, complexFn }, name),
    jitEmit,
    jitEmitC,
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
  unaryMathJitEmit("Math.sin", "tSin"),
  undefined,
  unaryMathJitEmitC("sin")
);

registerUnary(
  "cos",
  Math.cos,
  (re, im) => ({
    re: Math.cos(re) * Math.cosh(im),
    im: -Math.sin(re) * Math.sinh(im),
  }),
  unaryMathJitEmit("Math.cos", "tCos"),
  undefined,
  unaryMathJitEmitC("cos")
);

registerUnary(
  "tan",
  Math.tan,
  (re, im) => {
    const denom = Math.cos(2 * re) + Math.cosh(2 * im);
    return { re: Math.sin(2 * re) / denom, im: Math.sinh(2 * im) / denom };
  },
  unaryMathJitEmit("Math.tan", "tTan"),
  undefined,
  unaryMathJitEmitC("tan")
);

// Inverse trig (complex formulas from builtins/math.ts)

// Kahan's formulation of complex asin/acos via A, B, and 2x/(a1+a2) to
// avoid overflow in re^2 / im^2 and cancellation in a1-a2.  Preserves
// MATLAB's branch-cut convention for real x with |x|>1 and y==0.
function cAsinAcosCore(
  re: number,
  im: number
): { beta: number; alpha: number; signY: number } {
  const a1 = Math.hypot(re + 1, im);
  const a2 = Math.hypot(re - 1, im);
  const sum = a1 + a2;
  // β = (a1 - a2)/2 rewritten as 2*re/(a1+a2) to avoid cancellation.
  const betaRaw = sum === 0 ? 0 : (2 * re) / sum;
  const beta = betaRaw > 1 ? 1 : betaRaw < -1 ? -1 : betaRaw;
  const alpha = sum / 2;
  // MATLAB convention: for y == 0 and |re| > 1 the branch sign is -sign(re).
  let signY: number;
  if (im > 0) signY = 1;
  else if (im < 0) signY = -1;
  else signY = re > 1 ? -1 : 1;
  return { beta, alpha, signY };
}

function acoshSafe(alpha: number): number {
  if (alpha <= 1) return 0;
  // For alpha large enough that alpha*alpha overflows, use log(2*alpha).
  if (alpha > 1e150) return Math.LN2 + Math.log(alpha);
  return Math.acosh(alpha);
}

function cAsin(re: number, im: number): { re: number; im: number } {
  const { beta, alpha, signY } = cAsinAcosCore(re, im);
  return { re: Math.asin(beta), im: signY * acoshSafe(alpha) };
}

function cAcos(re: number, im: number): { re: number; im: number } {
  const { beta, alpha, signY } = cAsinAcosCore(re, im);
  return { re: Math.acos(beta), im: -signY * acoshSafe(alpha) };
}

// asin/acos: maybe-complex — out-of-domain real inputs (|x| > 1) produce
// complex results. No jitEmit fast path because the JIT type system doesn't
// track whether a value is within [-1, 1]; emitting `Math.asin` / `Math.acos`
// directly would silently produce NaN where the interpreter produces complex.
defineBuiltin({
  name: "asin",
  cases: unaryElemwiseCases(
    {
      realFn: Math.asin,
      complexFn: cAsin,
      maybeComplex: true,
      nativeOpCode: 13,
    },
    "asin"
  ),
});
defineBuiltin({
  name: "acos",
  cases: unaryElemwiseCases(
    {
      realFn: Math.acos,
      complexFn: cAcos,
      maybeComplex: true,
      nativeOpCode: 14,
    },
    "acos"
  ),
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
  unaryMathJitEmit("Math.atan", "tAtan"),
  undefined,
  unaryMathJitEmitC("atan")
);

// ── Hyperbolic ──────────────────────────────────────────────────────────

registerUnary(
  "sinh",
  Math.sinh,
  (re, im) => ({
    re: Math.sinh(re) * Math.cos(im),
    im: Math.cosh(re) * Math.sin(im),
  }),
  unaryMathJitEmit("Math.sinh", "tSinh"),
  undefined,
  unaryMathJitEmitC("sinh")
);

registerUnary(
  "cosh",
  Math.cosh,
  (re, im) => ({
    re: Math.cosh(re) * Math.cos(im),
    im: Math.sinh(re) * Math.sin(im),
  }),
  unaryMathJitEmit("Math.cosh", "tCosh"),
  undefined,
  unaryMathJitEmitC("cosh")
);

registerUnary(
  "tanh",
  Math.tanh,
  (re, im) => {
    const denom = Math.cosh(2 * re) + Math.cos(2 * im);
    return { re: Math.sinh(2 * re) / denom, im: Math.sin(2 * im) / denom };
  },
  unaryMathJitEmit("Math.tanh", "tTanh"),
  undefined,
  unaryMathJitEmitC("tanh")
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
  jitEmitC: unaryMathJitEmitC("exp"),
});

function complexLog(re: number, im: number): { re: number; im: number } {
  return {
    re: Math.log(Math.hypot(re, im)),
    im: Math.atan2(im, re),
  };
}

defineBuiltin({
  name: "log",
  cases: unaryElemwiseCases(
    {
      realFn: Math.log,
      complexFn: complexLog,
      maybeComplex: true,
      nativeOpCode: 1,
    },
    "log"
  ),
  jitEmit: unaryMathJitEmit("Math.log", "tLog", true),
  jitEmitC: unaryMathJitEmitC("log", true),
});

const complexLog2 = (re: number, im: number) => ({
  re: Math.log(Math.hypot(re, im)) / Math.LN2,
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
      {
        realFn: Math.log2,
        complexFn: complexLog2,
        maybeComplex: true,
        nativeOpCode: 2,
      },
      "log2"
    ),
  ],
  jitEmit: unaryMathJitEmit("Math.log2", "tLog2", true),
  jitEmitC: unaryMathJitEmitC("log2", true),
});

const complexLog10 = (re: number, im: number) => ({
  re: Math.log(Math.hypot(re, im)) / Math.LN10,
  im: Math.atan2(im, re) / Math.LN10,
});

defineBuiltin({
  name: "log10",
  cases: unaryElemwiseCases(
    {
      realFn: Math.log10,
      complexFn: complexLog10,
      maybeComplex: true,
      nativeOpCode: 3,
    },
    "log10"
  ),
  jitEmit: unaryMathJitEmit("Math.log10", "tLog10", true),
  jitEmitC: unaryMathJitEmitC("log10", true),
});

// ── Abs ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "abs",
  cases: unaryRealResultCases(Math.abs, Math.hypot, "abs"),
  jitEmit: unaryMathJitEmit("Math.abs", "tAbs"),
  jitEmitC: unaryMathJitEmitC("fabs"),
});

// ── Sqrt ────────────────────────────────────────────────────────────────

function complexSqrt(re: number, im: number): { re: number; im: number } {
  // Pure-real input: compute directly to avoid polar-form roundoff
  // (Math.sin(pi/2) isn't exactly 1, Math.cos(pi/2) isn't exactly 0).
  if (im === 0) {
    if (re >= 0) return { re: Math.sqrt(re), im: 0 };
    return { re: 0, im: Math.sqrt(-re) };
  }
  // hypot keeps |z| finite when re*re or im*im would overflow.
  const mag = Math.hypot(re, im);
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
      nativeOpCode: 4,
    },
    "sqrt"
  ),
  jitEmit: unaryMathJitEmit("Math.sqrt", "tSqrt", true),
  jitEmitC: unaryMathJitEmitC("sqrt", true),
});

// ── Sign ────────────────────────────────────────────────────────────────

registerUnary(
  "sign",
  Math.sign,
  (re, im) => {
    // Use hypot so magnitude stays finite when re*re or im*im overflow.
    const mag = Math.hypot(re, im);
    if (mag === 0) return { re: 0, im: 0 };
    return { re: re / mag, im: im / mag };
  },
  unaryMathJitEmit("Math.sign", "tSign"),
  undefined,
  unaryMathJitEmitC("numbl_sign")
);

// ── Rounding ────────────────────────────────────────────────────────────

function registerRounding(
  name: string,
  fn: (x: number) => number,
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null,
  jitEmitC?: (argCode: string[], argTypes: JitType[]) => string | null
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
    jitEmitC,
  });
}

registerRounding(
  "floor",
  Math.floor,
  unaryMathJitEmit("Math.floor", "tFloor"),
  unaryMathJitEmitC("floor")
);
registerRounding(
  "ceil",
  Math.ceil,
  unaryMathJitEmit("Math.ceil", "tCeil"),
  unaryMathJitEmitC("ceil")
);
registerRounding(
  "fix",
  Math.trunc,
  unaryMathJitEmit("Math.trunc", "tFix"),
  unaryMathJitEmitC("trunc")
);

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
  // C round(x) is half-away-from-zero per C99 — matches MATLAB.
  jitEmitC: unaryMathJitEmitC("round"),
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
  unaryMathJitEmit("Math.expm1", "tExpm1"),
  undefined,
  unaryMathJitEmitC("expm1")
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
  jitEmitC: unaryMathJitEmitC("log1p"),
});

// ── Hypot ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "hypot",
  cases: [
    {
      match: argTypes => binaryElemwiseMatch(argTypes),
      apply: args => applyBinaryElemwise(args, Math.hypot, "hypot"),
    },
  ],
  jitEmit: binaryMathJitEmit("Math.hypot"),
  jitEmitC: binaryMathJitEmitC("hypot"),
});

// ── Error functions ──────────────────────────────────────────────────────

const noComplexFn = () => ({ re: NaN, im: NaN });

registerUnary("erf", erfScalar, noComplexFn);
registerUnary("erfc", erfcScalar, noComplexFn);
registerUnary("erfinv", erfinvScalar, noComplexFn);
registerUnary("erfcinv", erfcinvScalar, noComplexFn);
registerUnary("erfcx", erfcxScalar, noComplexFn);

// ── Complex helpers for inverse hyperbolic / reciprocal ──────────────

function cRecip(re: number, im: number): { re: number; im: number } {
  // Smith's algorithm avoids spurious over/underflow in re*re + im*im.
  if (re === 0 && im === 0) return { re: 1 / 0, im: -1 / 0 };
  if (Math.abs(re) >= Math.abs(im)) {
    const r = im / re;
    const d = re + im * r;
    return { re: 1 / d, im: -r / d };
  }
  const r = re / im;
  const d = im + re * r;
  return { re: r / d, im: -1 / d };
}

function composeRecip(
  complexFn: (re: number, im: number) => { re: number; im: number }
): (re: number, im: number) => { re: number; im: number } {
  return (re, im) => {
    const r = cRecip(re, im);
    return complexFn(r.re, r.im);
  };
}

function cAcosh(re: number, im: number): { re: number; im: number } {
  const a1re = re - 1,
    a1im = im;
  const a1r = Math.sqrt(a1re * a1re + a1im * a1im);
  const s1re = Math.sqrt((a1r + a1re) / 2);
  const s1im =
    a1im >= 0 ? Math.sqrt((a1r - a1re) / 2) : -Math.sqrt((a1r - a1re) / 2);
  const a2re = re + 1,
    a2im = im;
  const a2r = Math.sqrt(a2re * a2re + a2im * a2im);
  const s2re = Math.sqrt((a2r + a2re) / 2);
  const s2im =
    a2im >= 0 ? Math.sqrt((a2r - a2re) / 2) : -Math.sqrt((a2r - a2re) / 2);
  const w2re = s1re * s2re - s1im * s2im;
  const w2im = s1re * s2im + s1im * s2re;
  const w3re = re + w2re,
    w3im = im + w2im;
  return {
    re: Math.log(Math.sqrt(w3re * w3re + w3im * w3im)),
    im: Math.atan2(w3im, w3re),
  };
}

function cAsinh(re: number, im: number): { re: number; im: number } {
  const z2re = re * re - im * im,
    z2im = 2 * re * im;
  const w1re = z2re + 1,
    w1im = z2im;
  const w1r = Math.sqrt(w1re * w1re + w1im * w1im);
  const w2re = Math.sqrt((w1r + w1re) / 2);
  const w2im =
    w1im >= 0 ? Math.sqrt((w1r - w1re) / 2) : -Math.sqrt((w1r - w1re) / 2);
  const w3re = re + w2re,
    w3im = im + w2im;
  return {
    re: Math.log(Math.sqrt(w3re * w3re + w3im * w3im)),
    im: Math.atan2(w3im, w3re),
  };
}

function cAtanh(re: number, im: number): { re: number; im: number } {
  if (im === 0 && (re > 1 || re < -1)) {
    return {
      re: 0.5 * Math.log(Math.abs((1 + re) / (1 - re))),
      im: re > 0 ? Math.PI / 2 : -Math.PI / 2,
    };
  }
  const w1re = 1 + re,
    w1im = im;
  const w2re = 1 - re,
    w2im = -im;
  const denom = w2re * w2re + w2im * w2im;
  const w3re = (w1re * w2re + w1im * w2im) / denom;
  const w3im = (w1im * w2re - w1re * w2im) / denom;
  const w4re = Math.log(Math.sqrt(w3re * w3re + w3im * w3im));
  const w4im = Math.atan2(w3im, w3re);
  return { re: w4re / 2, im: w4im / 2 };
}

// ── Degree trig constants ────────────────────────────────────────────

const deg2rad = Math.PI / 180;
const rad2deg = 180 / Math.PI;

// ── Reciprocal trig ──────────────────────────────────────────────────

registerUnary("sec", (x: number) => 1 / Math.cos(x), noComplexFn);
registerUnary("csc", (x: number) => 1 / Math.sin(x), noComplexFn);
registerUnary("cot", (x: number) => 1 / Math.tan(x), noComplexFn);

// ── Degree trig ──────────────────────────────────────────────────────

registerUnary("sind", (x: number) => Math.sin(x * deg2rad), noComplexFn);
registerUnary("cosd", (x: number) => Math.cos(x * deg2rad), noComplexFn);
registerUnary("tand", (x: number) => Math.tan(x * deg2rad), noComplexFn);
registerUnary("secd", (x: number) => 1 / Math.cos(x * deg2rad), noComplexFn);
registerUnary("cscd", (x: number) => 1 / Math.sin(x * deg2rad), noComplexFn);
registerUnary(
  "cotd",
  (x: number) => Math.cos(x * deg2rad) / Math.sin(x * deg2rad),
  noComplexFn
);

// ── Inverse trig (degree output) ────────────────────────────────────

registerUnary("atand", (x: number) => Math.atan(x) * rad2deg, noComplexFn);
registerUnary("acotd", (x: number) => Math.atan(1 / x) * rad2deg, noComplexFn);

// asind/acosd: may produce complex for |x| > 1
function toDegComplex(
  fn: (re: number, im: number) => { re: number; im: number }
): (re: number, im: number) => { re: number; im: number } {
  return (re, im) => {
    const r = fn(re, im);
    return { re: r.re * rad2deg, im: r.im * rad2deg };
  };
}

defineBuiltin({
  name: "asind",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.asin(x) * rad2deg,
      complexFn: toDegComplex(cAsin),
      maybeComplex: true,
    },
    "asind"
  ),
});
defineBuiltin({
  name: "acosd",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.acos(x) * rad2deg,
      complexFn: toDegComplex(cAcos),
      maybeComplex: true,
    },
    "acosd"
  ),
});

// ── Inverse reciprocal trig ─────────────────────────────────────────

defineBuiltin({
  name: "asec",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.acos(1 / x),
      complexFn: composeRecip(cAcos),
      maybeComplex: true,
    },
    "asec"
  ),
});
defineBuiltin({
  name: "acsc",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.asin(1 / x),
      complexFn: composeRecip(cAsin),
      maybeComplex: true,
    },
    "acsc"
  ),
});
registerUnary("acot", (x: number) => Math.atan(1 / x), noComplexFn);

// ── Inverse reciprocal trig (degree output) ─────────────────────────

defineBuiltin({
  name: "asecd",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.acos(1 / x) * rad2deg,
      complexFn: toDegComplex(composeRecip(cAcos)),
      maybeComplex: true,
    },
    "asecd"
  ),
});
defineBuiltin({
  name: "acscd",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.asin(1 / x) * rad2deg,
      complexFn: toDegComplex(composeRecip(cAsin)),
      maybeComplex: true,
    },
    "acscd"
  ),
});

// ── atan2d (binary) ─────────────────────────────────────────────────

defineBuiltin({
  name: "atan2d",
  cases: [
    {
      match: argTypes => binaryNumberOnly(argTypes),
      apply: args =>
        applyBinaryScalar(args, (y, x) => Math.atan2(y, x) * rad2deg, "atan2d"),
    },
  ],
  jitEmit: binaryMathJitEmit("((y,x)=>Math.atan2(y,x)*180/Math.PI)"),
});

// ── Inverse hyperbolic ──────────────────────────────────────────────

defineBuiltin({
  name: "asinh",
  cases: unaryElemwiseCases(
    { realFn: Math.asinh, complexFn: cAsinh, maybeComplex: true },
    "asinh"
  ),
  // asinh has no real-domain restriction, so C can emit it unguarded.
  // (JS-JIT doesn't provide a scalar fast-path — it falls back to $h.ib_asinh.)
  jitEmitC: unaryMathJitEmitC("asinh"),
});
defineBuiltin({
  name: "acosh",
  cases: unaryElemwiseCases(
    { realFn: Math.acosh, complexFn: cAcosh, maybeComplex: true },
    "acosh"
  ),
});
defineBuiltin({
  name: "atanh",
  cases: unaryElemwiseCases(
    { realFn: Math.atanh, complexFn: cAtanh, maybeComplex: true },
    "atanh"
  ),
});

// ── Reciprocal hyperbolic ───────────────────────────────────────────

registerUnary("sech", (x: number) => 1 / Math.cosh(x), noComplexFn);
registerUnary("csch", (x: number) => 1 / Math.sinh(x), noComplexFn);
registerUnary("coth", (x: number) => 1 / Math.tanh(x), noComplexFn);

// ── Inverse reciprocal hyperbolic ───────────────────────────────────

defineBuiltin({
  name: "asech",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.acosh(1 / x),
      complexFn: composeRecip(cAcosh),
      maybeComplex: true,
    },
    "asech"
  ),
});
defineBuiltin({
  name: "acsch",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.asinh(1 / x),
      complexFn: composeRecip(cAsinh),
      maybeComplex: true,
    },
    "acsch"
  ),
});
defineBuiltin({
  name: "acoth",
  cases: unaryElemwiseCases(
    {
      realFn: (x: number) => Math.atanh(1 / x),
      complexFn: composeRecip(cAtanh),
      maybeComplex: true,
    },
    "acoth"
  ),
});

// ── pow2, nextpow2 ──────────────────────────────────────────────────

registerUnary("pow2", (x: number) => Math.pow(2, x), noComplexFn);
registerUnary(
  "nextpow2",
  (x: number) => (x <= 0 ? 0 : Math.ceil(Math.log2(x))),
  noComplexFn
);

// ── nthroot (binary) ────────────────────────────────────────────────

defineBuiltin({
  name: "nthroot",
  cases: [
    {
      match: argTypes => binaryElemwiseMatch(argTypes),
      apply: args =>
        applyBinaryElemwise(
          args,
          (x, n) => {
            let result: number;
            if (x < 0 && n % 2 !== 0) {
              result = -Math.pow(-x, 1 / n);
            } else {
              result = Math.pow(x, 1 / n);
            }
            const rounded = Math.round(result);
            if (
              Math.abs(result - rounded) <
              1e-10 * Math.max(1, Math.abs(rounded))
            ) {
              if (Math.pow(rounded, n) === x) return rounded;
            }
            return result;
          },
          "nthroot"
        ),
    },
  ],
});

// ── gamma, factorial, beta ──────────────────────────────────────────

registerUnary(
  "gamma",
  (x: number) => {
    if (x === 0) return Infinity;
    if (x < 0 && Math.floor(x) === x) return NaN;
    if (x === Infinity) return Infinity;
    if (x === -Infinity || Number.isNaN(x)) return NaN;
    return lanczosGamma(x);
  },
  noComplexFn
);

registerUnary(
  "gammaln",
  (x: number) => {
    if (x < 0) throw new Error("Input must be nonnegative.");
    if (Number.isNaN(x)) return NaN;
    if (x === 0 || x === Infinity) return Infinity;
    // Lanczos approximation in log-space
    const g = 7;
    const coef = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    const xx = x - 1;
    let a = coef[0];
    for (let i = 1; i < coef.length; i++) {
      a += coef[i] / (xx + i);
    }
    const t = xx + g + 0.5;
    return (
      0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(a)
    );
  },
  noComplexFn
);

registerUnary(
  "factorial",
  (n: number) => {
    if (n < 0 || !Number.isFinite(n)) return NaN;
    if (Number.isInteger(n) && n <= 20) {
      let result = 1;
      for (let i = 2; i <= n; i++) result *= i;
      return result;
    }
    return lanczosGamma(n + 1);
  },
  noComplexFn
);

defineBuiltin({
  name: "beta",
  cases: [
    {
      match: argTypes => binaryElemwiseMatch(argTypes),
      apply: args =>
        applyBinaryElemwise(
          args,
          (x, y) => {
            if (
              (x === 0 || y === 0) &&
              !(Number.isInteger(x) && x < 0) &&
              !(Number.isInteger(y) && y < 0)
            ) {
              return Infinity;
            }
            const gx = lanczosGamma(x);
            const gy = lanczosGamma(y);
            const gxy = lanczosGamma(x + y);
            if (!isFinite(gx) || !isFinite(gy)) {
              if (!isFinite(gx) && !isFinite(gy)) return Infinity;
              if (!isFinite(gxy)) return NaN;
              return Infinity;
            }
            if (gxy === 0) return Infinity;
            return (gx / gxy) * gy;
          },
          "beta"
        ),
    },
  ],
});

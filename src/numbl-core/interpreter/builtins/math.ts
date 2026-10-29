/**
 * Unary math builtins: trig, hyperbolic, exp/log, rounding, abs, sqrt, sign,
 * plus the binary hypot, atan2d, nthroot, beta and pow2.
 *
 * Like MATLAB, the floating-point functions (sin, exp, gamma, ...) reject
 * char and non-numeric input with "Argument must be double or single";
 * numbl also accepts logical input for them (MATLAB does not). abs, sign
 * and the rounding functions accept char like MATLAB.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RuntimeError } from "../../runtime/error.js";
import { builtin } from "./builtin.js";
import { builtinError, mustBeRealError, type NumericArg } from "./args.js";
import {
  floatArg,
  hasImag,
  mustBeFloatError,
  notFullRealError,
  realDenseArg,
  realFloatArg,
} from "./math-args.js";
import {
  applyBinaryElemwise,
  applyUnaryElemwise,
  makeTensor,
  unaryElemwise,
  unaryRealResult,
  type UnaryElemwiseFns,
} from "./types.js";
import type { BuiltinHelp } from "../../runtime/builtinRegistry.js";
import {
  erfScalar,
  erfcScalar,
  erfinvScalar,
  erfcinvScalar,
  erfcxScalar,
} from "../../helpers/erf.js";
import { lanczosGamma } from "../../helpers/bessel.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { OpBin, OpUnary } from "../../ops/index.js";

type Complex = { re: number; im: number };
type ComplexFn = (re: number, im: number) => Complex;

// ── Registration helpers ────────────────────────────────────────────────

/** A floating-point unary element-wise builtin (sin, exp, sqrt, ...). */
function floatUnary(
  name: string,
  fns: UnaryElemwiseFns,
  opts: { mfile?: boolean; help?: BuiltinHelp } = {}
): void {
  const f = unaryElemwise(fns, name);
  builtin({
    name,
    singleOutputs: true,
    help: opts.help,
    nargin: 1,
    mfile: opts.mfile,
    apply: args => f([floatArg(args[0])]),
  });
}

function registerUnary(
  name: string,
  realFn: (x: number) => number,
  complexFn: ComplexFn,
  mfile = false,
  op?: number
): void {
  floatUnary(name, { realFn, complexFn, op }, { mfile });
}

/** MATLAB's error for sparse input to erf, gamma, ... */
const NOT_FULL_REAL = (name: string) => ({
  identifier: `MATLAB:${name}:notFullReal`,
  message: "Input must be real and full.",
});

/** A real-only floating-point unary builtin (erf, gamma, ...): complex or
 *  sparse input raises `err`. */
function realOnlyUnary(
  name: string,
  realFn: (x: number) => number,
  err: () => RuntimeError = () => notFullRealError(name)
): void {
  const f = unaryElemwise(
    { realFn, complexFn: () => ({ re: NaN, im: NaN }) },
    name
  );
  builtin({
    name,
    singleOutputs: true,
    nargin: 1,
    accepts: { sparse: NOT_FULL_REAL(name) },
    apply: args => f([realFloatArg(args[0], err)]),
  });
}

/** Element-wise binary builtin on real floating-point arguments. */
function realFloatBinary(
  name: string,
  fn: (a: number, b: number) => number,
  err: () => RuntimeError,
  mfile = false
): void {
  builtin({
    name,
    singleOutputs: true,
    nargin: 2,
    mfile,
    apply: args =>
      applyBinaryElemwise(
        [realFloatArg(args[0], err), realFloatArg(args[1], err)],
        fn
      ),
  });
}

// ── Complex helpers ─────────────────────────────────────────────────────

// Guard 0*Inf=NaN on the imaginary axis: a pure-imaginary input
// (sin(re)/cos(re) === 0) has a 0 lane, but cosh/sinh overflow to Inf
// for large |im| and 0*Inf would poison it. C99 csin/ccos special-case
// this; match it so all --opt levels agree.
const cSin: ComplexFn = (re, im) => {
  const sr = Math.sin(re);
  const cr = Math.cos(re);
  return {
    re: sr === 0 ? 0 : sr * Math.cosh(im),
    im: cr === 0 ? 0 : cr * Math.sinh(im),
  };
};

const cCos: ComplexFn = (re, im) => {
  const cr = Math.cos(re);
  const sr = Math.sin(re);
  return {
    re: cr === 0 ? 0 : cr * Math.cosh(im),
    im: sr === 0 ? 0 : -sr * Math.sinh(im),
  };
};

const cTan: ComplexFn = (re, im) => {
  const denom = Math.cos(2 * re) + Math.cosh(2 * im);
  return { re: Math.sin(2 * re) / denom, im: Math.sinh(2 * im) / denom };
};

const cAtan: ComplexFn = (re, im) => {
  // Real-valued input: clean real atan — avoids the spurious ±1e-17i
  // residue (and the Inf/Inf NaN at large |re|) the complex formula
  // leaves on the real axis.
  if (im === 0) return { re: Math.atan(re), im: 0 };
  // atan(z) = (i/2)·log((1 − iz)/(1 + iz)). MATLAB's branch for a
  // pure-imaginary z with |z| > 1 is −π/2 + … (verified: atan(2i) =
  // −1.5708 + 0.5493i); the libc/C99 Annex-G branch differs (+π/2), so
  // the C runtime overrides its catan to match this.
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
};

const cSinh: ComplexFn = (re, im) => ({
  re: Math.sinh(re) * Math.cos(im),
  im: Math.cosh(re) * Math.sin(im),
});

const cCosh: ComplexFn = (re, im) => ({
  re: Math.cosh(re) * Math.cos(im),
  im: Math.sinh(re) * Math.sin(im),
});

const cTanh: ComplexFn = (re, im) => {
  const denom = Math.cosh(2 * re) + Math.cos(2 * im);
  return { re: Math.sinh(2 * re) / denom, im: Math.sin(2 * im) / denom };
};

const complexExp: ComplexFn = (re, im) => ({
  re: Math.exp(re) * Math.cos(im),
  im: Math.exp(re) * Math.sin(im),
});

function complexLog(re: number, im: number): Complex {
  return {
    re: Math.log(Math.hypot(re, im)),
    im: Math.atan2(im, re),
  };
}

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

/** 1 / fn(z): the reciprocal functions (sec = 1/cos, ...). */
function recipOf(fn: ComplexFn): ComplexFn {
  return (re, im) => {
    const r = fn(re, im);
    return cRecip(r.re, r.im);
  };
}

const deg2rad = Math.PI / 180;
const rad2deg = 180 / Math.PI;

/** fn(z in degrees). */
function fromDeg(fn: ComplexFn): ComplexFn {
  return (re, im) => fn(re * deg2rad, im * deg2rad);
}

/** fn(z) converted to degrees. */
function toDegComplex(fn: ComplexFn): ComplexFn {
  return (re, im) => {
    const r = fn(re, im);
    return { re: r.re * rad2deg, im: r.im * rad2deg };
  };
}

// ── Trig ────────────────────────────────────────────────────────────────

registerUnary("sin", Math.sin, cSin, false, OpUnary.SIN);
registerUnary("cos", Math.cos, cCos, false, OpUnary.COS);
registerUnary("tan", Math.tan, cTan, false, OpUnary.TAN);

// asin/acos: maybe-complex — out-of-domain real inputs (|x| > 1) produce
// complex results.
floatUnary("asin", {
  realFn: Math.asin,
  complexFn: cAsin,
  maybeComplex: true,
  op: OpUnary.ASIN,
});
floatUnary("acos", {
  realFn: Math.acos,
  complexFn: cAcos,
  maybeComplex: true,
  op: OpUnary.ACOS,
});
registerUnary("atan", Math.atan, cAtan, false, OpUnary.ATAN);

// ── Hyperbolic ──────────────────────────────────────────────────────────

registerUnary("sinh", Math.sinh, cSinh, false, OpUnary.SINH);
registerUnary("cosh", Math.cosh, cCosh, false, OpUnary.COSH);
registerUnary("tanh", Math.tanh, cTanh, false, OpUnary.TANH);

// ── Exp / Log ───────────────────────────────────────────────────────────

registerUnary("exp", Math.exp, complexExp, false, OpUnary.EXP);

floatUnary("log", {
  realFn: Math.log,
  complexFn: complexLog,
  maybeComplex: true,
  op: OpUnary.LOG,
});

const complexLog2 = (re: number, im: number) => ({
  re: Math.log(Math.hypot(re, im)) / Math.LN2,
  im: Math.atan2(im, re) / Math.LN2,
});

function frexpScalar(x: number): { f: number; e: number } {
  if (x === 0) return { f: 0, e: 0 };
  if (!isFinite(x)) return { f: x, e: 0 };
  const e = Math.floor(Math.log2(Math.abs(x))) + 1;
  return { f: x / Math.pow(2, e), e };
}

const log2Elemwise = unaryElemwise(
  {
    realFn: Math.log2,
    complexFn: complexLog2,
    maybeComplex: true,
    op: OpUnary.LOG2,
  },
  "log2"
);

// log2 has a two-output form: [f, e] = log2(x) splits x into mantissa and
// exponent (x = f .* 2.^e).
builtin({
  name: "log2",
  singleOutputs: true,
  nargin: 1,
  nargout: 2,
  apply: (args, nargout) => {
    const x = floatArg(args[0]);
    if (nargout <= 1) return log2Elemwise([x]);
    const v = realFloatArg(
      x,
      () => new RuntimeError("log2: the two-output form requires real input")
    );
    if (typeof v === "number") {
      const { f, e } = frexpScalar(v);
      return [f, e];
    }
    const n = v.data.length;
    const fData = allocFloat64Array(n);
    const eData = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const { f, e } = frexpScalar(v.data[i]);
      fData[i] = f;
      eData[i] = e;
    }
    return [
      makeTensor(fData, undefined, v.shape.slice()),
      makeTensor(eData, undefined, v.shape.slice()),
    ];
  },
});

const complexLog10 = (re: number, im: number) => ({
  re: Math.log(Math.hypot(re, im)) / Math.LN10,
  im: Math.atan2(im, re) / Math.LN10,
});

floatUnary("log10", {
  realFn: Math.log10,
  complexFn: complexLog10,
  maybeComplex: true,
  op: OpUnary.LOG10,
});

// ── Abs ─────────────────────────────────────────────────────────────────

builtin({
  name: "abs",
  singleOutputs: true,
  nargin: 1,
  apply: unaryRealResult(Math.abs, Math.hypot, "abs", OpUnary.ABS),
});

// ── Sqrt ────────────────────────────────────────────────────────────────

function complexSqrt(re: number, im: number): Complex {
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

floatUnary("sqrt", {
  realFn: x => (x >= 0 ? Math.sqrt(x) : NaN),
  complexFn: complexSqrt,
  maybeComplex: true,
  op: OpUnary.SQRT,
});

// ── Realsqrt ────────────────────────────────────────────────────────────

const realsqrtElemwise = unaryElemwise(
  {
    realFn: x => {
      if (x < 0)
        throw builtinError(
          "MATLAB:realsqrt:complexResult",
          "Realsqrt produced complex result."
        );
      return Math.sqrt(x);
    },
    complexFn: () => ({ re: NaN, im: NaN }),
  },
  "realsqrt"
);

builtin({
  name: "realsqrt",
  singleOutputs: true,
  nargin: 1,
  mfile: true,
  apply: args => {
    // MATLAB validates realness before the floating-point type.
    if (isRuntimeChar(args[0])) throw mustBeFloatError();
    const notReal = () =>
      builtinError(
        "MATLAB:validators:mustBeReal",
        "Invalid argument at position 1. Value must be real."
      );
    let x: NumericArg;
    try {
      x = floatArg(args[0]);
    } catch {
      throw notReal();
    }
    return realsqrtElemwise([realFloatArg(x, notReal)]);
  },
});

// ── Sign ────────────────────────────────────────────────────────────────

builtin({
  name: "sign",
  singleOutputs: true,
  nargin: 1,
  apply: unaryElemwise(
    {
      realFn: x => Math.sign(x) + 0, // sign(-0) is 0
      op: OpUnary.SIGN,
      complexFn: (re, im) => {
        // Use hypot so magnitude stays finite when re*re or im*im overflow.
        const mag = Math.hypot(re, im);
        if (mag === 0) return { re: 0, im: 0 };
        return { re: re / mag, im: im / mag };
      },
    },
    "sign"
  ),
});

// ── Rounding ────────────────────────────────────────────────────────────

function registerRounding(
  name: string,
  fn: (x: number) => number,
  op: number
): void {
  builtin({
    name,
    singleOutputs: true,
    nargin: 1,
    apply: unaryElemwise(
      { realFn: fn, complexFn: (re, im) => ({ re: fn(re), im: fn(im) }), op },
      name
    ),
  });
}

registerRounding("floor", Math.floor, OpUnary.FLOOR);
registerRounding("ceil", Math.ceil, OpUnary.CEIL);
registerRounding("fix", Math.trunc, OpUnary.TRUNC);

// MATLAB round: half away from zero (not JS half-toward-+inf)
function matlabRound(x: number): number {
  return Math.sign(x) * Math.round(Math.abs(x));
}

const roundElemwise = unaryElemwise(
  {
    realFn: matlabRound,
    complexFn: (re, im) => ({ re: matlabRound(re), im: matlabRound(im) }),
  },
  "round"
);

function isNumericValue(v: RuntimeValue): boolean {
  return (
    typeof v === "number" ||
    typeof v === "boolean" ||
    isRuntimeTensor(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeChar(v)
  );
}

/** The digit count N of round(X, N): a real integer scalar (not logical). */
function roundDigitsArg(v: RuntimeValue): number {
  const err = () =>
    builtinError(
      "MATLAB:round:SecondArg",
      "Second argument must be a real integer scalar."
    );
  if (typeof v === "boolean" || (isRuntimeTensor(v) && v.cls === "logical"))
    throw err();
  let a: NumericArg;
  try {
    a = floatArg(v);
  } catch {
    throw err();
  }
  if (hasImag(a)) throw err();
  const d = realFloatArg(a, err);
  const n = typeof d === "number" ? d : d.data.length === 1 ? d.data[0] : NaN;
  if (!Number.isInteger(n)) throw err();
  return n;
}

builtin({
  name: "round",
  singleOutputs: true,
  // round(X), round(X, N), round(X, N, type), and the TieBreaker
  // name-value pair (not supported).
  nargin: [1, 5],
  apply: args => {
    if (args.length === 1) {
      if (typeof args[0] === "string" || !isNumericValue(args[0]))
        throw builtinError(
          "MATLAB:round:FirstArg",
          "First argument must be a numeric, logical, or char array."
        );
      return roundElemwise(args);
    }
    if (args.length > 3)
      throw new RuntimeError("round: the TieBreaker option is not supported");
    let x: NumericArg;
    try {
      x = floatArg(args[0]);
    } catch {
      throw builtinError(
        "MATLAB:round:NonfloatMultipleArgs",
        "First argument must be double or single with more than one input."
      );
    }
    const n = roundDigitsArg(args[1]);
    let significant = false;
    if (args.length === 3) {
      const t = args[2];
      const s =
        isRuntimeChar(t) || typeof t === "string"
          ? (isRuntimeChar(t) ? t.value : t).toLowerCase()
          : "";
      if (s !== "decimals" && s !== "significant")
        throw builtinError(
          "MATLAB:round:ThirdArg",
          "With 3 or 5 input arguments, the third input must be either 'decimals' or 'significant'."
        );
      significant = s === "significant";
    }
    const roundN = significant
      ? (v: number): number => {
          if (v === 0 || !isFinite(v)) return v;
          const scale = Math.pow(
            10,
            n - Math.floor(Math.log10(Math.abs(v))) - 1
          );
          return matlabRound(v * scale) / scale;
        }
      : (() => {
          const scale = Math.pow(10, n);
          return (v: number): number => matlabRound(v * scale) / scale;
        })();
    return applyUnaryElemwise(
      x,
      roundN,
      (re, im) => ({ re: roundN(re), im: roundN(im) }),
      "round"
    );
  },
});

// ── Precision math: expm1, log1p ─────────────────────────────────────────

registerUnary("expm1", Math.expm1, (re, im) => {
  // expm1(z) = exp(z) - 1
  const r = complexExp(re, im);
  return { re: r.re - 1, im: r.im };
});

floatUnary("log1p", {
  realFn: Math.log1p,
  // log1p(z) = log(1 + z)
  complexFn: (re, im) => complexLog(1 + re, im),
  maybeComplex: true,
});

// ── Hypot ────────────────────────────────────────────────────────────────

// hypot(a, b) = sqrt(abs(a).^2 + abs(b).^2), so complex input is reduced
// to its magnitude first.
const absOf = unaryRealResult(Math.abs, Math.hypot, "hypot", OpUnary.ABS);

builtin({
  name: "hypot",
  singleOutputs: true,
  nargin: 2,
  apply: args =>
    applyBinaryElemwise(
      [absOf([floatArg(args[0])]), absOf([floatArg(args[1])])],
      OpBin.HYPOT
    ),
});

// ── Error functions ──────────────────────────────────────────────────────

realOnlyUnary("erf", erfScalar);
realOnlyUnary("erfc", erfcScalar);
realOnlyUnary("erfinv", erfinvScalar);
realOnlyUnary("erfcinv", erfcinvScalar);
realOnlyUnary("erfcx", erfcxScalar);

// ── Reciprocal trig ──────────────────────────────────────────────────

registerUnary("sec", (x: number) => 1 / Math.cos(x), recipOf(cCos), true);
registerUnary("csc", (x: number) => 1 / Math.sin(x), recipOf(cSin), true);
registerUnary("cot", (x: number) => 1 / Math.tan(x), recipOf(cTan), true);

// ── Degree trig ──────────────────────────────────────────────────────

registerUnary("sind", (x: number) => Math.sin(x * deg2rad), fromDeg(cSin));
registerUnary("cosd", (x: number) => Math.cos(x * deg2rad), fromDeg(cCos));
registerUnary("tand", (x: number) => Math.tan(x * deg2rad), fromDeg(cTan));
registerUnary(
  "secd",
  (x: number) => 1 / Math.cos(x * deg2rad),
  recipOf(fromDeg(cCos)),
  true
);
registerUnary(
  "cscd",
  (x: number) => 1 / Math.sin(x * deg2rad),
  recipOf(fromDeg(cSin)),
  true
);
registerUnary(
  "cotd",
  (x: number) => Math.cos(x * deg2rad) / Math.sin(x * deg2rad),
  recipOf(fromDeg(cTan)),
  true
);

// ── Inverse trig (degree output) ────────────────────────────────────

registerUnary(
  "atand",
  (x: number) => Math.atan(x) * rad2deg,
  toDegComplex(cAtan)
);
registerUnary(
  "acotd",
  (x: number) => Math.atan(1 / x) * rad2deg,
  toDegComplex(composeRecip(cAtan))
);

// asind/acosd: may produce complex for |x| > 1
floatUnary("asind", {
  realFn: (x: number) => Math.asin(x) * rad2deg,
  complexFn: toDegComplex(cAsin),
  maybeComplex: true,
});
floatUnary("acosd", {
  realFn: (x: number) => Math.acos(x) * rad2deg,
  complexFn: toDegComplex(cAcos),
  maybeComplex: true,
});

// ── Inverse reciprocal trig ─────────────────────────────────────────

floatUnary("asec", {
  realFn: (x: number) => Math.acos(1 / x),
  complexFn: composeRecip(cAcos),
  maybeComplex: true,
});
floatUnary("acsc", {
  realFn: (x: number) => Math.asin(1 / x),
  complexFn: composeRecip(cAsin),
  maybeComplex: true,
});
registerUnary("acot", (x: number) => Math.atan(1 / x), composeRecip(cAtan));

// ── Inverse reciprocal trig (degree output) ─────────────────────────

floatUnary("asecd", {
  realFn: (x: number) => Math.acos(1 / x) * rad2deg,
  complexFn: toDegComplex(composeRecip(cAcos)),
  maybeComplex: true,
});
floatUnary("acscd", {
  realFn: (x: number) => Math.asin(1 / x) * rad2deg,
  complexFn: toDegComplex(composeRecip(cAsin)),
  maybeComplex: true,
});

// ── atan2d (binary) ─────────────────────────────────────────────────

realFloatBinary(
  "atan2d",
  (y, x) => Math.atan2(y, x) * rad2deg,
  mustBeRealError
);

// ── Inverse hyperbolic ──────────────────────────────────────────────

floatUnary("asinh", {
  realFn: Math.asinh,
  complexFn: cAsinh,
  maybeComplex: true,
});
floatUnary("acosh", {
  realFn: Math.acosh,
  complexFn: cAcosh,
  maybeComplex: true,
});
floatUnary("atanh", {
  realFn: Math.atanh,
  complexFn: cAtanh,
  maybeComplex: true,
});

// ── Reciprocal hyperbolic ───────────────────────────────────────────

registerUnary("sech", (x: number) => 1 / Math.cosh(x), recipOf(cCosh), true);
registerUnary("csch", (x: number) => 1 / Math.sinh(x), recipOf(cSinh), true);
registerUnary("coth", (x: number) => 1 / Math.tanh(x), recipOf(cTanh), true);

// ── Inverse reciprocal hyperbolic ───────────────────────────────────

floatUnary("asech", {
  realFn: (x: number) => Math.acosh(1 / x),
  complexFn: composeRecip(cAcosh),
  maybeComplex: true,
});
floatUnary("acsch", {
  realFn: (x: number) => Math.asinh(1 / x),
  complexFn: composeRecip(cAsinh),
  maybeComplex: true,
});
floatUnary("acoth", {
  realFn: (x: number) => Math.atanh(1 / x),
  complexFn: composeRecip(cAtanh),
  maybeComplex: true,
});

// ── pow2, nextpow2 ──────────────────────────────────────────────────

// pow2(X) = 2.^X; pow2(F, E) = F .* 2.^E (real only).
const pow2Elemwise = unaryElemwise(
  {
    realFn: (x: number) => Math.pow(2, x),
    complexFn: (re, im) => complexExp(re * Math.LN2, im * Math.LN2),
  },
  "pow2"
);

builtin({
  name: "pow2",
  singleOutputs: true,
  nargin: [1, 2],
  apply: args => {
    if (args.length === 1) return pow2Elemwise([floatArg(args[0])]);
    const err = () =>
      new RuntimeError(
        "pow2: complex input is not supported with two arguments"
      );
    return applyBinaryElemwise(
      [realFloatArg(args[0], err), realFloatArg(args[1], err)],
      (f, e) => f * Math.pow(2, e)
    );
  },
});

// nextpow2(X): the smallest integer p with 2^p >= abs(X).
function nextpow2Scalar(x: number): number {
  const a = Math.abs(x);
  if (a === 0) return 0;
  if (!isFinite(a)) return a;
  return Math.ceil(Math.log2(a));
}

builtin({
  name: "nextpow2",
  singleOutputs: true,
  nargin: 1,
  mfile: true,
  apply: unaryRealResult(
    nextpow2Scalar,
    (re, im) => nextpow2Scalar(Math.hypot(re, im)),
    "nextpow2"
  ),
});

// ── nthroot (binary) ────────────────────────────────────────────────

builtin({
  name: "nthroot",
  singleOutputs: true,
  nargin: 2,
  mfile: true,
  apply: args => {
    const err = () =>
      builtinError("MATLAB:nthroot:ComplexInput", "Both X and N must be real.");
    return applyBinaryElemwise(
      [realDenseArg(args[0], err), realDenseArg(args[1], err)],
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
      }
    );
  },
});

// ── gamma, factorial, beta ──────────────────────────────────────────

realOnlyUnary("gamma", (x: number) => {
  if (x === 0) return Infinity;
  if (x < 0 && Math.floor(x) === x) return NaN;
  if (x === Infinity) return Infinity;
  if (x === -Infinity || Number.isNaN(x)) return NaN;
  return lanczosGamma(x);
});

realOnlyUnary("gammaln", (x: number) => {
  if (x < 0)
    throw builtinError(
      "MATLAB:gammaln:negativeVal",
      "Input must be nonnegative."
    );
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
});

const factorialElemwise = unaryElemwise(
  {
    realFn: (n: number) => {
      if (n < 0 || !Number.isFinite(n)) return NaN;
      if (Number.isInteger(n) && n <= 20) {
        let result = 1;
        for (let i = 2; i <= n; i++) result *= i;
        return result;
      }
      return lanczosGamma(n + 1);
    },
    complexFn: () => ({ re: NaN, im: NaN }),
  },
  "factorial"
);

builtin({
  name: "factorial",
  nargin: 1,
  mfile: true,
  apply: args => {
    if (isRuntimeChar(args[0]))
      throw builtinError(
        "MATLAB:factorial:unsupportedType",
        "Input must be numeric."
      );
    const err = () =>
      builtinError(
        "MATLAB:factorial:NNegativeInt",
        "N must be an array of real non-negative integers."
      );
    let x: NumericArg;
    try {
      x = floatArg(args[0]);
    } catch {
      throw err();
    }
    return factorialElemwise([realFloatArg(x, err)]);
  },
});

builtin({
  name: "beta",
  singleOutputs: true,
  nargin: 2,
  mfile: true,
  apply: args => {
    const err = () => notFullRealError("gammaln");
    return applyBinaryElemwise(
      [realFloatArg(args[0], err), realFloatArg(args[1], err)],
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
      }
    );
  },
});

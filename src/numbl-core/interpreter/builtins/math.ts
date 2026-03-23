/**
 * Unary math builtins: trig, hyperbolic, exp/log, rounding, abs, sqrt, sign.
 */

import { FloatXArray, isRuntimeTensor } from "../../runtime/types.js";
import {
  registerIBuiltin,
  makeTensor,
  unaryPreserveType,
  unaryAlwaysReal,
  applyUnaryElemwise,
  applyUnaryElemwiseMaybeComplex,
  applyUnaryRealResult,
  unaryMathJitEmit,
} from "./types.js";
import type { JitType } from "../jit/jitTypes.js";

// ── Simple unary registration helper ────────────────────────────────────

function registerUnary(
  name: string,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null
): void {
  registerIBuiltin({
    name,
    typeRule: argTypes => unaryPreserveType(argTypes),
    apply: args => applyUnaryElemwise(args[0], realFn, complexFn, name),
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

// asin/acos: use maybe-complex variant so out-of-domain real inputs (e.g. acos(2)) produce complex
registerIBuiltin({
  name: "asin",
  typeRule: argTypes => unaryPreserveType(argTypes),
  apply: args =>
    applyUnaryElemwiseMaybeComplex(args[0], Math.asin, cAsin, "asin"),
  jitEmit: unaryMathJitEmit("Math.asin", "tAsin"),
});
registerIBuiltin({
  name: "acos",
  typeRule: argTypes => unaryPreserveType(argTypes),
  apply: args =>
    applyUnaryElemwiseMaybeComplex(args[0], Math.acos, cAcos, "acos"),
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

registerIBuiltin({
  name: "exp",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    switch (a.kind) {
      case "number":
        return [{ kind: "number", nonneg: true }];
      case "complex":
        return [{ kind: "complex" }];
      case "realTensor":
        return [{ kind: "realTensor", shape: a.shape, nonneg: true }];
      case "complexTensor":
        return [{ kind: "complexTensor", shape: a.shape }];
      default:
        return null;
    }
  },
  apply: args =>
    applyUnaryElemwise(
      args[0],
      Math.exp,
      (re, im) => ({
        re: Math.exp(re) * Math.cos(im),
        im: Math.exp(re) * Math.sin(im),
      }),
      "exp"
    ),
  jitEmit: unaryMathJitEmit("Math.exp", "tExp"),
});

function complexLog(re: number, im: number): { re: number; im: number } {
  return { re: Math.log(Math.sqrt(re * re + im * im)), im: Math.atan2(im, re) };
}

registerIBuiltin({
  name: "log",
  typeRule: argTypes => unaryPreserveType(argTypes),
  apply: args =>
    applyUnaryElemwiseMaybeComplex(args[0], Math.log, complexLog, "log"),
  jitEmit: unaryMathJitEmit("Math.log", "tLog", true),
});

const complexLog2 = (re: number, im: number) => ({
  re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN2,
  im: Math.atan2(im, re) / Math.LN2,
});

registerIBuiltin({
  name: "log2",
  typeRule: (argTypes, nargout) => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (nargout > 1) {
      // [f, e] = log2(x) — frexp form, both outputs are real numbers
      switch (a.kind) {
        case "number":
          return [{ kind: "number" }, { kind: "number" }];
        case "realTensor":
          return [
            { kind: "realTensor", shape: a.shape },
            { kind: "realTensor", shape: a.shape },
          ];
        default:
          return null;
      }
    }
    switch (a.kind) {
      case "number":
        return [{ kind: "number" }];
      case "complex":
        return [{ kind: "complex" }];
      case "realTensor":
        return [{ kind: "realTensor", shape: a.shape }];
      case "complexTensor":
        return [{ kind: "complexTensor", shape: a.shape }];
      default:
        return null;
    }
  },
  apply: (args, nargout) => {
    if (nargout > 1) {
      // frexp: x = f * 2^e, 0.5 <= |f| < 1 (or f=0 if x=0)
      function frexpScalar(x: number): { f: number; e: number } {
        if (x === 0) return { f: 0, e: 0 };
        if (!isFinite(x)) return { f: x, e: 0 };
        const e = Math.floor(Math.log2(Math.abs(x))) + 1;
        return { f: x / Math.pow(2, e), e };
      }
      const v = args[0];
      if (typeof v === "number") {
        const { f, e } = frexpScalar(v);
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
    }
    return applyUnaryElemwiseMaybeComplex(
      args[0],
      Math.log2,
      complexLog2,
      "log2"
    );
  },
  jitEmit: unaryMathJitEmit("Math.log2", "tLog2", true),
});

const complexLog10 = (re: number, im: number) => ({
  re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN10,
  im: Math.atan2(im, re) / Math.LN10,
});

registerIBuiltin({
  name: "log10",
  typeRule: argTypes => unaryPreserveType(argTypes),
  apply: args =>
    applyUnaryElemwiseMaybeComplex(args[0], Math.log10, complexLog10, "log10"),
  jitEmit: unaryMathJitEmit("Math.log10", "tLog10", true),
});

// ── Abs ─────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "abs",
  typeRule: argTypes => unaryAlwaysReal(argTypes),
  apply: args =>
    applyUnaryRealResult(
      args[0],
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

registerIBuiltin({
  name: "sqrt",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    switch (a.kind) {
      case "number":
        if (a.nonneg) return [{ kind: "number", nonneg: true }];
        return [{ kind: "complex" }];
      case "complex":
        return [{ kind: "complex" }];
      case "realTensor":
        if (a.nonneg)
          return [{ kind: "realTensor", shape: a.shape, nonneg: true }];
        return [{ kind: "complexTensor", shape: a.shape }];
      case "complexTensor":
        return [{ kind: "complexTensor", shape: a.shape }];
      default:
        return null;
    }
  },
  apply: args => {
    return applyUnaryElemwiseMaybeComplex(
      args[0],
      x => (x >= 0 ? Math.sqrt(x) : NaN),
      complexSqrt,
      "sqrt"
    );
  },
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
  registerIBuiltin({
    name,
    typeRule: argTypes => {
      if (argTypes.length !== 1) return null;
      const a = argTypes[0];
      switch (a.kind) {
        case "number":
          return [{ kind: "number", nonneg: !!a.nonneg }];
        case "complex":
          return [{ kind: "complex" }];
        case "realTensor":
          return [{ kind: "realTensor", shape: a.shape, nonneg: !!a.nonneg }];
        case "complexTensor":
          return [{ kind: "complexTensor", shape: a.shape }];
        default:
          return null;
      }
    },
    apply: args =>
      applyUnaryElemwise(
        args[0],
        fn,
        (re, im) => ({ re: fn(re), im: fn(im) }),
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

registerIBuiltin({
  name: "round",
  typeRule: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    switch (a.kind) {
      case "number":
        return [{ kind: "number", nonneg: !!a.nonneg }];
      case "complex":
        return [{ kind: "complex" }];
      case "realTensor":
        return [{ kind: "realTensor", shape: a.shape, nonneg: !!a.nonneg }];
      case "complexTensor":
        return [{ kind: "complexTensor", shape: a.shape }];
      default:
        return null;
    }
  },
  apply: args => {
    if (args.length === 1) {
      return applyUnaryElemwise(
        args[0],
        matlabRound,
        (re, im) => ({ re: matlabRound(re), im: matlabRound(im) }),
        "round"
      );
    }
    // Two-arg form: round(x, n) — round to n decimal places
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
});

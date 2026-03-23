/**
 * Unary math builtins: trig, hyperbolic, exp/log, rounding, abs, sqrt, sign.
 */

import {
  FloatXArray,
  isRuntimeComplexNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import {
  registerIBuiltin,
  mkc,
  makeTensor,
  unaryPreserveType,
  unaryAlwaysReal,
  applyUnaryElemwise,
  applyUnaryRealResult,
} from "./types.js";

// ── Simple unary registration helper ────────────────────────────────────

function registerUnary(
  name: string,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number }
): void {
  registerIBuiltin({
    name,
    typeRule: argTypes => unaryPreserveType(argTypes),
    apply: args => applyUnaryElemwise(args[0], realFn, complexFn, name),
  });
}

// ── Trig ────────────────────────────────────────────────────────────────

registerUnary("sin", Math.sin, (re, im) => ({
  re: Math.sin(re) * Math.cosh(im),
  im: Math.cos(re) * Math.sinh(im),
}));

registerUnary("cos", Math.cos, (re, im) => ({
  re: Math.cos(re) * Math.cosh(im),
  im: -Math.sin(re) * Math.sinh(im),
}));

registerUnary("tan", Math.tan, (re, im) => {
  const denom = Math.cos(2 * re) + Math.cosh(2 * im);
  return { re: Math.sin(2 * re) / denom, im: Math.sinh(2 * im) / denom };
});

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

registerUnary("asin", Math.asin, cAsin);
registerUnary("acos", Math.acos, cAcos);
registerUnary("atan", Math.atan, (re, im) => {
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
});

// ── Hyperbolic ──────────────────────────────────────────────────────────

registerUnary("sinh", Math.sinh, (re, im) => ({
  re: Math.sinh(re) * Math.cos(im),
  im: Math.cosh(re) * Math.sin(im),
}));

registerUnary("cosh", Math.cosh, (re, im) => ({
  re: Math.cosh(re) * Math.cos(im),
  im: Math.sinh(re) * Math.sin(im),
}));

registerUnary("tanh", Math.tanh, (re, im) => {
  const denom = Math.cosh(2 * re) + Math.cos(2 * im);
  return { re: Math.sinh(2 * re) / denom, im: Math.sin(2 * im) / denom };
});

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
        return [{ kind: "realTensor", nonneg: true }];
      case "complexTensor":
        return [{ kind: "complexTensor" }];
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
});

function complexLog(re: number, im: number): { re: number; im: number } {
  return { re: Math.log(Math.sqrt(re * re + im * im)), im: Math.atan2(im, re) };
}

registerUnary("log", Math.log, complexLog);

registerUnary("log2", Math.log2, (re, im) => ({
  re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN2,
  im: Math.atan2(im, re) / Math.LN2,
}));

registerUnary("log10", Math.log10, (re, im) => ({
  re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN10,
  im: Math.atan2(im, re) / Math.LN10,
}));

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
        if (a.nonneg) return [{ kind: "realTensor", nonneg: true }];
        return [{ kind: "complexTensor" }];
      case "complexTensor":
        return [{ kind: "complexTensor" }];
      default:
        return null;
    }
  },
  apply: args => {
    const v = args[0];

    if (typeof v === "number") {
      if (v >= 0) return Math.sqrt(v);
      const r = complexSqrt(v, 0);
      return mkc(r.re, r.im);
    }

    if (isRuntimeComplexNumber(v)) {
      const r = complexSqrt(v.re, v.im);
      return mkc(r.re, r.im);
    }

    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      if (!v.imag) {
        let hasNeg = false;
        for (let i = 0; i < n; i++) {
          if (v.data[i] < 0) {
            hasNeg = true;
            break;
          }
        }
        if (!hasNeg) {
          const out = new FloatXArray(n);
          for (let i = 0; i < n; i++) out[i] = Math.sqrt(v.data[i]);
          return makeTensor(out, undefined, v.shape.slice());
        }
        const outR = new FloatXArray(n);
        const outI = new FloatXArray(n);
        for (let i = 0; i < n; i++) {
          const r = complexSqrt(v.data[i], 0);
          outR[i] = r.re;
          outI[i] = r.im;
        }
        return makeTensor(outR, outI, v.shape.slice());
      }
      const outR = new FloatXArray(n);
      const outI = new FloatXArray(n);
      for (let i = 0; i < n; i++) {
        const r = complexSqrt(v.data[i], v.imag[i]);
        outR[i] = r.re;
        outI[i] = r.im;
      }
      return makeTensor(outR, outI, v.shape.slice());
    }

    throw new Error("sqrt: unsupported argument type");
  },
});

// ── Sign ────────────────────────────────────────────────────────────────

registerUnary("sign", Math.sign, (re, im) => {
  const mag = Math.sqrt(re * re + im * im);
  if (mag === 0) return { re: 0, im: 0 };
  return { re: re / mag, im: im / mag };
});

// ── Rounding ────────────────────────────────────────────────────────────

function registerRounding(name: string, fn: (x: number) => number): void {
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
          return [{ kind: "realTensor", nonneg: !!a.nonneg }];
        case "complexTensor":
          return [{ kind: "complexTensor" }];
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
  });
}

registerRounding("floor", Math.floor);
registerRounding("ceil", Math.ceil);
registerRounding("round", Math.round);
registerRounding("fix", Math.trunc);

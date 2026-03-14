/**
 * Mathematical builtin functions (elementwise and binary operations)
 */

import { RTV, toNumber, RuntimeError, RuntimeValue } from "../runtime/index.js";
import { ItemType } from "../lowering/itemTypes.js";
import {
  register,
  builtinSingle,
  BuiltinFn,
  BuiltinFnBranch,
} from "./registry.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeChar,
  isRuntimeClassInstance,
  RuntimeTensor,
} from "../runtime/types.js";
import {
  lanczosGamma,
  besselj,
  bessely,
  besseli,
  besselk,
  airyAi,
  airyAiPrime,
  airyBi,
  airyBiPrime,
  airyAllComplex,
} from "./bessel.js";
import {
  erfcinvScalar,
  erfcScalar,
  erfcxScalar,
  erfinvScalar,
  erfScalar,
} from "./erf.js";

// ── Complex math formulas (reusable for reciprocal variants) ────────────

type ComplexResult = { re: number; im: number };

/** Complex reciprocal: 1/z */
function cRecip(re: number, im: number): ComplexResult {
  const d = re * re + im * im;
  return { re: re / d, im: -im / d };
}

/** Compose: apply complexFn to 1/z */
function composeRecip(
  complexFn: (re: number, im: number) => ComplexResult
): (re: number, im: number) => ComplexResult {
  return (re, im) => {
    const r = cRecip(re, im);
    return complexFn(r.re, r.im);
  };
}

function cAcos(re: number, im: number): ComplexResult {
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

function cAsin(re: number, im: number): ComplexResult {
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

function cAcosh(re: number, im: number): ComplexResult {
  // acosh(z) = log(z + sqrt(z-1)*sqrt(z+1))
  // Using separate sqrt(z-1)*sqrt(z+1) instead of sqrt(z^2-1) for correct
  // branch cut behavior when re < -1, im = 0
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

function cAsinh(re: number, im: number): ComplexResult {
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

function cAtanh(re: number, im: number): ComplexResult {
  // For purely real values on the branch cut (|re| > 1, im === 0),
  // use convention: imaginary part has sign(re) * pi/2
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

// ── Complex-aware math helpers ───────────────────────────────────────────

function applyToComplex(
  v: RuntimeValue,
  fn: (re: number, im: number) => { re: number; im: number }
): RuntimeValue {
  // Handle scalars
  if (isRuntimeComplexNumber(v)) {
    const result = fn(v.re, v.im);
    return result.im === 0
      ? RTV.num(result.re)
      : RTV.complex(result.re, result.im);
  }
  if (isRuntimeNumber(v)) {
    const result = fn(v, 0);
    return result.im === 0
      ? RTV.num(result.re)
      : RTV.complex(result.re, result.im);
  }
  if (isRuntimeLogical(v)) {
    const result = fn(v ? 1 : 0, 0);
    return result.im === 0
      ? RTV.num(result.re)
      : RTV.complex(result.re, result.im);
  }
  // Handle tensors
  if (isRuntimeTensor(v)) {
    const resultRe = new FloatXArray(v.data.length);
    const resultIm = new FloatXArray(v.data.length);
    for (let i = 0; i < v.data.length; i++) {
      const r = fn(v.data[i], v.imag ? v.imag[i] : 0);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
    // Check if result is purely real
    const isReal = resultIm.every(x => x === 0);
    return RTV.tensor(resultRe, v.shape, isReal ? undefined : resultIm);
  }
  throw new RuntimeError(`Expected numeric argument`);
}

/** Create a complex-aware elemwise builtin. Prepends complex branch before the real branch. */
function complexElemwise(
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  complexOutKind: "ComplexNumber" | "Number",
  o?: { nativeJs: string }
): BuiltinFn {
  const complexApply = (args: RuntimeValue[]) => {
    if (args.length !== 1) throw new RuntimeError(`Expected 1 argument`);
    return applyToComplex(args[0], complexFn);
  };
  const realApply = (args: RuntimeValue[]) => {
    if (args.length !== 1) throw new RuntimeError(`Expected 1 argument`);
    const v = args[0];
    // Handle complex at runtime even from real branch (e.g., sqrt(-1), acos(2))
    if (isRuntimeComplexNumber(v)) return applyToComplex(v, complexFn);
    if (isRuntimeNumber(v)) {
      const result = realFn(v);
      if (isNaN(result)) {
        // Real input outside domain — use complex formula (e.g., acos(2), log(-1))
        const r = complexFn(v, 0);
        return r.im === 0 ? RTV.num(r.re) : RTV.complex(r.re, r.im);
      }
      return RTV.num(result);
    }
    if (isRuntimeLogical(v)) return RTV.num(realFn(v ? 1 : 0));
    if (isRuntimeTensor(v)) {
      // Handle complex tensors
      if (v.imag !== undefined) {
        const resultRe = new FloatXArray(v.data.length);
        const resultIm = new FloatXArray(v.data.length);
        for (let i = 0; i < v.data.length; i++) {
          const r = complexFn(v.data[i], v.imag[i]);
          resultRe[i] = r.re;
          resultIm[i] = r.im;
        }
        const isReal =
          complexOutKind === "Number" || resultIm.every(x => x === 0);
        return RTV.tensor(resultRe, v.shape, isReal ? undefined : resultIm);
      }
      // Real tensor — some elements may be out of domain (e.g., acos([0, 2]))
      const resultRe = new FloatXArray(v.data.length);
      const resultIm = new FloatXArray(v.data.length);
      let hasImag = false;
      for (let i = 0; i < v.data.length; i++) {
        const r = realFn(v.data[i]);
        if (isNaN(r)) {
          const c = complexFn(v.data[i], 0);
          resultRe[i] = c.re;
          resultIm[i] = c.im;
          if (c.im !== 0) hasImag = true;
        } else {
          resultRe[i] = r;
          resultIm[i] = 0;
        }
      }
      return RTV.tensor(resultRe, v.shape, hasImag ? resultIm : undefined);
    }
    throw new RuntimeError("Expected numeric argument");
  };

  const ret: BuiltinFnBranch[] = [];
  if (o?.nativeJs) {
    ret.push({
      check: (argTypes: ItemType[]) => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "Number") return null;
        return { outputTypes: [{ kind: "Number" }] };
      },
      apply: realApply,
      nativeJsFn: o.nativeJs,
    });
  }
  ret.push({
    check: (argTypes: ItemType[]) => {
      if (argTypes.length !== 1) return null;
      if (argTypes[0].kind !== "ComplexNumber") return null;
      return { outputTypes: [{ kind: complexOutKind }] };
    },
    apply: complexApply,
  });
  ret.push({
    check:
      complexOutKind === "ComplexNumber"
        ? elemwiseCheck
        : elemwiseCheckRealOutput,
    apply: realApply,
  });
  return ret;
}

// ── Math (elementwise) ──────────────────────────────────────────────────

function elemwiseCheck(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  if (nargout !== 1 || argTypes.length !== 1) return null;
  const t = argTypes[0];
  if (t.kind === "Number" || t.kind === "Boolean")
    return { outputTypes: [{ kind: "Number" }] };
  if (t.kind === "Tensor") return { outputTypes: [t] };
  return { outputTypes: [{ kind: "Unknown" }] };
}

function elemwiseCheckRealOutput(
  argTypes: ItemType[],
  nargout: number
): { outputTypes: ItemType[] } | null {
  if (nargout !== 1 || argTypes.length !== 1) return null;
  const t = argTypes[0];
  if (t.kind === "Number" || t.kind === "Boolean")
    return { outputTypes: [{ kind: "Number" }] };
  if (t.kind === "Tensor") {
    // return real-valued tensor of same shape (even if input is complex, since this is used for real-valued outputs like abs)
    return { outputTypes: [{ kind: "Tensor" }] };
  }
  return { outputTypes: [{ kind: "Unknown" }] };
}

function elemwise(
  fn: (x: number) => number,
  o?: { nativeJs?: string; isLogical?: boolean }
): BuiltinFn {
  const ret: BuiltinFnBranch[] = [];
  const apply = (args: RuntimeValue[]) => {
    if (args.length !== 1) throw new RuntimeError(`Expected 1 argument`);
    const v = args[0];
    if (isRuntimeNumber(v)) {
      const result = fn(v);
      return o?.isLogical ? RTV.logical(!!result) : RTV.num(result);
    }
    if (isRuntimeLogical(v)) {
      const result = fn(v ? 1 : 0);
      return o?.isLogical ? RTV.logical(!!result) : RTV.num(result);
    }
    if (isRuntimeTensor(v)) {
      const result = new FloatXArray(v.data.length);
      for (let i = 0; i < v.data.length; i++) result[i] = fn(v.data[i]);
      const t = RTV.tensor(result, v.shape);
      if (o?.isLogical) t._isLogical = true;
      return t;
    }
    throw new RuntimeError("Expected numeric argument");
  };
  if (o?.nativeJs) {
    const outKind = o.isLogical ? "Boolean" : "Number";
    ret.push({
      check: (argTypes: ItemType[]) => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "Number") return null;
        return { outputTypes: [{ kind: outKind }] };
      },
      apply,
      // Disable native JS fast-path for logical builtins — their scalar
      // results must stay as RuntimeValue logicals so the codegen emits
      // $rt.not() instead of the `=== 0` fast path.
      nativeJsFn: o.isLogical ? undefined : o.nativeJs,
    });
  }
  ret.push({
    check: elemwiseCheck,
    apply,
  });
  return ret;
}

/**
 * Like elemwise but also handles complex scalars and complex tensors
 * by applying fn independently to real and imaginary parts.
 */
function elemwiseComplex(
  fn: (x: number) => number,
  o?: { nativeJs?: string }
): BuiltinFn {
  const ret: BuiltinFnBranch[] = [];
  const apply = (args: RuntimeValue[]) => {
    if (args.length !== 1) throw new RuntimeError(`Expected 1 argument`);
    const v = args[0];
    if (isRuntimeNumber(v)) return RTV.num(fn(v));
    if (isRuntimeLogical(v)) return RTV.num(fn(v ? 1 : 0));
    if (isRuntimeComplexNumber(v)) return RTV.complex(fn(v.re), fn(v.im));
    if (isRuntimeTensor(v)) {
      const result = new FloatXArray(v.data.length);
      for (let i = 0; i < v.data.length; i++) result[i] = fn(v.data[i]);
      let imag: FloatXArrayType | undefined;
      if (v.imag) {
        imag = new FloatXArray(v.imag.length);
        for (let i = 0; i < v.imag.length; i++) imag[i] = fn(v.imag[i]);
      }
      return RTV.tensor(result, v.shape, imag);
    }
    throw new RuntimeError("Expected numeric argument");
  };
  if (o?.nativeJs) {
    ret.push({
      check: (argTypes: ItemType[]) => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "Number") return null;
        return { outputTypes: [{ kind: "Number" }] };
      },
      apply,
      nativeJsFn: o.nativeJs,
    });
  }
  ret.push({
    check: elemwiseCheck,
    apply,
  });
  return ret;
}

// ── Binary scalar/tensor dispatch ────────────────────────────────────────

/** Apply a binary (a, b) -> number function across scalar/tensor combinations. */
function binaryApply(
  a: RuntimeValue,
  b: RuntimeValue,
  fn: (x: number, y: number) => number
): RuntimeValue {
  const aIsT = isRuntimeTensor(a);
  const bIsT = isRuntimeTensor(b);
  if (aIsT && bIsT) {
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++)
      result[i] = fn(a.data[i], b.data[i]);
    return RTV.tensor(result, a.shape);
  }
  if (aIsT) {
    const bv = toNumber(b);
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = fn(a.data[i], bv);
    return RTV.tensor(result, a.shape);
  }
  if (bIsT) {
    const av = toNumber(a);
    const result = new FloatXArray(b.data.length);
    for (let i = 0; i < b.data.length; i++) result[i] = fn(av, b.data[i]);
    return RTV.tensor(result, b.shape);
  }
  return RTV.num(fn(toNumber(a), toNumber(b)));
}

function binaryScalar(
  fn: (a: number, b: number) => number,
  o?: { nativeJs: string }
): BuiltinFn {
  const ret: BuiltinFnBranch[] = [];
  const apply = (args: RuntimeValue[]) => {
    if (args.length !== 2) throw new RuntimeError(`Expected 2 arguments`);
    return binaryApply(args[0], args[1], fn);
  };
  if (o?.nativeJs) {
    ret.push({
      check: (argTypes: ItemType[]) => {
        if (argTypes.length !== 2) return null;
        if (argTypes[0].kind !== "Number" || argTypes[1].kind !== "Number")
          return null;
        return { outputTypes: [{ kind: "Number" }] };
      },
      apply,
      nativeJsFn: o.nativeJs,
    });
  }
  ret.push({
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    check: (_argTypes, _nargout) => {
      return { outputTypes: [{ kind: "Unknown" }] };
    },
    apply,
  });
  return ret;
}

export function registerMathFunctions(): void {
  register(
    "sin",
    complexElemwise(
      Math.sin,
      (re, im) => ({
        re: Math.sin(re) * Math.cosh(im),
        im: Math.cos(re) * Math.sinh(im),
      }),
      "ComplexNumber",
      { nativeJs: "Math.sin" }
    ),
    1
  );
  register(
    "cos",
    complexElemwise(
      Math.cos,
      (re, im) => ({
        re: Math.cos(re) * Math.cosh(im),
        im: -Math.sin(re) * Math.sinh(im),
      }),
      "ComplexNumber",
      { nativeJs: "Math.cos" }
    ),
    1
  );
  register(
    "tan",
    complexElemwise(
      Math.tan,
      (re, im) => {
        const denom = Math.cos(2 * re) + Math.cosh(2 * im);
        return { re: Math.sin(2 * re) / denom, im: Math.sinh(2 * im) / denom };
      },
      "ComplexNumber",
      { nativeJs: "Math.tan" }
    ),
    1
  );
  register(
    "asin",
    complexElemwise(Math.asin, cAsin, "ComplexNumber", {
      nativeJs: "Math.asin",
    }),
    1
  );
  register(
    "acos",
    complexElemwise(Math.acos, cAcos, "ComplexNumber", {
      nativeJs: "Math.acos",
    }),
    1
  );
  register(
    "atan",
    complexElemwise(
      Math.atan,
      (re, im) => {
        // atan(z) = i/2 * log((1-iz)/(1+iz))
        // iz = (-im, re)
        // 1 - iz = (1+im, -re)
        const w1re = 1 + im,
          w1im = -re;
        // 1 + iz = (1-im, re)
        const w2re = 1 - im,
          w2im = re;
        // (1-iz)/(1+iz): complex division
        const denom = w2re * w2re + w2im * w2im;
        const w3re = (w1re * w2re + w1im * w2im) / denom;
        const w3im = (w1im * w2re - w1re * w2im) / denom;
        // log(w3)
        const w4re = Math.log(Math.sqrt(w3re * w3re + w3im * w3im));
        const w4im = Math.atan2(w3im, w3re);
        // i/2 * log = (-w4im/2, w4re/2)
        return { re: -w4im / 2, im: w4re / 2 };
      },
      "ComplexNumber",
      { nativeJs: "Math.atan" }
    ),
    1
  );
  // Hyperbolic functions
  register(
    "sinh",
    complexElemwise(
      Math.sinh,
      (re, im) => ({
        re: Math.sinh(re) * Math.cos(im),
        im: Math.cosh(re) * Math.sin(im),
      }),
      "ComplexNumber",
      { nativeJs: "Math.sinh" }
    ),
    1
  );
  register(
    "cosh",
    complexElemwise(
      Math.cosh,
      (re, im) => ({
        re: Math.cosh(re) * Math.cos(im),
        im: Math.sinh(re) * Math.sin(im),
      }),
      "ComplexNumber",
      { nativeJs: "Math.cosh" }
    ),
    1
  );
  register(
    "tanh",
    complexElemwise(
      Math.tanh,
      (re, im) => {
        const denom = Math.cosh(2 * re) + Math.cos(2 * im);
        return {
          re: Math.sinh(2 * re) / denom,
          im: Math.sin(2 * im) / denom,
        };
      },
      "ComplexNumber",
      { nativeJs: "Math.tanh" }
    ),
    1
  );

  // Inverse hyperbolic functions
  register(
    "asinh",
    complexElemwise(Math.asinh, cAsinh, "ComplexNumber", {
      nativeJs: "Math.asinh",
    }),
    1
  );
  register(
    "acosh",
    complexElemwise(Math.acosh, cAcosh, "ComplexNumber", {
      nativeJs: "Math.acosh",
    }),
    1
  );
  register(
    "atanh",
    complexElemwise(Math.atanh, cAtanh, "ComplexNumber", {
      nativeJs: "Math.atanh",
    }),
    1
  );

  // Degree-based trigonometric functions
  const deg2rad = Math.PI / 180;
  register(
    "sind",
    elemwise((x: number) => Math.sin(x * deg2rad)),
    1
  );
  register(
    "cosd",
    elemwise((x: number) => Math.cos(x * deg2rad)),
    1
  );
  register(
    "tand",
    elemwise((x: number) => Math.tan(x * deg2rad)),
    1
  );

  register(
    "exp",
    complexElemwise(
      Math.exp,
      (re, im) => ({
        re: Math.exp(re) * Math.cos(im),
        im: Math.exp(re) * Math.sin(im),
      }),
      "ComplexNumber",
      { nativeJs: "Math.exp" }
    ),
    1
  );
  register(
    "log",
    complexElemwise(
      Math.log,
      (re, im) => ({
        re: Math.log(Math.sqrt(re * re + im * im)),
        im: Math.atan2(im, re),
      }),
      "ComplexNumber",
      { nativeJs: "Math.log" }
    ),
    1
  );
  {
    // log2 with single output: standard log base 2
    const log2Single = complexElemwise(
      Math.log2,
      (re, im) => ({
        re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN2,
        im: Math.atan2(im, re) / Math.LN2,
      }),
      "ComplexNumber",
      { nativeJs: "Math.log2" }
    );
    // log2 with two outputs: [F, E] = log2(X) where X = F .* 2.^E, 0.5 <= |F| < 1
    const frexpBranch: BuiltinFnBranch = {
      check: (argTypes: ItemType[], nargout: number) => {
        if (nargout !== 2 || argTypes.length !== 1) return null;
        const t = argTypes[0];
        if (t.kind === "Number" || t.kind === "Boolean" || t.kind === "Tensor")
          return { outputTypes: [{ kind: "Number" }, { kind: "Number" }] };
        return { outputTypes: [{ kind: "Unknown" }, { kind: "Unknown" }] };
      },
      apply: (args: RuntimeValue[]) => {
        if (args.length !== 1) throw new RuntimeError("Expected 1 argument");
        const v = args[0];
        if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
          const x = toNumber(v);
          if (x === 0) return [RTV.num(0), RTV.num(0)];
          const e = Math.floor(Math.log2(Math.abs(x))) + 1;
          const f = x * Math.pow(2, -e);
          return [RTV.num(f), RTV.num(e)];
        }
        if (isRuntimeTensor(v)) {
          const fData = new FloatXArray(v.data.length);
          const eData = new FloatXArray(v.data.length);
          for (let i = 0; i < v.data.length; i++) {
            const x = v.data[i];
            if (x === 0) {
              fData[i] = 0;
              eData[i] = 0;
            } else {
              const exp = Math.floor(Math.log2(Math.abs(x))) + 1;
              fData[i] = x * Math.pow(2, -exp);
              eData[i] = exp;
            }
          }
          return [RTV.tensor(fData, v.shape), RTV.tensor(eData, v.shape)];
        }
        throw new RuntimeError("Expected numeric argument");
      },
    };
    register("log2", [frexpBranch, ...log2Single], 1);
  }
  register(
    "log10",
    complexElemwise(
      Math.log10,
      (re, im) => ({
        re: Math.log(Math.sqrt(re * re + im * im)) / Math.LN10,
        im: Math.atan2(im, re) / Math.LN10,
      }),
      "ComplexNumber",
      { nativeJs: "Math.log10" }
    ),
    1
  );
  register(
    "sqrt",
    complexElemwise(
      Math.sqrt,
      (re, im) => {
        const r = Math.sqrt(re * re + im * im);
        const theta = Math.atan2(im, re);
        const sqrtR = Math.sqrt(r);
        return {
          re: sqrtR * Math.cos(theta / 2),
          im: sqrtR * Math.sin(theta / 2),
        };
      },
      "ComplexNumber"
    ),
    1
  );
  register(
    "abs",
    complexElemwise(
      Math.abs,
      (re, im) => ({ re: Math.sqrt(re * re + im * im), im: 0 }),
      "Number",
      { nativeJs: "Math.abs" }
    ),
    1
  );
  register("floor", elemwiseComplex(Math.floor, { nativeJs: "Math.floor" }), 1);
  register("ceil", elemwiseComplex(Math.ceil, { nativeJs: "Math.ceil" }), 1);
  // Rounds half away from zero (unlike JS Math.round which rounds half toward +inf)
  // Supports round(x) and round(x, n) where n is the number of decimal places
  register(
    "round",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("round requires 1 or 2 arguments");
      const roundHalfAwayFromZero = (x: number) =>
        Math.sign(x) * Math.round(Math.abs(x));
      const n = args.length === 2 ? toNumber(args[1]) : 0;
      const factor = Math.pow(10, n);
      const roundN = (x: number) =>
        n === 0
          ? roundHalfAwayFromZero(x)
          : roundHalfAwayFromZero(x * factor) / factor;
      const v = args[0];
      if (isRuntimeNumber(v)) return RTV.num(roundN(v));
      if (isRuntimeLogical(v)) return RTV.num(roundN(v ? 1 : 0));
      if (isRuntimeComplexNumber(v))
        return RTV.complex(roundN(v.re), roundN(v.im));
      if (isRuntimeTensor(v)) {
        const result = new FloatXArray(v.data.length);
        for (let i = 0; i < v.data.length; i++) result[i] = roundN(v.data[i]);
        let imag: FloatXArrayType | undefined;
        if (v.imag) {
          imag = new FloatXArray(v.imag.length);
          for (let i = 0; i < v.imag.length; i++) imag[i] = roundN(v.imag[i]);
        }
        return RTV.tensor(result, v.shape, imag);
      }
      throw new RuntimeError("round: expected numeric argument");
    })
  );
  register(
    "fix",
    elemwise(x => (x >= 0 ? Math.floor(x) : Math.ceil(x)), {
      nativeJs: "Math.trunc",
    }),
    1
  );
  register(
    "sign",
    builtinSingle(args => {
      if (args.length !== 1) throw new RuntimeError("sign requires 1 argument");
      const v = args[0];
      if (isRuntimeComplexNumber(v)) {
        const mag = Math.sqrt(v.re * v.re + v.im * v.im);
        if (mag === 0) return RTV.num(0);
        return RTV.complex(v.re / mag, v.im / mag);
      }
      if (isRuntimeNumber(v)) return RTV.num(Math.sign(v));
      if (isRuntimeLogical(v)) return RTV.num(Math.sign(v ? 1 : 0));
      if (isRuntimeTensor(v)) {
        if (v.imag) {
          // Complex tensor: sign(z) = z / abs(z)
          const resultRe = new FloatXArray(v.data.length);
          const resultIm = new FloatXArray(v.data.length);
          for (let i = 0; i < v.data.length; i++) {
            const re = v.data[i];
            const im = v.imag[i];
            const mag = Math.sqrt(re * re + im * im);
            if (mag === 0) {
              resultRe[i] = 0;
              resultIm[i] = 0;
            } else {
              resultRe[i] = re / mag;
              resultIm[i] = im / mag;
            }
          }
          return RTV.tensor(resultRe, v.shape, resultIm);
        }
        const result = new FloatXArray(v.data.length);
        for (let i = 0; i < v.data.length; i++)
          result[i] = Math.sign(v.data[i]);
        return RTV.tensor(result, v.shape);
      }
      throw new RuntimeError("Expected numeric argument");
    }),
    1
  );

  register(
    "gamma",
    elemwise(x => {
      // Handle special cases
      if (x === 0) return Infinity;
      if (x < 0 && Math.floor(x) === x) return NaN; // Negative integer
      if (x === Infinity) return Infinity;
      if (x === -Infinity || Number.isNaN(x)) return NaN;

      return lanczosGamma(x);
    }),
    1
  );

  register(
    "factorial",
    elemwise(n => {
      if (n < 0 || !Number.isFinite(n)) return NaN;
      // Exact integer computation for small n
      if (Number.isInteger(n) && n <= 20) {
        let result = 1;
        for (let i = 2; i <= n; i++) result *= i;
        return result;
      }
      // Use gamma for large or non-integer n: factorial(n) = gamma(n+1)
      return lanczosGamma(n + 1);
    }),
    1
  );

  // beta(x,y) = gamma(x)*gamma(y)/gamma(x+y)
  register(
    "beta",
    binaryScalar((x, y) => {
      // Handle special cases where x+y is a non-positive integer
      // but x and y individually are not
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
        // Both gamma(x) and gamma(y) are Inf => result depends
        if (!isFinite(gx) && !isFinite(gy)) return Infinity;
        if (!isFinite(gxy)) return NaN;
        return Infinity;
      }
      if (gxy === 0) return Infinity;
      return (gx / gxy) * gy;
    }),
    2
  );

  // Result has sign of divisor
  register(
    "mod",
    binaryScalar(
      (a, b) => {
        if (b === 0) return a;
        const m = a % b;
        return m !== 0 && Math.sign(m) !== Math.sign(b) ? m + b : m;
      },
      { nativeJs: "$rt.mod" }
    ),
    2
  );

  register(
    "rem",
    binaryScalar((a, b) => a % b, { nativeJs: "$rt.rem" }),
    2
  );

  register("atan2", binaryScalar(Math.atan2, { nativeJs: "Math.atan2" }), 2);

  register("power", binaryScalar(Math.pow, { nativeJs: "Math.pow" }), 2);

  register(
    "double",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("double requires 1 argument");
      const v = args[0];
      if (isRuntimeChar(v)) {
        if (v.value.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
        if (v.value.length === 1) return RTV.num(v.value.charCodeAt(0));
        const codes = Array.from(v.value).map(c => c.charCodeAt(0));
        return RTV.row(codes);
      }
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeTensor(v)) {
        if (v._isLogical) {
          return RTV.tensor(new FloatXArray(v.data), v.shape);
        }
        return v;
      }
      if (isRuntimeClassInstance(v) && v._builtinData !== undefined) {
        return v._builtinData;
      }
      return RTV.num(toNumber(v));
    }),
    1
  );

  // Shared factory for numeric predicate builtins (isnan, isinf, isfinite)
  function numericPredicate(
    name: string,
    scalarTest: (x: number) => boolean,
    logicalDefault: boolean,
    combineReIm: "or" | "and"
  ): void {
    register(
      name,
      builtinSingle(args => {
        if (args.length !== 1)
          throw new RuntimeError(`${name} requires 1 argument`);
        const v = args[0];
        if (isRuntimeNumber(v)) return RTV.logical(scalarTest(v));
        if (isRuntimeLogical(v)) return RTV.logical(logicalDefault);
        if (isRuntimeComplexNumber(v)) {
          const re = scalarTest(v.re),
            im = scalarTest(v.im);
          return RTV.logical(combineReIm === "or" ? re || im : re && im);
        }
        if (isRuntimeTensor(v)) {
          const result = new FloatXArray(v.data.length);
          for (let i = 0; i < v.data.length; i++) {
            const re = scalarTest(v.data[i]);
            const im = v.imag ? scalarTest(v.imag[i]) : logicalDefault;
            result[i] = (combineReIm === "or" ? re || im : re && im) ? 1 : 0;
          }
          const t = RTV.tensor(result, v.shape);
          t._isLogical = true;
          return t;
        }
        throw new RuntimeError("Expected numeric argument");
      }),
      1
    );
  }
  numericPredicate("isnan", Number.isNaN, false, "or");
  numericPredicate(
    "isinf",
    x => !Number.isFinite(x) && !Number.isNaN(x),
    false,
    "or"
  );
  numericPredicate("isfinite", Number.isFinite, true, "and");

  // ── Complex-specific builtins ─────────────────────────────────────────

  register(
    "real",
    builtinSingle(args => {
      if (args.length !== 1) throw new RuntimeError("real requires 1 argument");
      const v = args[0];
      if (isRuntimeComplexNumber(v)) return RTV.num(v.re);
      if (isRuntimeNumber(v)) return v;
      if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
      if (isRuntimeTensor(v)) {
        // Return real part of complex tensor, or tensor itself if already real
        if (v.imag === undefined) return v;
        return RTV.tensor(v.data, v.shape);
      }
      throw new RuntimeError("real requires a numeric argument");
    }),
    1
  );

  register(
    "imag",
    builtinSingle(args => {
      if (args.length !== 1) throw new RuntimeError("imag requires 1 argument");
      const v = args[0];
      if (isRuntimeComplexNumber(v)) return RTV.num(v.im);
      if (isRuntimeNumber(v)) return RTV.num(0);
      if (isRuntimeLogical(v)) return RTV.num(0);
      if (isRuntimeTensor(v)) {
        // Return imaginary part of complex tensor, or zeros if real
        if (v.imag === undefined) {
          return RTV.tensor(new FloatXArray(v.data.length), v.shape);
        }
        return RTV.tensor(v.imag, v.shape);
      }
      throw new RuntimeError("imag requires a numeric argument");
    }),
    1
  );

  register(
    "conj",
    builtinSingle(args => {
      if (args.length !== 1) throw new RuntimeError("conj requires 1 argument");
      const v = args[0];
      if (isRuntimeComplexNumber(v))
        return v.im === 0 ? RTV.num(v.re) : RTV.complex(v.re, -v.im);
      if (isRuntimeNumber(v) || isRuntimeLogical(v)) return v;
      if (isRuntimeTensor(v)) {
        // Return conjugate of complex tensor, or tensor itself if real
        if (v.imag === undefined) return v;
        const conjImag = new FloatXArray(v.imag.length);
        for (let i = 0; i < v.imag.length; i++) {
          conjImag[i] = -v.imag[i];
        }
        return RTV.tensor(v.data, v.shape, conjImag);
      }
      throw new RuntimeError("conj requires a numeric argument");
    }),
    1
  );

  register(
    "angle",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("angle requires 1 argument");
      const v = args[0];
      if (isRuntimeComplexNumber(v)) return RTV.num(Math.atan2(v.im, v.re));
      if (isRuntimeNumber(v)) return RTV.num(v >= 0 ? 0 : Math.PI);
      if (isRuntimeLogical(v)) return RTV.num(0);
      if (isRuntimeTensor(v)) {
        const result = new FloatXArray(v.data.length);
        for (let i = 0; i < v.data.length; i++) {
          const re = v.data[i];
          const im = v.imag ? v.imag[i] : 0;
          result[i] = Math.atan2(im, re);
        }
        return RTV.tensor(result, v.shape);
      }
      throw new RuntimeError("angle requires a numeric argument");
    }),
    1
  );

  register(
    "isreal",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isreal requires 1 argument");
      const v = args[0];
      if (isRuntimeComplexNumber(v)) return RTV.logical(false);
      if (isRuntimeTensor(v) && v.imag !== undefined) return RTV.logical(false);
      return RTV.logical(true);
    }),
    1
  );

  register(
    "complex",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("complex requires 1 or 2 arguments");
      const a = args[0];
      const b = args.length === 2 ? args[1] : undefined;
      // complex(a) - return a as complex (with zero imaginary part if real)
      if (b === undefined) {
        if (isRuntimeComplexNumber(a)) return a;
        if (isRuntimeNumber(a)) return RTV.complex(a, 0);
        if (isRuntimeLogical(a)) return RTV.complex(a ? 1 : 0, 0);
        if (isRuntimeTensor(a)) {
          const im = a.imag || new FloatXArray(a.data.length);
          return RTV.tensor(
            new FloatXArray(a.data),
            a.shape,
            new FloatXArray(im)
          );
        }
        throw new RuntimeError("complex requires numeric arguments");
      }
      // complex(a, b) - return a + bi
      if (
        (isRuntimeNumber(a) || isRuntimeLogical(a)) &&
        (isRuntimeNumber(b) || isRuntimeLogical(b))
      ) {
        const re = isRuntimeLogical(a) ? (a ? 1 : 0) : (a as number);
        const im = isRuntimeLogical(b) ? (b ? 1 : 0) : (b as number);
        return RTV.complex(re, im);
      }
      // Tensor cases
      if (isRuntimeTensor(a) || isRuntimeTensor(b)) {
        const aIsT = isRuntimeTensor(a);
        const bIsT = isRuntimeTensor(b);
        const aData = aIsT ? a.data : null;
        const bData = bIsT ? b.data : null;
        const shape = aIsT ? a.shape : (b as RuntimeTensor).shape;
        const len = aIsT ? a.data.length : (b as RuntimeTensor).data.length;
        const reArr = new FloatXArray(len);
        const imArr = new FloatXArray(len);
        const aScalar = !aIsT ? toNumber(a) : 0;
        const bScalar = !bIsT ? toNumber(b) : 0;
        for (let i = 0; i < len; i++) {
          reArr[i] = aData ? aData[i] : aScalar;
          imArr[i] = bData ? bData[i] : bScalar;
        }
        return RTV.tensor(reArr, shape, imArr);
      }
      throw new RuntimeError("complex requires numeric arguments");
    }),
    1
  );

  // ── Degree-based inverse trig ────────────────────────────────────────

  const rad2deg = 180 / Math.PI;
  register(
    "asind",
    elemwise((x: number) => Math.asin(x) * rad2deg),
    1
  );
  register(
    "acosd",
    elemwise((x: number) => Math.acos(x) * rad2deg),
    1
  );
  register(
    "atand",
    elemwise((x: number) => Math.atan(x) * rad2deg),
    1
  );
  register(
    "atan2d",
    binaryScalar((y, x) => Math.atan2(y, x) * rad2deg),
    2
  );

  // ── Reciprocal trig ────────────────────────────────────────────────

  register(
    "sec",
    elemwise((x: number) => 1 / Math.cos(x)),
    1
  );
  register(
    "csc",
    elemwise((x: number) => 1 / Math.sin(x)),
    1
  );
  register(
    "cot",
    elemwise((x: number) => 1 / Math.tan(x)),
    1
  );
  register(
    "sech",
    elemwise((x: number) => 1 / Math.cosh(x)),
    1
  );
  register(
    "csch",
    elemwise((x: number) => 1 / Math.sinh(x)),
    1
  );
  register(
    "coth",
    elemwise((x: number) => 1 / Math.tanh(x)),
    1
  );

  // Inverse hyperbolic reciprocal functions (complex-aware: e.g. asech(2) returns complex)
  register(
    "asech",
    complexElemwise(
      (x: number) => Math.acosh(1 / x),
      composeRecip(cAcosh),
      "ComplexNumber"
    ),
    1
  );
  register(
    "acsch",
    complexElemwise(
      (x: number) => Math.asinh(1 / x),
      composeRecip(cAsinh),
      "ComplexNumber"
    ),
    1
  );
  register(
    "acoth",
    complexElemwise(
      (x: number) => Math.atanh(1 / x),
      composeRecip(cAtanh),
      "ComplexNumber"
    ),
    1
  );

  // Degree-based reciprocal trig
  register(
    "secd",
    elemwise((x: number) => 1 / Math.cos(x * deg2rad)),
    1
  );
  register(
    "cscd",
    elemwise((x: number) => 1 / Math.sin(x * deg2rad)),
    1
  );
  register(
    "cotd",
    elemwise((x: number) => Math.cos(x * deg2rad) / Math.sin(x * deg2rad)),
    1
  );

  // Inverse reciprocal trig (complex-aware: e.g. asec(0.5) returns complex)
  register(
    "asec",
    complexElemwise(
      (x: number) => Math.acos(1 / x),
      composeRecip(cAcos),
      "ComplexNumber"
    ),
    1
  );
  register(
    "acsc",
    complexElemwise(
      (x: number) => Math.asin(1 / x),
      composeRecip(cAsin),
      "ComplexNumber"
    ),
    1
  );
  register(
    "acot",
    elemwise((x: number) => Math.atan(1 / x)),
    1
  );

  // Degree-based inverse reciprocal trig (complex-aware)
  const toDeg =
    (fn: (re: number, im: number) => ComplexResult) =>
    (re: number, im: number): ComplexResult => {
      const r = fn(re, im);
      return { re: r.re * rad2deg, im: r.im * rad2deg };
    };
  register(
    "asecd",
    complexElemwise(
      (x: number) => Math.acos(1 / x) * rad2deg,
      toDeg(composeRecip(cAcos)),
      "ComplexNumber"
    ),
    1
  );
  register(
    "acscd",
    complexElemwise(
      (x: number) => Math.asin(1 / x) * rad2deg,
      toDeg(composeRecip(cAsin)),
      "ComplexNumber"
    ),
    1
  );
  register(
    "acotd",
    elemwise((x: number) => Math.atan(1 / x) * rad2deg),
    1
  );

  // ── Elementary math ────────────────────────────────────────────────

  register("hypot", binaryScalar(Math.hypot), 2);

  register(
    "nthroot",
    binaryScalar((x, n) => {
      let result: number;
      if (x < 0 && n % 2 !== 0) {
        result = -Math.pow(-x, 1 / n);
      } else {
        result = Math.pow(x, 1 / n);
      }
      // Round to nearest integer if very close
      const rounded = Math.round(result);
      if (Math.abs(result - rounded) < 1e-10 * Math.max(1, Math.abs(rounded))) {
        if (Math.pow(rounded, n) === x) return rounded;
      }
      return result;
    }),
    2
  );

  register("log1p", elemwise(Math.log1p, { nativeJs: "Math.log1p" }), 1);
  register("expm1", elemwise(Math.expm1, { nativeJs: "Math.expm1" }), 1);
  register("erf", elemwise(erfScalar), 1);
  register("erfc", elemwise(erfcScalar), 1);
  register("erfinv", elemwise(erfinvScalar), 1);
  register("erfcinv", elemwise(erfcinvScalar), 1);
  register("erfcx", elemwise(erfcxScalar), 1);
  register(
    "pow2",
    elemwise((x: number) => Math.pow(2, x)),
    1
  );
  register(
    "nextpow2",
    elemwise((x: number) => {
      if (x <= 0) return 0;
      return Math.ceil(Math.log2(x));
    }),
    1
  );

  // ── Bessel functions ──────────────────────────────────────────────────

  // Bessel functions share the same registration pattern; only the
  // underlying function and scaling factor differ.
  const besselDefs: [
    string,
    (nu: number, z: number) => number,
    (z: number) => number,
  ][] = [
    ["besselj", besselj, z => Math.exp(-Math.abs(z))],
    ["bessely", bessely, z => Math.exp(-Math.abs(z))],
    ["besseli", besseli, z => Math.exp(-Math.abs(z))],
    ["besselk", besselk, z => Math.exp(z)],
  ];
  for (const [name, fn, scaleFn] of besselDefs) {
    register(
      name,
      builtinSingle(args => {
        if (args.length < 2 || args.length > 3)
          throw new RuntimeError(`${name} requires 2 or 3 arguments`);
        const scale = args.length === 3 ? toNumber(args[2]) : 0;
        return binaryApply(args[0], args[1], (nu, z) => {
          const val = fn(nu, z);
          return scale === 1 ? val * scaleFn(z) : val;
        });
      })
    );
  }

  // ── Airy functions ──────────────────────────────────────────────────

  const airyFns = [airyAi, airyAiPrime, airyBi, airyBiPrime];

  const airyComplexKeys = ["ai", "aip", "bi", "bip"] as const;

  function applyAiryElementwise(
    xArg: RuntimeValue,
    n: number,
    scaled: boolean
  ): RuntimeValue {
    // Complex scalar
    if (isRuntimeComplexNumber(xArg)) {
      const all = airyAllComplex(xArg.re, xArg.im);
      const r = all[airyComplexKeys[n]];
      return r.im === 0 ? RTV.num(r.re) : RTV.complex(r.re, r.im);
    }
    // Complex tensor
    if (isRuntimeTensor(xArg) && xArg.imag !== undefined) {
      const len = xArg.data.length;
      const resultRe = new FloatXArray(len);
      const resultIm = new FloatXArray(len);
      for (let i = 0; i < len; i++) {
        const all = airyAllComplex(xArg.data[i], xArg.imag[i]);
        const r = all[airyComplexKeys[n]];
        resultRe[i] = r.re;
        resultIm[i] = r.im;
      }
      const isReal = resultIm.every(x => x === 0);
      return RTV.tensor(resultRe, xArg.shape, isReal ? undefined : resultIm);
    }
    // Real tensor
    const fn = airyFns[n];
    if (isRuntimeTensor(xArg)) {
      const result = new FloatXArray(xArg.data.length);
      for (let i = 0; i < xArg.data.length; i++) {
        const val = fn(xArg.data[i]);
        result[i] = scaled ? scaleAiry(n, xArg.data[i], val) : val;
      }
      return RTV.tensor(result, xArg.shape);
    }
    const x = toNumber(xArg);
    const val = fn(x);
    return RTV.num(scaled ? scaleAiry(n, x, val) : val);
  }

  function scaleAiry(n: number, x: number, val: number): number {
    const zeta = (2 / 3) * Math.pow(Math.abs(x), 1.5);
    if (n <= 1) {
      // Ai or Ai': scale by exp(zeta) for x >= 0
      return x >= 0 ? val * Math.exp(zeta) : val;
    }
    // Bi or Bi': scale by exp(-zeta) for x >= 0
    return x >= 0 ? val * Math.exp(-zeta) : val;
  }

  register(
    "airy",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 3)
        throw new RuntimeError("airy requires 1 to 3 arguments");

      let nArg: RuntimeValue | undefined;
      let xArg: RuntimeValue;
      let scaled = false;

      if (args.length === 1) {
        // airy(x) — default n=0
        xArg = args[0];
      } else {
        // airy(n, x) or airy(n, x, 1)
        nArg = args[0];
        xArg = args[1];
        if (args.length === 3) {
          scaled = toNumber(args[2]) === 1;
        }
      }

      if (nArg === undefined) {
        // Simple case: airy(x) with n=0
        return applyAiryElementwise(xArg, 0, scaled);
      }

      // n must be a scalar 0, 1, 2, or 3
      const n = Math.round(toNumber(nArg));
      if (n < 0 || n > 3) throw new RuntimeError("K must be 0, 1, 2, or 3.");
      return applyAiryElementwise(xArg, n, scaled);
    })
  );

  // ── ellipj ──────────────────────────────────────────────────────────────

  /**
   * Compute Jacobi elliptic functions sn, cn, dn using the
   * arithmetic-geometric mean (AGM) algorithm.
   */
  function ellipjScalar(
    u: number,
    m: number,
    tol: number
  ): { sn: number; cn: number; dn: number } {
    // Edge cases
    if (m < 0 || m > 1) {
      return { sn: NaN, cn: NaN, dn: NaN };
    }
    if (m === 0) {
      return { sn: Math.sin(u), cn: Math.cos(u), dn: 1 };
    }
    if (m === 1) {
      const s = Math.tanh(u);
      const c = 1 / Math.cosh(u);
      return { sn: s, cn: c, dn: c };
    }

    // AGM iteration
    const a: number[] = [1];
    const b: number[] = [Math.sqrt(1 - m)];
    const c: number[] = [Math.sqrt(m)];
    let i = 0;
    while (Math.abs(c[i]) > tol) {
      a.push((a[i] + b[i]) / 2);
      b.push(Math.sqrt(a[i] * b[i]));
      c.push((a[i] - b[i]) / 2);
      i++;
      if (i > 100) break; // safety limit
    }
    const n = i;

    // Backward recurrence for phi
    let phi = Math.pow(2, n) * a[n] * u;
    for (let j = n; j >= 1; j--) {
      phi = (phi + Math.asin((c[j] / a[j]) * Math.sin(phi))) / 2;
    }

    const sn = Math.sin(phi);
    const cn = Math.cos(phi);
    const dn = Math.sqrt(1 - m * sn * sn);
    return { sn, cn, dn };
  }

  register(
    "ellipj",
    builtinSingle((args, nargout) => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("ellipj requires 2 or 3 arguments");

      const uArg = args[0];
      const mArg = args[1];
      const tol = args.length >= 3 ? toNumber(args[2]) : 2.220446049250313e-16; // eps

      const uIsT = isRuntimeTensor(uArg);
      const mIsT = isRuntimeTensor(mArg);

      // Helper to build result from parallel arrays
      const buildResult = (
        snArr: FloatXArrayType,
        cnArr: FloatXArrayType,
        dnArr: FloatXArrayType,
        shape: number[]
      ): RuntimeValue | RuntimeValue[] => {
        const effNargout = Math.max(nargout, 1);
        const results: RuntimeValue[] = [];
        if (effNargout >= 1) results.push(RTV.tensor(snArr, [...shape]));
        if (effNargout >= 2) results.push(RTV.tensor(cnArr, [...shape]));
        if (effNargout >= 3) results.push(RTV.tensor(dnArr, [...shape]));
        return results.length === 1 ? results[0] : results;
      };

      if (!uIsT && !mIsT) {
        // scalar-scalar
        const r = ellipjScalar(toNumber(uArg), toNumber(mArg), tol);
        const effNargout = Math.max(nargout, 1);
        if (effNargout === 1) return RTV.num(r.sn);
        const results: RuntimeValue[] = [RTV.num(r.sn)];
        if (effNargout >= 2) results.push(RTV.num(r.cn));
        if (effNargout >= 3) results.push(RTV.num(r.dn));
        return results;
      }

      if (uIsT && !mIsT) {
        // tensor-scalar
        const mv = toNumber(mArg);
        const len = uArg.data.length;
        const snArr = new FloatXArray(len);
        const cnArr = new FloatXArray(len);
        const dnArr = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const r = ellipjScalar(uArg.data[i], mv, tol);
          snArr[i] = r.sn;
          cnArr[i] = r.cn;
          dnArr[i] = r.dn;
        }
        return buildResult(snArr, cnArr, dnArr, uArg.shape);
      }

      if (!uIsT && mIsT) {
        // scalar-tensor
        const uv = toNumber(uArg);
        const mT = mArg as RuntimeTensor;
        const len = mT.data.length;
        const snArr = new FloatXArray(len);
        const cnArr = new FloatXArray(len);
        const dnArr = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const r = ellipjScalar(uv, mT.data[i], tol);
          snArr[i] = r.sn;
          cnArr[i] = r.cn;
          dnArr[i] = r.dn;
        }
        return buildResult(snArr, cnArr, dnArr, mT.shape);
      }

      // tensor-tensor: same size
      const uT = uArg as RuntimeTensor;
      const mT = mArg as RuntimeTensor;
      const len = uT.data.length;
      const snArr = new FloatXArray(len);
      const cnArr = new FloatXArray(len);
      const dnArr = new FloatXArray(len);
      for (let i = 0; i < len; i++) {
        const r = ellipjScalar(uT.data[i], mT.data[i], tol);
        snArr[i] = r.sn;
        cnArr[i] = r.cn;
        dnArr[i] = r.dn;
      }
      return buildResult(snArr, cnArr, dnArr, uT.shape);
    })
  );

  // ── legendre: Associated Legendre functions ──────────────────────────
  register(
    "legendre",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3) {
        throw new RuntimeError("legendre requires 2 or 3 arguments");
      }

      const n = Math.round(toNumber(args[0]));
      if (n < 0 || !isFinite(n)) {
        throw new RuntimeError("Degree n must be a non-negative integer");
      }

      // Parse normalization
      let normalization = "unnorm";
      if (args.length === 3) {
        const normArg = args[2];
        if (isRuntimeChar(normArg)) {
          normalization = normArg.value;
        } else if (typeof normArg === "string") {
          normalization = normArg;
        } else {
          throw new RuntimeError(
            "Third argument must be a normalization string"
          );
        }
        if (
          normalization !== "unnorm" &&
          normalization !== "sch" &&
          normalization !== "norm"
        ) {
          throw new RuntimeError(
            "Normalization must be 'unnorm', 'sch', or 'norm'"
          );
        }
      }

      // Get x values as flat array and determine output shape
      const xArg = args[1];
      let xValues: number[];
      let xShape: number[];
      if (isRuntimeNumber(xArg)) {
        xValues = [xArg as number];
        xShape = [1, 1];
      } else if (isRuntimeLogical(xArg)) {
        xValues = [xArg ? 1 : 0];
        xShape = [1, 1];
      } else if (isRuntimeTensor(xArg)) {
        const t = xArg as RuntimeTensor;
        xValues = Array.from(t.data);
        xShape = t.shape;
      } else {
        throw new RuntimeError("X must be a numeric value");
      }

      const numX = xValues.length;
      const numOrders = n + 1; // m = 0, 1, ..., n

      // Compute associated Legendre functions using recurrence
      // Output shape: (n+1) x [xShape...]
      const outSize = numOrders * numX;
      const result = new FloatXArray(outSize);

      for (let xi = 0; xi < numX; xi++) {
        const x = xValues[xi];
        // Compute P_m^n(x) for m = 0..n using recurrence in degree
        // We use upward recurrence on the degree l for each order m
        const pmn = legendreAllOrders(n, x);

        for (let m = 0; m <= n; m++) {
          let val = pmn[m];

          // Apply normalization
          if (normalization === "sch") {
            if (m > 0) {
              // S_n^m = (-1)^m * sqrt(2*(n-m)!/(n+m)!) * P_n^m
              const scale =
                Math.pow(-1, m) *
                Math.sqrt((2 * factorialVal(n - m)) / factorialVal(n + m));
              val = scale * val;
            }
          } else if (normalization === "norm") {
            // N_n^m = (-1)^m * sqrt((n+0.5)*(n-m)!/(n+m)!) * P_n^m
            const scale =
              Math.pow(-1, m) *
              Math.sqrt(
                ((n + 0.5) * factorialVal(n - m)) / factorialVal(n + m)
              );
            val = scale * val;
          }

          result[xi * numOrders + m] = val;
        }
      }

      // If x was scalar, output is column vector of size (n+1) x 1
      if (isRuntimeNumber(xArg) || isRuntimeLogical(xArg)) {
        return RTV.tensor(result, [numOrders, 1]);
      }
      // If x is a vector, output is (n+1) x length(X)
      // If x is a matrix/nd-array, output has one more dimension: (n+1) x size(X)
      if (xShape.length === 2 && (xShape[0] === 1 || xShape[1] === 1)) {
        // Vector case
        return RTV.tensor(result, [numOrders, numX]);
      }
      // General nd case
      return RTV.tensor(result, [numOrders, ...xShape]);
    })
  );
}

// Compute factorial for small integers
function factorialVal(n: number): number {
  if (n <= 1) return 1;
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

// Compute associated Legendre functions P_n^m(x) for all m = 0..n
// Uses the standard recurrence relations with Condon-Shortley phase
function legendreAllOrders(n: number, x: number): number[] {
  const result = new Array<number>(n + 1);

  if (n === 0) {
    result[0] = 1;
    return result;
  }

  // Start with sectoral harmonics P_m^m using:
  // P_0^0 = 1
  // P_m^m = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
  // Then use recurrence to get P_n^m from P_m^m and P_{m+1}^{m+1}

  const sqrtFactor = Math.sqrt(1 - x * x);

  // Compute P_l^m(x) for each m using recurrence on l
  for (let m = 0; m <= n; m++) {
    // P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    let pmm = 1.0;
    if (m > 0) {
      // (2m-1)!! = 1*3*5*...*(2m-1)
      let dblFact = 1.0;
      for (let i = 1; i <= m; i++) {
        dblFact *= 2 * i - 1;
      }
      pmm = Math.pow(-1, m) * dblFact * Math.pow(sqrtFactor, m);
    }

    if (m === n) {
      result[m] = pmm;
      continue;
    }

    // P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    const pmm1 = x * (2 * m + 1) * pmm;

    if (m + 1 === n) {
      result[m] = pmm1;
      continue;
    }

    // Recurrence: (l-m)*P_l^m = x*(2l-1)*P_{l-1}^m - (l+m-1)*P_{l-2}^m
    let pPrev2 = pmm;
    let pPrev1 = pmm1;
    let pCurr = 0;
    for (let l = m + 2; l <= n; l++) {
      pCurr = (x * (2 * l - 1) * pPrev1 - (l + m - 1) * pPrev2) / (l - m);
      pPrev2 = pPrev1;
      pPrev1 = pCurr;
    }
    result[m] = pCurr;
  }

  return result;
}

/**
 * Special math builtins: airy, ellipj, legendre, besselj, bessely, besseli, besselk.
 */

import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import {
  airyAi,
  airyAiPrime,
  airyBi,
  airyBiPrime,
  airyAllComplex,
  besselj,
  bessely,
  besseli,
  besselk,
} from "../../helpers/bessel.js";
import { tensorOps, OpBessel } from "../../ops/index.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin } from "./builtin.js";
import { builtinError, realScalarArg } from "./args.js";
import { floatArg, hasImag, realFloatArg } from "./math-args.js";

// ── Helpers ──────────────────────────────────────────────────────────────

type Real = number | RuntimeTensor;

/** Unwrap a 1-element tensor to a number. */
function asScalar(a: Real): Real {
  return isRuntimeTensor(a) && a.data.length === 1 ? a.data[0] : a;
}

/** Apply fn element-wise to (a, b): scalar/array pairs, or two arrays of
 *  the same size (`mismatch` is raised otherwise). */
function pairApply<T>(
  a0: Real,
  b0: Real,
  fn: (x: number, y: number) => T,
  mismatch: () => RuntimeError
): { values: T[]; shape: number[] } {
  const a = asScalar(a0);
  const b = asScalar(b0);
  if (typeof a === "number" && typeof b === "number")
    return { values: [fn(a, b)], shape: [1, 1] };
  if (typeof a === "number") {
    const t = b as RuntimeTensor;
    return { values: Array.from(t.data, y => fn(a, y)), shape: t.shape };
  }
  if (typeof b === "number")
    return { values: Array.from(a.data, x => fn(x, b)), shape: a.shape };
  if (
    a.data.length !== b.data.length ||
    a.shape.length !== b.shape.length ||
    a.shape.some((d, k) => d !== b.shape[k])
  )
    throw mismatch();
  return {
    values: Array.from(a.data, (x, k) => fn(x, b.data[k])),
    shape: a.shape,
  };
}

function realResult(r: { values: number[]; shape: number[] }): RuntimeValue {
  if (r.values.length === 1 && r.shape.every(d => d === 1)) return r.values[0];
  return RTV.tensor(allocFloat64Array(r.values), r.shape);
}

function complexResult(r: {
  values: { re: number; im: number }[];
  shape: number[];
}): RuntimeValue {
  if (r.values.length === 1 && r.shape.every(d => d === 1)) {
    const v = r.values[0];
    return v.im === 0 ? v.re : RTV.complex(v.re, v.im);
  }
  const re = allocFloat64Array(r.values.length);
  const im = allocFloat64Array(r.values.length);
  let anyImag = false;
  r.values.forEach((v, k) => {
    re[k] = v.re;
    im[k] = v.im;
    if (v.im !== 0) anyImag = true;
  });
  return RTV.tensor(re, r.shape, anyImag ? im : undefined);
}

/** The NU and Z arguments of the Bessel functions: real NU, and real Z
 *  (complex Z is not implemented). */
function besselArgs(
  name: string,
  nuArg: RuntimeValue,
  zArg: RuntimeValue
): { nu: Real; z: Real } {
  const nu = realFloatArg(nuArg, () =>
    builtinError(`MATLAB:${name}:nonRealNU`, "NU must be real.")
  );
  const zf = floatArg(zArg);
  if (hasImag(zf))
    throw new RuntimeError(`${name}: complex Z is not supported`);
  const z = realFloatArg(zf, () => new RuntimeError(""));
  return { nu, z };
}

const sizeMismatch = (name: string) => () =>
  builtinError(
    `MATLAB:${name}:NUAndZSizeMismatch`,
    "NU and Z must be the same size or one must be a scalar."
  );

// ── Bessel functions ─────────────────────────────────────────────────────

const besselDefs: [
  string,
  (nu: number, z: number) => number,
  (z: number) => number,
  number /* OpBessel op code for the tensor fast path */,
][] = [
  // J and Y are scaled by exp(-abs(imag(z))), which is 1 for real z.
  ["besselj", besselj, () => 1, OpBessel.J],
  ["bessely", bessely, () => 1, OpBessel.Y],
  ["besseli", besseli, z => Math.exp(-Math.abs(z)), OpBessel.I],
  ["besselk", besselk, z => Math.exp(z), OpBessel.K],
];

for (const [name, fn, scaleFn, opCode] of besselDefs) {
  // J and Y need no scaling for real z.
  const scalesReal = name === "besseli" || name === "besselk";
  builtin({
    name,
    nargin: [2, 3],
    apply: args => {
      const { nu, z } = besselArgs(name, args[0], args[1]);
      const scale = args.length === 3 ? realScalarArg(args[2]) : 0;
      const scaled = scale === 1 && scalesReal;
      const nuS = asScalar(nu);
      // Fast tensor path: scalar nu, real tensor z.
      if (typeof nuS === "number" && isRuntimeTensor(z) && z.data.length > 1) {
        const out = allocFloat64Array(z.data.length);
        tensorOps.besselReal(
          opCode,
          nuS,
          z.data.length,
          z.data as Float64Array,
          scaled ? 1 : 0,
          out as Float64Array
        );
        return RTV.tensor(out, z.shape);
      }
      return realResult(
        pairApply(
          nu,
          z,
          (n, x) => {
            const val = fn(n, x);
            return scaled ? val * scaleFn(x) : val;
          },
          sizeMismatch(name)
        )
      );
    },
  });
}

// ── besselh: Hankel function (Bessel function of the third kind) ─────────

function besselhScalar(
  nu: number,
  k: number,
  z: number,
  scale: boolean
): { re: number; im: number } {
  const J = besselj(nu, z);
  const Y = bessely(nu, z);
  let re = J;
  let im = k === 1 ? Y : -Y;
  if (scale) {
    // K=1: multiply by e^(-iz) = cos(z) - i*sin(z)
    // K=2: multiply by e^(+iz) = cos(z) + i*sin(z)
    const c = Math.cos(z);
    const s = Math.sin(z);
    if (k === 1) {
      const newRe = re * c + im * s;
      const newIm = im * c - re * s;
      re = newRe;
      im = newIm;
    } else {
      const newRe = re * c - im * s;
      const newIm = re * s + im * c;
      re = newRe;
      im = newIm;
    }
  }
  return { re, im };
}

builtin({
  name: "besselh",
  // besselh(nu, z), besselh(nu, K, z), besselh(nu, K, z, scale)
  nargin: [2, 4],
  apply: args => {
    let k = 1;
    let zArg: RuntimeValue;
    let scale = false;
    if (args.length === 2) {
      zArg = args[1];
    } else {
      const kErr = () =>
        builtinError("MATLAB:besselh:KNotOneOrTwo", "K must be 1 or 2.");
      try {
        k = realScalarArg(args[1]);
      } catch {
        throw kErr();
      }
      if (k !== 1 && k !== 2) throw kErr();
      zArg = args[2];
      if (args.length === 4) scale = realScalarArg(args[3]) === 1;
    }
    const { nu, z } = besselArgs("besselh", args[0], zArg);
    const nuS = asScalar(nu);
    // Fast tensor path: scalar nu, real tensor z.
    if (typeof nuS === "number" && isRuntimeTensor(z) && z.data.length > 1) {
      const len = z.data.length;
      const outRe = allocFloat64Array(len);
      const outIm = allocFloat64Array(len);
      tensorOps.besselH(
        k,
        nuS,
        len,
        z.data as Float64Array,
        scale ? 1 : 0,
        outRe as Float64Array,
        outIm as Float64Array
      );
      return RTV.tensor(outRe, z.shape, outIm);
    }
    return complexResult(
      pairApply(
        nu,
        z,
        (n, x) => besselhScalar(n, k, x, scale),
        sizeMismatch("besselh")
      )
    );
  },
});

// ── Airy functions ───────────────────────────────────────────────────────

const airyFns = [airyAi, airyAiPrime, airyBi, airyBiPrime];
const airyComplexKeys = ["ai", "aip", "bi", "bip"] as const;

function scaleAiry(n: number, x: number, val: number): number {
  const zeta = (2 / 3) * Math.pow(Math.abs(x), 1.5);
  if (n <= 1) return x >= 0 ? val * Math.exp(zeta) : val;
  return x >= 0 ? val * Math.exp(-zeta) : val;
}

/** (2/3) z^(3/2) on the principal branch. */
function airyZeta(re: number, im: number): { re: number; im: number } {
  const r = Math.pow(Math.hypot(re, im), 1.5);
  const th = 1.5 * Math.atan2(im, re);
  return { re: (2 / 3) * r * Math.cos(th), im: (2 / 3) * r * Math.sin(th) };
}

/** Airy function k of complex z, with MATLAB's scaling: Ai, Ai' times
 *  exp(zeta) and Bi, Bi' times exp(-abs(real(zeta))). */
function airyComplex(
  k: number,
  re: number,
  im: number,
  scaled: boolean
): { re: number; im: number } {
  const r = airyAllComplex(re, im)[airyComplexKeys[k]];
  if (!scaled) return r;
  const z = airyZeta(re, im);
  if (k >= 2) {
    const f = Math.exp(-Math.abs(z.re));
    return { re: r.re * f, im: r.im * f };
  }
  const m = Math.exp(z.re);
  const c = m * Math.cos(z.im);
  const s = m * Math.sin(z.im);
  return { re: r.re * c - r.im * s, im: r.re * s + r.im * c };
}

function applyAiryElementwise(
  xArg: RuntimeValue,
  n: number,
  scaled: boolean
): RuntimeValue {
  const x = floatArg(xArg);
  if (isRuntimeComplexNumber(x) && x.im !== 0) {
    const r = airyComplex(n, x.re, x.im, scaled);
    return r.im === 0 ? RTV.num(r.re) : RTV.complex(r.re, r.im);
  }
  if (hasImag(x)) {
    const t = x as RuntimeTensor;
    const im = t.imag!;
    const len = t.data.length;
    const resultRe = allocFloat64Array(len);
    const resultIm = allocFloat64Array(len);
    for (let i = 0; i < len; i++) {
      const r = airyComplex(n, t.data[i], im[i], scaled);
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
    const isReal = resultIm.every(v => v === 0);
    return RTV.tensor(resultRe, t.shape, isReal ? undefined : resultIm);
  }
  const xr = realFloatArg(x, () => new RuntimeError("airy: unexpected input"));
  const fn = airyFns[n];
  if (isRuntimeTensor(xr)) {
    const result = allocFloat64Array(xr.data.length);
    for (let i = 0; i < xr.data.length; i++) {
      const val = fn(xr.data[i]);
      result[i] = scaled ? scaleAiry(n, xr.data[i], val) : val;
    }
    return RTV.tensor(result, xr.shape);
  }
  const val = fn(xr);
  return RTV.num(scaled ? scaleAiry(n, xr, val) : val);
}

builtin({
  name: "airy",
  // airy(z), airy(k, z), airy(k, z, scale)
  nargin: [1, 3],
  apply: args => {
    if (args.length === 1) return applyAiryElementwise(args[0], 0, false);
    const kErr = () =>
      builtinError("MATLAB:airy:InvalidOrder", "K must be 0, 1, 2, or 3.");
    let n: number;
    try {
      n = realScalarArg(args[0]);
    } catch {
      throw kErr();
    }
    if (!(n === 0 || n === 1 || n === 2 || n === 3)) throw kErr();
    const scaled = args.length === 3 && realScalarArg(args[2]) === 1;
    return applyAiryElementwise(args[1], n, scaled);
  },
});

// ── ellipj ───────────────────────────────────────────────────────────────

function ellipjScalar(
  u: number,
  m: number,
  tol: number
): { sn: number; cn: number; dn: number } {
  if (m < 0 || m > 1) return { sn: NaN, cn: NaN, dn: NaN };
  if (m === 0) return { sn: Math.sin(u), cn: Math.cos(u), dn: 1 };
  if (m === 1) {
    const s = Math.tanh(u);
    const c = 1 / Math.cosh(u);
    return { sn: s, cn: c, dn: c };
  }
  const a: number[] = [1];
  const b: number[] = [Math.sqrt(1 - m)];
  const cv: number[] = [Math.sqrt(m)];
  let i = 0;
  while (Math.abs(cv[i]) > tol) {
    a.push((a[i] + b[i]) / 2);
    b.push(Math.sqrt(a[i] * b[i]));
    cv.push((a[i] - b[i]) / 2);
    i++;
    if (i > 100) break;
  }
  const n = i;
  let phi = Math.pow(2, n) * a[n] * u;
  for (let j = n; j >= 1; j--) {
    phi = (phi + Math.asin((cv[j] / a[j]) * Math.sin(phi))) / 2;
  }
  const sn = Math.sin(phi);
  const cn = Math.cos(phi);
  const dn = Math.sqrt(1 - m * sn * sn);
  return { sn, cn, dn };
}

builtin({
  name: "ellipj",
  // [sn, cn, dn] = ellipj(u, m), ellipj(u, m, tol)
  nargin: [0, 3],
  nargout: 3,
  mfile: true,
  apply: (args, nargout) => {
    if (args.length < 2)
      throw builtinError(
        "MATLAB:ellipj:NotEnoughInputs",
        "Not enough input arguments."
      );
    const complexErr = () =>
      builtinError(
        "MATLAB:ellipj:ComplexInputs",
        "Input arguments must be real."
      );
    const u = realFloatArg(args[0], complexErr);
    const m = realFloatArg(args[1], complexErr);
    const tol =
      args.length >= 3 ? realScalarArg(args[2]) : 2.220446049250313e-16;
    const r = pairApply(
      u,
      m,
      (uv, mv) => ellipjScalar(uv, mv, tol),
      () => new RuntimeError("ellipj: U and M must be the same size")
    );
    const pick = (key: "sn" | "cn" | "dn") =>
      realResult({ values: r.values.map(v => v[key]), shape: r.shape });
    if (nargout <= 1) return pick("sn");
    const results: RuntimeValue[] = [pick("sn"), pick("cn")];
    if (nargout >= 3) results.push(pick("dn"));
    return results;
  },
});

// ── legendre ─────────────────────────────────────────────────────────────

function factorialVal(n: number): number {
  if (n <= 1) return 1;
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

function legendreAllOrders(n: number, x: number): number[] {
  const result = new Array<number>(n + 1);
  if (n === 0) {
    result[0] = 1;
    return result;
  }
  const sqrtFactor = Math.sqrt(1 - x * x);

  for (let m = 0; m <= n; m++) {
    let pmm = 1.0;
    if (m > 0) {
      let dblFact = 1.0;
      for (let i = 1; i <= m; i++) dblFact *= 2 * i - 1;
      pmm = Math.pow(-1, m) * dblFact * Math.pow(sqrtFactor, m);
    }
    if (m === n) {
      result[m] = pmm;
      continue;
    }
    const pmm1 = x * (2 * m + 1) * pmm;
    if (m + 1 === n) {
      result[m] = pmm1;
      continue;
    }
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

builtin({
  name: "legendre",
  // legendre(n, x), legendre(n, x, normalization)
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    let n: number;
    try {
      n = Math.round(realScalarArg(args[0]));
    } catch {
      throw builtinError(
        "MATLAB:legendre:InvalidN",
        "N must be a positive scalar integer."
      );
    }
    if (n < 0 || !isFinite(n))
      throw new RuntimeError("Degree n must be a non-negative integer");

    let normalization = "unnorm";
    if (args.length === 3) {
      const normArg = args[2];
      if (isRuntimeChar(normArg)) {
        normalization = normArg.value;
      } else if (typeof normArg === "string") {
        normalization = normArg;
      } else {
        throw new RuntimeError("Third argument must be a normalization string");
      }
      if (
        normalization !== "unnorm" &&
        normalization !== "sch" &&
        normalization !== "norm"
      )
        throw new RuntimeError(
          "Normalization must be 'unnorm', 'sch', or 'norm'"
        );
    }

    const xErr = () =>
      builtinError(
        "MATLAB:legendre:InvalidX",
        "X must be real and in the range (-1,1)."
      );
    if (isRuntimeChar(args[1])) throw xErr();
    let xr: number | RuntimeTensor;
    try {
      xr = realFloatArg(args[1], xErr);
    } catch {
      throw xErr();
    }
    const xValues = typeof xr === "number" ? [xr] : Array.from(xr.data);
    const xShape = typeof xr === "number" ? [1, 1] : xr.shape;

    const numX = xValues.length;
    const numOrders = n + 1;
    const outSize = numOrders * numX;
    const result = allocFloat64Array(outSize);

    for (let xi = 0; xi < numX; xi++) {
      const x = xValues[xi];
      const pmn = legendreAllOrders(n, x);
      for (let m = 0; m <= n; m++) {
        let val = pmn[m];
        if (normalization === "sch") {
          if (m > 0) {
            val *=
              Math.pow(-1, m) *
              Math.sqrt((2 * factorialVal(n - m)) / factorialVal(n + m));
          }
        } else if (normalization === "norm") {
          val *=
            Math.pow(-1, m) *
            Math.sqrt(((n + 0.5) * factorialVal(n - m)) / factorialVal(n + m));
        }
        result[xi * numOrders + m] = val;
      }
    }

    if (typeof xr === "number") return RTV.tensor(result, [numOrders, 1]);
    if (xShape.length === 2 && (xShape[0] === 1 || xShape[1] === 1))
      return RTV.tensor(result, [numOrders, numX]);
    return RTV.tensor(result, [numOrders, ...xShape]);
  },
});

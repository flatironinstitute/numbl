/**
 * Numerical builtins: eps, conv, conv2, deconv, polyval, polyfit, trapz,
 * cumtrapz, gradient, accumarray, interp1, roots, poly, cov, corrcoef,
 * histc, discretize, mkpp, ppval, fnint/ppint.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
  tensorSize2D,
} from "../../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeFunction,
  isRuntimeStruct,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { builtin } from "./builtin.js";
import {
  builtinError,
  numericArg,
  realScalarArg,
  type NumericArg,
} from "./args.js";
import { floatArg, hasImag, realDenseArg } from "./math-args.js";
import { unaryRealResult } from "./types.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { linsolveLapack } from "../../helpers/linsolve.js";
import { allocFloat64Array, withScratch } from "../../runtime/alloc.js";

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);
const textOf = (v: RuntimeValue) =>
  (isRuntimeChar(v) ? v.value : String(v)).toLowerCase();

const complexUnsupported = (name: string) => () =>
  new RuntimeError(`${name}: complex input is not supported`);

/** The values of a real numeric or logical argument. Char is accepted
 *  only with `allowChar` (as its character codes), and complex input
 *  raises an error instead of dropping the imaginary part. */
function toFloatArray(
  v: RuntimeValue,
  name: string,
  allowChar = false
): Float64Array {
  if (isRuntimeChar(v) && !allowChar)
    throw new RuntimeError(`${name}: char input is not supported`);
  let a: number | RuntimeTensor;
  try {
    a = realDenseArg(v, complexUnsupported(name));
  } catch (e) {
    if (hasImagValue(v)) throw e;
    throw new RuntimeError(`${name}: expected a numeric argument`);
  }
  return typeof a === "number"
    ? allocFloat64Array([a])
    : allocFloat64Array(a.data);
}

function hasImagValue(v: RuntimeValue): boolean {
  return (
    (isRuntimeComplexNumber(v) && v.im !== 0) ||
    (isRuntimeTensor(v) && !!v.imag && v.imag.some(x => x !== 0))
  );
}

/** The values of a numeric or logical (not char) argument, with its
 *  imaginary part (undefined when real). */
function toComplexArrays(
  v: RuntimeValue,
  name: string
): { re: Float64Array; im: Float64Array | undefined } {
  if (isRuntimeChar(v))
    throw new RuntimeError(`${name}: char input is not supported`);
  let a: NumericArg;
  try {
    a = numericArg(v);
  } catch {
    throw new RuntimeError(`${name}: expected a numeric argument`);
  }
  if (isRuntimeSparseMatrix(a)) a = sparseToDense(a);
  if (typeof a === "number")
    return { re: allocFloat64Array([a]), im: undefined };
  if (isRuntimeComplexNumber(a))
    return {
      re: allocFloat64Array([a.re]),
      im: a.im !== 0 ? allocFloat64Array([a.im]) : undefined,
    };
  return {
    re: allocFloat64Array(a.data),
    im: hasImag(a) ? allocFloat64Array(a.imag!) : undefined,
  };
}

/** A tensor from real and (optional) imaginary parts. */
function complexTensor(
  re: Float64Array,
  im: Float64Array | undefined,
  shape: number[]
): RuntimeValue {
  const anyImag = !!im && im.some(x => x !== 0);
  return RTV.tensor(re, shape, anyImag ? im : undefined);
}

function isColumnVector(v: RuntimeValue): boolean {
  if (isRuntimeTensor(v))
    return v.shape[0] > 1 && (v.shape.length < 2 || v.shape[1] === 1);
  return false;
}

// ── eps ──────────────────────────────────────────────────────────────

/** Compute eps(x): distance from abs(x) to the next larger double. */
function epsOfScalar(x: number): number {
  x = Math.abs(x);
  if (!isFinite(x) || isNaN(x)) return NaN;
  if (x === 0) return Number.MIN_VALUE; // smallest positive subnormal ≈ 5e-324
  const buf = allocFloat64Array(1);
  const view = new DataView(buf.buffer);
  buf[0] = x;
  const bits = view.getBigUint64(0, true);
  view.setBigUint64(0, bits + 1n, true);
  return buf[0] - x;
}

const DOUBLE_EPS = 2.220446049250313e-16; // 2^-52
const SINGLE_EPS = 1.1920928955078125e-7; // 2^-23

const epsClassError = () =>
  builtinError(
    "MATLAB:eps:invalidClassName",
    'Class name must be a class that supports EPS, such as "double" or "single".'
  );

builtin({
  name: "eps",
  // eps, eps(x), eps(classname), eps('like', p)
  nargin: [0, 2],
  apply: args => {
    if (args.length === 0) return DOUBLE_EPS;
    const v = args[0];
    if (args.length === 2) {
      if (!(isText(v) && textOf(v) === "like"))
        throw builtinError(
          "MATLAB:eps:mustBeLike",
          'With two input arguments, the first argument must be "like".'
        );
      return DOUBLE_EPS;
    }
    if (isText(v)) {
      const s = textOf(v);
      if (s === "double") return DOUBLE_EPS;
      if (s === "single") return SINGLE_EPS;
      throw epsClassError();
    }
    let x: NumericArg;
    try {
      x = floatArg(v);
    } catch {
      throw epsClassError();
    }
    // eps of a complex value is eps of its magnitude.
    return unaryRealResult(
      epsOfScalar,
      (re, im) => epsOfScalar(Math.hypot(re, im)),
      "eps"
    )([x]);
  },
});

// ── conv ─────────────────────────────────────────────────────────────

builtin({
  name: "conv",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const a = toComplexArrays(args[0], "conv");
    const b = toComplexArrays(args[1], "conv");
    const m = a.re.length;
    const n = b.re.length;
    const colVec = isColumnVector(args[0]);

    const fullLen = m + n - 1;
    const full = allocFloat64Array(fullLen);
    const complex = !!a.im || !!b.im;
    const fullIm = complex ? allocFloat64Array(fullLen) : undefined;
    for (let i = 0; i < m; i++) {
      const ar = a.re[i];
      const ai = a.im ? a.im[i] : 0;
      for (let j = 0; j < n; j++) {
        const br = b.re[j];
        const bi = b.im ? b.im[j] : 0;
        full[i + j] += ar * br - ai * bi;
        if (fullIm) fullIm[i + j] += ar * bi + ai * br;
      }
    }

    let shape: string = "full";
    if (args.length === 3) {
      const opt = args[2];
      if (isRuntimeChar(opt)) shape = opt.value.toLowerCase();
      else if (isRuntimeString(opt)) shape = (opt as string).toLowerCase();
      else throw new RuntimeError("conv: third argument must be a string");
    }

    let start: number;
    let len: number;
    if (shape === "full") {
      start = 0;
      len = fullLen;
    } else if (shape === "same") {
      start = Math.floor(n / 2);
      len = m;
    } else if (shape === "valid") {
      len = Math.max(m, n) - Math.min(m, n) + 1;
      start = Math.min(m, n) - 1;
    } else {
      throw new RuntimeError(`conv: unknown shape '${shape}'`);
    }
    const result = full.slice(start, start + len);
    const resultIm = fullIm ? fullIm.slice(start, start + len) : undefined;

    const outShape: [number, number] = colVec ? [len, 1] : [1, len];
    return complexTensor(result, resultIm, outShape);
  },
});

// ── conv2 ────────────────────────────────────────────────────────────

/** A 2-D matrix view used by conv2: column-major real (+ optional imag). */
interface ConvMat {
  re: Float64Array;
  im?: Float64Array;
  m: number; // rows
  n: number; // cols
}

/** Coerce a conv2 argument (scalar / logical / tensor) to a 2-D matrix. */
function conv2Matrix(v: RuntimeValue, who: string): ConvMat {
  if (isRuntimeComplexNumber(v))
    return {
      re: allocFloat64Array([v.re]),
      im: allocFloat64Array([v.im]),
      m: 1,
      n: 1,
    };
  if (isRuntimeSparseMatrix(v)) return conv2Matrix(sparseToDense(v), who);
  if (isRuntimeNumber(v))
    return { re: allocFloat64Array([v as number]), m: 1, n: 1 };
  if (isRuntimeLogical(v))
    return { re: allocFloat64Array([v ? 1 : 0]), m: 1, n: 1 };
  if (isRuntimeTensor(v)) {
    const s = v.shape;
    for (let i = 2; i < s.length; i++) {
      if (s[i] !== 1)
        throw new RuntimeError("conv2: N-D arrays are not supported.");
    }
    const m = s.length >= 1 ? s[0] : 0;
    const n = s.length >= 2 ? s[1] : 1;
    return { re: v.data, im: v.imag, m, n };
  }
  throw new RuntimeError(`conv2: ${who} must be numeric`);
}

/** Outer product u(:) * v(:).' — the kernel of the separable conv2(u,v,A). */
function conv2Outer(u: ConvMat, v: ConvMat): ConvMat {
  const p = u.re.length; // numel(u)
  const q = v.re.length; // numel(v)
  const re = allocFloat64Array(p * q);
  const complex = !!u.im || !!v.im;
  const im = complex ? allocFloat64Array(p * q) : undefined;
  for (let j = 0; j < q; j++) {
    const vr = v.re[j];
    const vi = v.im ? v.im[j] : 0;
    for (let i = 0; i < p; i++) {
      const ur = u.re[i];
      const ui = u.im ? u.im[i] : 0;
      const idx = i + j * p;
      re[idx] = ur * vr - ui * vi;
      if (im) im[idx] = ur * vi + ui * vr;
    }
  }
  return { re, im, m: p, n: q };
}

/** Full 2-D convolution C(i+bi, j+bj) += A(i,j) * B(bi,bj). */
function conv2Core(A: ConvMat, B: ConvMat): ConvMat {
  const { re: Are, im: Aim, m, n } = A;
  const { re: Bre, im: Bim, m: p, n: q } = B;
  const Cm = m + p - 1;
  const Cn = n + q - 1;
  if (Cm <= 0 || Cn <= 0 || m === 0 || n === 0 || p === 0 || q === 0) {
    return { re: allocFloat64Array(0), m: Math.max(Cm, 0), n: Math.max(Cn, 0) };
  }
  const complex = !!Aim || !!Bim;
  const Cre = allocFloat64Array(Cm * Cn);
  const Cim = complex ? allocFloat64Array(Cm * Cn) : undefined;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      const aIdx = i + j * m;
      const ar = Are[aIdx];
      const ai = Aim ? Aim[aIdx] : 0;
      if (ar === 0 && ai === 0) continue;
      for (let bj = 0; bj < q; bj++) {
        const colBase = (j + bj) * Cm;
        const bColBase = bj * p;
        for (let bi = 0; bi < p; bi++) {
          const br = Bre[bColBase + bi];
          const cIdx = colBase + (i + bi);
          if (Cim) {
            const bi2 = Bim ? Bim[bColBase + bi] : 0;
            Cre[cIdx] += ar * br - ai * bi2;
            Cim[cIdx] += ar * bi2 + ai * br;
          } else {
            Cre[cIdx] += ar * br;
          }
        }
      }
    }
  }
  return { re: Cre, im: Cim, m: Cm, n: Cn };
}

/** Crop a full convolution to 'full' | 'same' | 'valid' (relative to A). */
function conv2Crop(
  C: ConvMat,
  shape: string,
  aM: number,
  aN: number,
  bM: number,
  bN: number
): ConvMat {
  if (shape === "full") return C;
  let rows: number, cols: number, offR: number, offC: number;
  if (shape === "same") {
    rows = aM;
    cols = aN;
    offR = Math.floor(bM / 2);
    offC = Math.floor(bN / 2);
  } else if (shape === "valid") {
    rows = Math.max(aM - bM + 1, 0);
    cols = Math.max(aN - bN + 1, 0);
    offR = bM - 1;
    offC = bN - 1;
  } else {
    throw new RuntimeError(`conv2: unknown shape '${shape}'`);
  }
  const re = allocFloat64Array(rows * cols);
  const im = C.im ? allocFloat64Array(rows * cols) : undefined;
  for (let j = 0; j < cols; j++) {
    const srcCol = (j + offC) * C.m;
    const dstCol = j * rows;
    for (let i = 0; i < rows; i++) {
      const s = srcCol + (i + offR);
      re[dstCol + i] = C.re[s];
      if (im) im[dstCol + i] = C.im![s];
    }
  }
  return { re, im, m: rows, n: cols };
}

builtin({
  name: "conv2",
  nargin: [2, 4],
  apply: args => {
    // Optional trailing shape string.
    let shape = "full";
    let numeric = args;
    const last = args[args.length - 1];
    if (isRuntimeChar(last) || isRuntimeString(last)) {
      shape = (isRuntimeChar(last) ? last.value : (last as string))
        .trim()
        .toLowerCase();
      numeric = args.slice(0, -1);
    }

    let A: ConvMat;
    let B: ConvMat;
    if (numeric.length === 2) {
      // conv2(A, B)
      A = conv2Matrix(numeric[0], "A");
      B = conv2Matrix(numeric[1], "B");
    } else if (numeric.length === 3) {
      // conv2(u, v, A): separable, equivalent to conv2(A, u(:)*v(:).')
      const u = conv2Matrix(numeric[0], "u");
      const v = conv2Matrix(numeric[1], "v");
      A = conv2Matrix(numeric[2], "A");
      B = conv2Outer(u, v);
    } else {
      throw new RuntimeError(
        "conv2: expected conv2(A,B), conv2(u,v,A), optionally with a shape"
      );
    }

    const C = conv2Crop(conv2Core(A, B), shape, A.m, A.n, B.m, B.n);
    return RTV.tensor(C.re, [C.m, C.n], C.im);
  },
});

// ── deconv ───────────────────────────────────────────────────────────

builtin({
  name: "deconv",
  nargin: 2,
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    const b = toFloatArray(args[0], "deconv");
    const a = toFloatArray(args[1], "deconv");
    const nb = b.length;
    const na = a.length;
    const colVec = isColumnVector(args[0]);
    const mkShape = (n: number): [number, number] => (colVec ? [n, 1] : [1, n]);

    if (na === 0 || a[0] === 0)
      throw new RuntimeError("deconv: leading coefficient of divisor is zero");

    const nq = nb - na + 1;
    if (nq <= 0) {
      if (nargout <= 1) return RTV.tensor(allocFloat64Array([0]), mkShape(1));
      return [
        RTV.tensor(allocFloat64Array([0]), mkShape(1)),
        RTV.tensor(allocFloat64Array(b), mkShape(nb)),
      ];
    }

    const r = allocFloat64Array(b);
    const q = allocFloat64Array(nq);
    for (let i = 0; i < nq; i++) {
      q[i] = r[i] / a[0];
      for (let j = 0; j < na; j++) {
        r[i + j] -= q[i] * a[j];
      }
    }

    if (nargout <= 1) return RTV.tensor(q, mkShape(nq));
    return [RTV.tensor(q, mkShape(nq)), RTV.tensor(r, mkShape(nb))];
  },
});

// ── polyval ──────────────────────────────────────────────────────────

builtin({
  name: "polyval",
  // y = polyval(p, x), polyval(p, x, S, mu); [y, delta] is not supported
  nargin: [2, 4],
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    if (nargout > 1)
      throw new RuntimeError("polyval: the delta output is not supported");
    const p = toComplexArrays(args[0], "polyval");
    const np = p.re.length;
    let xv: NumericArg;
    try {
      xv = numericArg(args[1]);
    } catch {
      throw new RuntimeError("polyval: second argument must be numeric");
    }
    if (isRuntimeSparseMatrix(xv)) xv = sparseToDense(xv);
    let x: { re: Float64Array; im: Float64Array | undefined };
    let shape: number[];
    if (typeof xv === "number") {
      x = { re: allocFloat64Array([xv]), im: undefined };
      shape = [1, 1];
    } else if (isRuntimeComplexNumber(xv)) {
      x = { re: allocFloat64Array([xv.re]), im: allocFloat64Array([xv.im]) };
      shape = [1, 1];
    } else {
      x = {
        re: allocFloat64Array(xv.data),
        im: xv.imag ? allocFloat64Array(xv.imag) : undefined,
      };
      shape = [...xv.shape];
    }
    // Centering and scaling: x -> (x - mu(1)) / mu(2)
    if (args.length === 4) {
      const mu = toFloatArray(args[3], "polyval");
      if (mu.length !== 2)
        throw new RuntimeError("polyval: mu must have two elements");
      for (let i = 0; i < x.re.length; i++) {
        x.re[i] = (x.re[i] - mu[0]) / mu[1];
        if (x.im) x.im[i] = x.im[i] / mu[1];
      }
    }
    const n = x.re.length;
    const outRe = allocFloat64Array(n);
    const complex = !!p.im || !!x.im;
    const outIm = complex ? allocFloat64Array(n) : undefined;
    for (let k = 0; k < n; k++) {
      const xr = x.re[k];
      const xi = x.im ? x.im[k] : 0;
      let rr = np > 0 ? p.re[0] : 0;
      let ri = np > 0 && p.im ? p.im[0] : 0;
      for (let i = 1; i < np; i++) {
        const tr = rr * xr - ri * xi + p.re[i];
        const ti = rr * xi + ri * xr + (p.im ? p.im[i] : 0);
        rr = tr;
        ri = ti;
      }
      outRe[k] = rr;
      if (outIm) outIm[k] = ri;
    }
    if (typeof xv === "number" || isRuntimeComplexNumber(xv)) {
      const im = outIm ? outIm[0] : 0;
      return im !== 0 ? RTV.complex(outRe[0], im) : RTV.num(outRe[0]);
    }
    return complexTensor(outRe, outIm, shape);
  },
});

// ── polyfit ──────────────────────────────────────────────────────────

builtin({
  name: "polyfit",
  nargin: 3,
  nargout: 3,
  mfile: true,
  apply: (args, nargout) => {
    if (nargout > 1)
      throw new RuntimeError("polyfit: the S and mu outputs are not supported");
    const xArr = toFloatArray(args[0], "polyfit");
    const yArr = toFloatArray(args[1], "polyfit");
    const n = Math.round(realScalarArg(args[2]));
    const m = xArr.length;

    if (xArr.length !== yArr.length)
      throw new RuntimeError("polyfit: x and y must have the same length");
    if (n < 0) throw new RuntimeError("polyfit: degree must be non-negative");

    const ncols = n + 1;

    const V = allocFloat64Array(m * ncols);
    for (let j = 0; j < ncols; j++) {
      const power = n - j;
      for (let i = 0; i < m; i++) {
        V[j * m + i] = Math.pow(xArr[i], power);
      }
    }

    const B = allocFloat64Array(m);
    for (let i = 0; i < m; i++) B[i] = yArr[i];

    const X = linsolveLapack(V, m, ncols, B, 1);

    const result = allocFloat64Array(ncols);
    for (let i = 0; i < ncols; i++) result[i] = X[i];
    return RTV.tensor(result, [1, ncols]);
  },
});

// ── trapz ────────────────────────────────────────────────────────────

builtin({
  name: "trapz",
  nargin: [1, 3],
  mfile: true,
  apply: args => {
    if (args.length === 3)
      throw new RuntimeError("trapz: the dim argument is not supported");
    const yArg = args.length === 1 ? args[0] : args[1];
    if (isRuntimeTensor(yArg) && tensorSize2D(yArg).every(d => d > 1))
      throw new RuntimeError("trapz: matrix input is not supported");
    let x: Float64Array | null = null;
    let y: Float64Array;

    if (args.length === 1) {
      y = toFloatArray(args[0], "trapz", true);
    } else {
      x = toFloatArray(args[0], "trapz", true);
      y = toFloatArray(args[1], "trapz", true);
      if (x.length !== y.length)
        throw new RuntimeError("trapz: x and y must have the same length");
    }

    if (y.length <= 1) return 0;

    let sum = 0;
    for (let i = 0; i < y.length - 1; i++) {
      const dx = x ? x[i + 1] - x[i] : 1;
      sum += ((y[i] + y[i + 1]) * dx) / 2;
    }
    return sum;
  },
});

// ── cumtrapz ─────────────────────────────────────────────────────────

builtin({
  name: "cumtrapz",
  nargin: [1, 3],
  mfile: true,
  apply: args => {
    let x: Float64Array | null = null;
    let y: Float64Array;
    const yArg = args.length === 1 ? args[0] : args[1];

    if (args.length === 3)
      throw new RuntimeError("cumtrapz: the dim argument is not supported");
    if (args.length === 1) {
      y = toFloatArray(args[0], "cumtrapz");
    } else {
      x = toFloatArray(args[0], "cumtrapz");
      y = toFloatArray(args[1], "cumtrapz");
      if (x.length !== y.length)
        throw new RuntimeError("cumtrapz: x and y must have the same length");
    }

    const n = y.length;
    const result = allocFloat64Array(n);
    result[0] = 0;
    for (let i = 1; i < n; i++) {
      const dx = x ? x[i] - x[i - 1] : 1;
      result[i] = result[i - 1] + ((y[i - 1] + y[i]) * dx) / 2;
    }

    if (isColumnVector(yArg)) return RTV.tensor(result, [n, 1]);
    return RTV.tensor(result, [1, n]);
  },
});

// ── gradient ─────────────────────────────────────────────────────────

builtin({
  name: "gradient",
  nargin: [1, Infinity],
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    if (args.length > 2)
      throw new RuntimeError(
        "gradient: per-dimension spacing arguments are not supported"
      );
    const f = realDenseArg(args[0], complexUnsupported("gradient"));

    let hScalar: number | null = 1;
    let hVec: Float64Array | null = null;
    if (args.length === 2) {
      const spacing = realDenseArg(args[1], complexUnsupported("gradient"));
      if (typeof spacing === "number") {
        hScalar = spacing;
      } else if (spacing.data.length === 1) {
        hScalar = spacing.data[0];
      } else {
        hVec = allocFloat64Array(spacing.data);
        hScalar = null;
      }
    }

    const grad1dUniform = (
      data: Float64Array,
      len: number,
      spacing: number
    ): Float64Array => {
      const result = allocFloat64Array(len);
      if (len < 2) {
        result[0] = 0;
        return result;
      }
      result[0] = (data[1] - data[0]) / spacing;
      for (let i = 1; i < len - 1; i++) {
        result[i] = (data[i + 1] - data[i - 1]) / (2 * spacing);
      }
      result[len - 1] = (data[len - 1] - data[len - 2]) / spacing;
      return result;
    };

    const grad1dNonUniform = (
      data: Float64Array,
      len: number,
      coords: Float64Array
    ): Float64Array => {
      const result = allocFloat64Array(len);
      if (len < 2) {
        result[0] = 0;
        return result;
      }
      result[0] = (data[1] - data[0]) / (coords[1] - coords[0]);
      for (let i = 1; i < len - 1; i++) {
        result[i] =
          (data[i + 1] - data[i - 1]) / (coords[i + 1] - coords[i - 1]);
      }
      result[len - 1] =
        (data[len - 1] - data[len - 2]) / (coords[len - 1] - coords[len - 2]);
      return result;
    };

    if (!isRuntimeTensor(f)) {
      return nargout >= 2 ? [RTV.num(0), RTV.num(0)] : RTV.num(0);
    }

    const nRows = f.shape[0] || 1;
    const nCols = f.data.length / nRows;
    const isVec = nRows === 1 || nCols === 1;

    if (isVec) {
      const n = f.data.length;
      if (hVec) {
        const result = grad1dNonUniform(f.data, n, hVec);
        const shape = isColumnVector(f) ? [n, 1] : [1, n];
        return RTV.tensor(result, shape);
      }
      const result = grad1dUniform(f.data, n, hScalar!);
      const shape = isColumnVector(f) ? [n, 1] : [1, n];
      return RTV.tensor(result, shape);
    }

    const sp = hScalar ?? 1;
    const fxData = allocFloat64Array(nRows * nCols);
    for (let r = 0; r < nRows; r++) {
      const rowSlice = allocFloat64Array(nCols);
      for (let c = 0; c < nCols; c++) rowSlice[c] = f.data[c * nRows + r];
      const rowGrad = grad1dUniform(rowSlice, nCols, sp);
      for (let c = 0; c < nCols; c++) fxData[c * nRows + r] = rowGrad[c];
    }
    const fx = RTV.tensor(fxData, [nRows, nCols]);

    if (nargout >= 2) {
      const fyData = allocFloat64Array(nRows * nCols);
      for (let c = 0; c < nCols; c++) {
        const colSlice = allocFloat64Array(nRows);
        for (let r = 0; r < nRows; r++) colSlice[r] = f.data[c * nRows + r];
        const colGrad = grad1dUniform(colSlice, nRows, sp);
        for (let r = 0; r < nRows; r++) fyData[c * nRows + r] = colGrad[r];
      }
      return [fx, RTV.tensor(fyData, [nRows, nCols])];
    }
    return fx;
  },
});

// ── accumarray ───────────────────────────────────────────────────────

builtin({
  name: "accumarray",
  nargin: [2, 6],
  apply: args => {
    const subs = Array.from(toFloatArray(args[0], "accumarray"), (x: number) =>
      Math.round(x)
    );
    const nSubs = subs.length;

    let vals = toFloatArray(args[1], "accumarray");
    if (vals.length === 1 && nSubs !== 1) {
      // a scalar value is used for every subscript
      const v = vals[0];
      vals = allocFloat64Array(nSubs);
      vals.fill(v);
    }

    let maxIdx = 0;
    for (let i = 0; i < nSubs; i++) {
      if (subs[i] > maxIdx) maxIdx = subs[i];
      if (subs[i] < 1)
        throw new RuntimeError(
          "accumarray: subscript indices must be positive integers"
        );
    }

    if (args.length >= 3 && args[2] !== undefined) {
      const szArg = args[2];
      if (isRuntimeTensor(szArg) && szArg.data.length > 0) {
        const requestedMax = Math.round(szArg.data[0]);
        if (requestedMax > maxIdx) maxIdx = requestedMax;
      } else if (isRuntimeNumber(szArg)) {
        if ((szArg as number) > maxIdx) maxIdx = Math.round(szArg as number);
      }
    }

    let reduceFn: (group: number[]) => number;
    if (args.length >= 4 && isRuntimeFunction(args[3])) {
      const fnName = args[3].name;
      switch (fnName) {
        case "sum":
          reduceFn = g => g.reduce((a, b) => a + b, 0);
          break;
        case "max":
          reduceFn = g => Math.max(...g);
          break;
        case "min":
          reduceFn = g => Math.min(...g);
          break;
        case "mean":
          reduceFn = g => g.reduce((a, b) => a + b, 0) / g.length;
          break;
        case "prod":
          reduceFn = g => g.reduce((a, b) => a * b, 1);
          break;
        case "numel":
          reduceFn = g => g.length;
          break;
        default:
          throw new RuntimeError(
            `accumarray: unsupported function handle @${fnName}`
          );
      }
    } else {
      reduceFn = g => g.reduce((a, b) => a + b, 0);
    }

    const groups: number[][] = new Array(maxIdx);
    for (let i = 0; i < maxIdx; i++) groups[i] = [];
    for (let i = 0; i < nSubs; i++) {
      groups[subs[i] - 1].push(vals[i]);
    }

    const result = allocFloat64Array(maxIdx);
    for (let i = 0; i < maxIdx; i++) {
      if (groups[i].length > 0) result[i] = reduceFn(groups[i]);
    }

    return RTV.tensor(result, [maxIdx, 1]);
  },
});

// ── interp1 ──────────────────────────────────────────────────────────

builtin({
  name: "interp1",
  nargin: [2, 5],
  mfile: true,
  apply: args => {
    if (args.length < 3)
      throw new RuntimeError(
        "interp1: the interp1(v, xq) form is not supported; pass sample points"
      );
    const xArr = toFloatArray(args[0], "interp1");

    // interp1(x, y, method, 'pp') → piecewise-polynomial struct
    // (same layout as mkpp; evaluated with ppval)
    if (
      args.length === 4 &&
      isRuntimeChar(args[2]) &&
      isRuntimeChar(args[3]) &&
      args[3].value.toLowerCase() === "pp"
    ) {
      const method = args[2].value.toLowerCase();
      if (method !== "linear")
        throw new RuntimeError(
          `interp1: 'pp' output is only supported for 'linear' method (got '${method}')`
        );
      const yArr = toFloatArray(args[1], "interp1");
      const n = xArr.length;
      if (yArr.length !== n)
        throw new RuntimeError("interp1: x and y must have the same length");
      if (n < 2)
        throw new RuntimeError(
          "interp1: at least 2 sample points are required"
        );
      for (let i = 0; i + 1 < n; i++) {
        if (!(xArr[i] < xArr[i + 1]))
          throw new RuntimeError(
            "interp1: sample points must be unique and sorted for 'pp' output"
          );
      }
      const L = n - 1;
      const breaksOut = allocFloat64Array(n);
      for (let i = 0; i < n; i++) breaksOut[i] = xArr[i];
      // coefs row i (column-major storage): [slope_i, y_i]
      const coefs = allocFloat64Array(L * 2);
      for (let i = 0; i < L; i++) {
        coefs[i] = (yArr[i + 1] - yArr[i]) / (xArr[i + 1] - xArr[i]);
        coefs[L + i] = yArr[i];
      }
      const fields = new Map<string, RuntimeValue>();
      fields.set("form", "pp");
      fields.set("breaks", RTV.tensor(breaksOut, [1, n]));
      fields.set("coefs", RTV.tensor(coefs, [L, 2]));
      fields.set("pieces", L);
      fields.set("order", 2);
      fields.set("dim", 1);
      return RTV.struct(fields);
    }

    const xqArg = args[2];
    const nn = xArr.length;

    // y may be a vector (single data set) or a matrix whose columns are
    // each interpolated against x (MATLAB: size(y,1) == length(x)).
    if (hasImagValue(args[1])) throw complexUnsupported("interp1")();
    const yVal = args[1];
    let yCols: number; // number of data sets (matrix columns)
    let yColData: (col: number) => Float64Array;
    let yIsMatrix = false;
    if (isRuntimeTensor(yVal)) {
      const rows = yVal.shape[0] ?? 1;
      const cols = yVal.shape.length >= 2 ? yVal.shape[1] : 1;
      const isVector = rows === 1 || cols === 1;
      if (isVector) {
        const flat = yVal.data;
        yCols = 1;
        yColData = () => flat;
      } else {
        yIsMatrix = true;
        yCols = cols;
        // Column-major: column c occupies data[c*rows .. c*rows+rows-1].
        yColData = c =>
          yVal.data.subarray(c * rows, c * rows + rows) as Float64Array;
      }
    } else {
      const flat = toFloatArray(yVal, "interp1");
      yCols = 1;
      yColData = () => flat;
    }

    // Each data set must have one sample per x.
    if (yColData(0).length !== nn)
      throw new RuntimeError("interp1: x and y must have the same length");

    let method = "linear";
    let doExtrap = false;
    let fillValue = NaN;
    for (let i = 3; i < args.length; i++) {
      const a = args[i];
      if (isRuntimeChar(a)) {
        const s = a.value.toLowerCase();
        if (s === "extrap") doExtrap = true;
        else method = s;
      } else if (isRuntimeString(a)) {
        const s = (a as string).toLowerCase();
        if (s === "extrap") doExtrap = true;
        else method = s;
      } else if (isRuntimeNumber(a) || isRuntimeTensor(a)) {
        // interp1(x, y, xq, method, fillValue): numeric extrapolation
        // value used outside the sample range
        fillValue = toNumber(a);
      }
    }

    const interpOne = (yArr: Float64Array, xq: number): number => {
      if (xq < xArr[0] || xq > xArr[nn - 1]) {
        if (!doExtrap) return fillValue;
        if (method === "linear") {
          if (xq < xArr[0]) {
            const slope = (yArr[1] - yArr[0]) / (xArr[1] - xArr[0]);
            return yArr[0] + slope * (xq - xArr[0]);
          } else {
            const slope =
              (yArr[nn - 1] - yArr[nn - 2]) / (xArr[nn - 1] - xArr[nn - 2]);
            return yArr[nn - 1] + slope * (xq - xArr[nn - 1]);
          }
        }
      }

      if (xq <= xArr[0]) return yArr[0];
      if (xq >= xArr[nn - 1]) return yArr[nn - 1];

      let lo = 0,
        hi = nn - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xArr[mid] <= xq) lo = mid;
        else hi = mid;
      }

      if (method === "nearest") {
        return xq - xArr[lo] < xArr[hi] - xq ? yArr[lo] : yArr[hi];
      }

      const t = (xq - xArr[lo]) / (xArr[hi] - xArr[lo]);
      return yArr[lo] + t * (yArr[hi] - yArr[lo]);
    };

    if (hasImagValue(xqArg))
      throw builtinError(
        "MATLAB:griddedInterpolant:ComplexDataPointErrId",
        "Input coordinates must be real."
      );
    // Scalar query against a single-column y → scalar result.
    if (isRuntimeNumber(xqArg) && !yIsMatrix)
      return RTV.num(interpOne(yColData(0), xqArg as number));

    const xqData = isRuntimeNumber(xqArg)
      ? allocFloat64Array([xqArg as number])
      : isRuntimeTensor(xqArg)
        ? xqArg.data
        : null;
    if (!xqData)
      throw new RuntimeError("interp1: query points must be numeric");
    const nq = xqData.length;

    if (yIsMatrix) {
      // Result is nq-by-(number of columns), column-major.
      const result = allocFloat64Array(nq * yCols);
      for (let c = 0; c < yCols; c++) {
        const col = yColData(c);
        for (let i = 0; i < nq; i++)
          result[c * nq + i] = interpOne(col, xqData[i]);
      }
      return RTV.tensor(result, [nq, yCols]);
    }

    const col = yColData(0);
    const result = allocFloat64Array(nq);
    for (let i = 0; i < nq; i++) result[i] = interpOne(col, xqData[i]);
    return RTV.tensor(
      result,
      isRuntimeTensor(xqArg) ? [...xqArg.shape] : [1, 1]
    );
  },
});

// ── roots ────────────────────────────────────────────────────────────

builtin({
  name: "roots",
  nargin: 1,
  mfile: true,
  apply: args => {
    const p = toFloatArray(args[0], "roots");
    const n = p.length - 1;

    if (n <= 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);

    const C = allocFloat64Array(n * n);
    for (let i = 0; i < n - 1; i++) C[(i + 1) * n + i] = 1;
    for (let i = 0; i < n; i++) C[i] = -p[i + 1] / p[0];

    const { wr, wi } = getLapackBridge().eig(C, n, false, false, true);
    const realPart = allocFloat64Array(n);
    for (let i = 0; i < n; i++) realPart[i] = wr[i];
    if (!wi.some(v => Math.abs(v) > 0)) return RTV.tensor(realPart, [n, 1]);
    const imagPart = allocFloat64Array(n);
    for (let i = 0; i < n; i++) imagPart[i] = wi[i];
    return RTV.tensor(realPart, [n, 1], imagPart);
  },
});

// ── poly ─────────────────────────────────────────────────────────────

builtin({
  name: "poly",
  nargin: 1,
  mfile: true,
  apply: args => {
    const A = args[0];
    if (
      !(isRuntimeNumber(A) || isRuntimeComplexNumber(A) || isRuntimeTensor(A))
    )
      throw new RuntimeError("poly: argument must be numeric");

    const [m, n] = isRuntimeTensor(A) ? tensorSize2D(A) : [1, 1];

    if (m === 1 || n === 1) {
      // Expand prod(x - r) over the roots r (possibly complex).
      const r = toComplexArrays(A, "poly");
      const nr = r.re.length;
      const cr = allocFloat64Array(nr + 1);
      const ci = allocFloat64Array(nr + 1);
      cr[0] = 1;
      for (let i = 0; i < nr; i++) {
        const rr = r.re[i];
        const ri = r.im ? r.im[i] : 0;
        for (let j = i + 1; j >= 1; j--) {
          cr[j] -= rr * cr[j - 1] - ri * ci[j - 1];
          ci[j] -= rr * ci[j - 1] + ri * cr[j - 1];
        }
      }
      return complexTensor(cr, r.im ? ci : undefined, [1, nr + 1]);
    }

    if (m !== n) throw new RuntimeError("poly: matrix must be square");
    if (hasImagValue(A)) throw complexUnsupported("poly")();

    const At = A as RuntimeTensor;
    const { wr } = getLapackBridge().eig(At.data, n, false, false, true);
    let coeffs = allocFloat64Array(n + 1);
    coeffs[0] = 1;
    for (let i = 0; i < n; i++) {
      const newCoeffs = allocFloat64Array(n + 1);
      newCoeffs[0] = coeffs[0];
      for (let j = 1; j <= i + 1; j++) {
        newCoeffs[j] = coeffs[j] - wr[i] * coeffs[j - 1];
      }
      coeffs = newCoeffs;
    }
    return RTV.tensor(coeffs, [1, n + 1]);
  },
});

// ── cov ──────────────────────────────────────────────────────────────

builtin({
  name: "cov",
  nargin: [1, Infinity],
  mfile: true,
  apply: args =>
    withScratch(() => {
      if (args.length > 2)
        throw new RuntimeError("cov: options are not supported");
      // the variance of a single value is 0
      if (args.length === 1 && isRuntimeComplexNumber(args[0]))
        return RTV.num(0);
      for (const a of args)
        if (hasImagValue(a)) throw complexUnsupported("cov")();
      args = args.map(a => {
        try {
          const v = numericArg(a);
          return isRuntimeSparseMatrix(v) ? sparseToDense(v) : v;
        } catch {
          throw new RuntimeError("cov: argument must be numeric");
        }
      });
      if (
        args.length === 2 &&
        isRuntimeTensor(args[0]) &&
        isRuntimeTensor(args[1])
      ) {
        const x = toFloatArray(args[0], "cov");
        const y = toFloatArray(args[1], "cov");
        const n = x.length;

        let mx = 0,
          my = 0;
        for (let i = 0; i < n; i++) {
          mx += x[i];
          my += y[i];
        }
        mx /= n;
        my /= n;

        let sxx = 0,
          sxy = 0,
          syy = 0;
        for (let i = 0; i < n; i++) {
          sxx += (x[i] - mx) * (x[i] - mx);
          sxy += (x[i] - mx) * (y[i] - my);
          syy += (y[i] - my) * (y[i] - my);
        }
        const d = n - 1;
        return RTV.tensor(
          allocFloat64Array([sxx / d, sxy / d, sxy / d, syy / d]),
          [2, 2]
        );
      }

      const X = args[0];
      if (isRuntimeNumber(X) || isRuntimeComplexNumber(X)) return RTV.num(0);
      if (!isRuntimeTensor(X))
        throw new RuntimeError("cov: argument must be numeric");

      const [m, ncols] = tensorSize2D(X);

      if (m === 1 || ncols === 1) {
        const arr = toFloatArray(X, "cov");
        const n = arr.length;
        let mean = 0;
        for (let i = 0; i < n; i++) mean += arr[i];
        mean /= n;
        let s = 0;
        for (let i = 0; i < n; i++) s += (arr[i] - mean) * (arr[i] - mean);
        return RTV.num(s / (n - 1));
      }

      const means = allocFloat64Array(ncols);
      for (let j = 0; j < ncols; j++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += X.data[j * m + i];
        means[j] = s / m;
      }

      const result = allocFloat64Array(ncols * ncols);
      for (let p = 0; p < ncols; p++) {
        for (let q = 0; q < ncols; q++) {
          let s = 0;
          for (let i = 0; i < m; i++) {
            s +=
              (X.data[p * m + i] - means[p]) * (X.data[q * m + i] - means[q]);
          }
          result[q * ncols + p] = s / (m - 1);
        }
      }
      return RTV.tensor(result, [ncols, ncols]);
    }),
});

// ── corrcoef ─────────────────────────────────────────────────────────

builtin({
  name: "corrcoef",
  nargin: [1, Infinity],
  nargout: 4,
  mfile: true,
  apply: (args, nargout) =>
    withScratch(() => {
      if (nargout > 1)
        throw new RuntimeError("corrcoef: only the first output is supported");
      if (args.length > 2)
        throw new RuntimeError("corrcoef: options are not supported");
      let data: Float64Array;
      let m: number, ncols: number;

      if (args.length === 2) {
        const x = toFloatArray(args[0], "corrcoef", true);
        const y = toFloatArray(args[1], "corrcoef", true);
        const n = x.length;
        m = n;
        ncols = 2;
        data = allocFloat64Array(n * 2);
        for (let i = 0; i < n; i++) {
          data[i] = x[i];
          data[n + i] = y[i];
        }
      } else {
        const X = realDenseArg(args[0], complexUnsupported("corrcoef"));
        if (!isRuntimeTensor(X))
          throw new RuntimeError("corrcoef: argument must be numeric");
        [m, ncols] = tensorSize2D(X);
        if (m === 1 || ncols === 1) return RTV.num(1);
        data = allocFloat64Array(X.data);
      }

      const means = allocFloat64Array(ncols);
      for (let j = 0; j < ncols; j++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += data[j * m + i];
        means[j] = s / m;
      }

      const covMat = allocFloat64Array(ncols * ncols);
      for (let p = 0; p < ncols; p++) {
        for (let q = 0; q < ncols; q++) {
          let s = 0;
          for (let i = 0; i < m; i++) {
            s += (data[p * m + i] - means[p]) * (data[q * m + i] - means[q]);
          }
          covMat[q * ncols + p] = s / (m - 1);
        }
      }

      const result = allocFloat64Array(ncols * ncols);
      for (let p = 0; p < ncols; p++) {
        for (let q = 0; q < ncols; q++) {
          const denom = Math.sqrt(
            covMat[p * ncols + p] * covMat[q * ncols + q]
          );
          result[q * ncols + p] =
            denom === 0 ? 0 : covMat[q * ncols + p] / denom;
        }
      }
      return RTV.tensor(result, [ncols, ncols]);
    }),
});

// ── histc ────────────────────────────────────────────────────────────

builtin({
  name: "histc",
  nargin: [2, 3],
  nargout: 2,
  apply: (args, nargout) => {
    const xArg = isRuntimeChar(args[0])
      ? realDenseArg(args[0], complexUnsupported("histc"))
      : args[0];
    if (hasImagValue(xArg)) throw complexUnsupported("histc")();
    const edges = toFloatArray(args[1], "histc");
    const N = edges.length;

    const findBin = (v: number): number => {
      if (Number.isNaN(v)) return 0;
      if (v < edges[0] || v > edges[N - 1]) return 0;
      if (v === edges[N - 1]) return N;
      let lo = 0,
        hi = N - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (edges[mid] <= v) lo = mid;
        else hi = mid;
      }
      return lo + 1;
    };

    // Determine x shape and decide working dimension.
    let xRows: number, xCols: number;
    let xData: Float64Array;
    let xShape: number[];
    if (isRuntimeNumber(xArg) || isRuntimeLogical(xArg)) {
      xRows = 1;
      xCols = 1;
      xData = allocFloat64Array([
        isRuntimeLogical(xArg) ? (xArg ? 1 : 0) : (xArg as number),
      ]);
      xShape = [1, 1];
    } else if (isRuntimeTensor(xArg)) {
      [xRows, xCols] = tensorSize2D(xArg);
      xData = xArg.data;
      xShape = xArg.shape.slice();
    } else {
      throw new RuntimeError("histc: x must be numeric");
    }

    let dim: number;
    if (args.length >= 3) {
      dim = Math.round(realScalarArg(args[2]));
      if (dim !== 1 && dim !== 2)
        throw new RuntimeError("histc: dim must be 1 or 2");
    } else if (xRows === 1 || xCols === 1) {
      // Vector input: operate along the non-singleton dimension.
      dim = xRows === 1 ? 2 : 1;
    } else {
      dim = 1;
    }

    // Output bincounts shape: replace dim with N.
    let countShape: number[];
    if (dim === 1) countShape = [N, xCols];
    else countShape = [xRows, N];

    const counts = allocFloat64Array(countShape[0] * countShape[1]);
    const indData = allocFloat64Array(xRows * xCols);

    if (dim === 1) {
      // Iterate columns; each column is a separate series.
      for (let c = 0; c < xCols; c++) {
        for (let r = 0; r < xRows; r++) {
          const v = xData[c * xRows + r];
          const bin = findBin(v);
          indData[c * xRows + r] = bin;
          if (bin > 0) counts[c * N + (bin - 1)] += 1;
        }
      }
    } else {
      // dim === 2: iterate rows; each row is a separate series.
      for (let r = 0; r < xRows; r++) {
        for (let c = 0; c < xCols; c++) {
          const v = xData[c * xRows + r];
          const bin = findBin(v);
          indData[c * xRows + r] = bin;
          if (bin > 0) counts[(bin - 1) * xRows + r] += 1;
        }
      }
    }

    const isVectorX = xRows === 1 || xCols === 1;
    let countsResult: RuntimeValue;
    if (isRuntimeNumber(xArg) || isRuntimeLogical(xArg)) {
      // Scalar x — bincounts is a vector of length N (row).
      countsResult = RTV.tensor(counts, [1, N]);
    } else if (isVectorX) {
      // Vector x — bincounts vector matching x orientation.
      const isRow = xRows === 1;
      countsResult = RTV.tensor(counts, isRow ? [1, N] : [N, 1]);
    } else {
      countsResult = RTV.tensor(counts, countShape);
    }

    if (nargout <= 1) return countsResult;
    const indResult: RuntimeValue =
      isRuntimeNumber(xArg) || isRuntimeLogical(xArg)
        ? RTV.num(indData[0])
        : RTV.tensor(indData, xShape);
    return [countsResult, indResult];
  },
});

// ── discretize ───────────────────────────────────────────────────────

builtin({
  name: "discretize",
  help: {
    signatures: [
      "Y = discretize(X, edges)",
      "Y = discretize(X, edges, values)",
      "Y = discretize(..., 'IncludedEdge', side)",
      "[Y, E] = discretize(...)",
    ],
    description:
      "Bin index of each element of X for the bins defined by edges. Bin k is [edges(k), edges(k+1)), the last bin also includes its right edge; elements outside the edges (and NaN) give NaN. With values, returns values(k) instead of k. IncludedEdge 'right' uses bins (edges(k), edges(k+1)] with the first bin closed. The number-of-bins form discretize(X, N) is not supported.",
  },
  nargin: [2, Infinity],
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    const xArg = args[0];
    let rest = args.slice(2);
    let values: Float64Array | null = null;
    if (
      rest.length > 0 &&
      !isRuntimeChar(rest[0]) &&
      !isRuntimeString(rest[0])
    ) {
      values = toFloatArray(rest[0], "discretize");
      rest = rest.slice(1);
    }
    let right = false;
    for (let i = 0; i < rest.length; i += 2) {
      const opt =
        isRuntimeChar(rest[i]) || isRuntimeString(rest[i])
          ? (isRuntimeChar(rest[i])
              ? (rest[i] as { value: string }).value
              : (rest[i] as string)
            ).toLowerCase()
          : "";
      if (opt === "categorical")
        throw new RuntimeError(
          "discretize: categorical output is not supported"
        );
      if (opt !== "includededge" || i + 1 >= rest.length)
        throw new RuntimeError("discretize: invalid option");
      const v = rest[i + 1];
      const side = (isRuntimeChar(v) ? v.value : String(v)).toLowerCase();
      if (side !== "left" && side !== "right")
        throw new RuntimeError(
          "discretize: IncludedEdge must be 'left' or 'right'"
        );
      right = side === "right";
    }
    const edgesArg = args[1];
    if (
      isRuntimeNumber(edgesArg) ||
      (isRuntimeTensor(edgesArg) && edgesArg.data.length === 1)
    )
      throw new RuntimeError(
        "discretize: the number-of-bins form is not supported; pass bin edges"
      );
    const edges = toFloatArray(edgesArg, "discretize");
    const N = edges.length;
    if (N < 2)
      throw new RuntimeError("discretize: edges must have at least 2 elements");
    for (let i = 1; i < N; i++)
      if (!(edges[i] >= edges[i - 1]))
        throw new RuntimeError(
          "discretize: edges must be monotonically non-decreasing"
        );
    if (values && values.length !== N - 1)
      throw new RuntimeError(
        "discretize: values must have numel(edges)-1 elements"
      );

    // 1-based bin index of v, or 0 if outside
    const findBin = (v: number): number => {
      if (!(v >= edges[0] && v <= edges[N - 1])) return 0;
      if (!right) {
        if (v === edges[N - 1]) return N - 1; // last bin is closed
        // largest k with edges[k] <= v
        let lo = 0,
          hi = N - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (edges[mid] <= v) lo = mid;
          else hi = mid;
        }
        return lo + 1;
      }
      if (v === edges[0]) return 1; // first bin is closed
      // smallest k >= 1 with v <= edges[k]
      let lo = 0,
        hi = N - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (v <= edges[mid]) hi = mid;
        else lo = mid;
      }
      return hi;
    };
    const mapBin = (v: number): number => {
      const b = findBin(v);
      if (b === 0) return NaN;
      return values ? values[b - 1] : b;
    };

    let Y: RuntimeValue;
    if (isRuntimeNumber(xArg) || isRuntimeLogical(xArg)) {
      Y = RTV.num(
        mapBin(isRuntimeLogical(xArg) ? (xArg ? 1 : 0) : (xArg as number))
      );
    } else if (isRuntimeTensor(xArg) || isRuntimeComplexNumber(xArg)) {
      if (isRuntimeComplexNumber(xArg) || xArg.imag)
        throw builtinError(
          "MATLAB:discretize:invalidType",
          "Invalid data type. Data to be binned must be a real numeric or logical array."
        );
      const out = allocFloat64Array(xArg.data.length);
      for (let i = 0; i < out.length; i++) out[i] = mapBin(xArg.data[i]);
      Y = RTV.tensor(out, xArg.shape.slice());
    } else {
      throw new RuntimeError("discretize: X must be numeric");
    }
    if (nargout <= 1) return Y;
    return [
      Y,
      RTV.tensor(
        allocFloat64Array(edges),
        isColumnVector(edgesArg) ? [N, 1] : [1, N]
      ),
    ];
  },
});

// ── mkpp / ppval ─────────────────────────────────────────────────────

function dimArgToVec(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [Math.round(v as number)];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) {
    const out: number[] = [];
    for (let i = 0; i < v.data.length; i++) out.push(Math.round(v.data[i]));
    return out;
  }
  throw new RuntimeError("mkpp: dim must be a numeric scalar or vector");
}

// fnint (MATLAB Curve Fitting Toolbox) / ppint (Octave): antiderivative of a
// pp form, with F(breaks(1)) = 0. Scalar-valued pp only (dim == 1).
const ppIntApply = (fname: string) => (args: RuntimeValue[]) => {
  const pp = args[0];
  if (!isRuntimeStruct(pp))
    throw new RuntimeError(`${fname}: argument must be a pp struct`);
  const breaksRT = pp.fields.get("breaks");
  const coefsRT = pp.fields.get("coefs");
  const piecesRT = pp.fields.get("pieces");
  const orderRT = pp.fields.get("order");
  const dimRT = pp.fields.get("dim");
  if (
    breaksRT === undefined ||
    coefsRT === undefined ||
    piecesRT === undefined ||
    orderRT === undefined
  )
    throw new RuntimeError(`${fname}: pp struct missing required fields`);
  const breaks = toFloatArray(breaksRT, fname);
  const L = Math.round(toNumber(piecesRT));
  const k = Math.round(toNumber(orderRT));
  const dim = dimRT === undefined ? [1] : dimArgToVec(dimRT);
  if (dim.reduce((a, b) => a * b, 1) !== 1)
    throw new RuntimeError(`${fname}: only scalar-valued pp is supported`);
  if (!isRuntimeTensor(coefsRT))
    throw new RuntimeError(`${fname}: pp.coefs must be a matrix`);
  const coefs = coefsRT.data; // column-major, L rows x k cols
  // New coefs: L rows x (k+1) cols. Column c (< k) is old column c divided
  // by its new power (k - c); column k holds the accumulated constants.
  const out = allocFloat64Array(L * (k + 1));
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < L; i++) out[c * L + i] = coefs[c * L + i] / (k - c);
  }
  let A = 0;
  for (let i = 0; i < L; i++) {
    out[k * L + i] = A;
    const h = breaks[i + 1] - breaks[i];
    // Evaluate the antiderivative piece at its right end (Horner over the
    // new coefficients)
    let v = out[0 * L + i];
    for (let c = 1; c <= k; c++) v = v * h + out[c * L + i];
    A = v;
  }
  const fields = new Map<string, RuntimeValue>();
  fields.set("form", "pp");
  const breaksOut = allocFloat64Array(breaks.length);
  for (let i = 0; i < breaks.length; i++) breaksOut[i] = breaks[i];
  fields.set("breaks", RTV.tensor(breaksOut, [1, breaks.length]));
  fields.set("coefs", RTV.tensor(out, [L, k + 1]));
  fields.set("pieces", L);
  fields.set("order", k + 1);
  fields.set("dim", 1);
  return RTV.struct(fields);
};

for (const fname of ["fnint", "ppint"]) {
  builtin({ name: fname, nargin: 1, apply: ppIntApply(fname) });
}

builtin({
  name: "mkpp",
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const breaks = toFloatArray(args[0], "mkpp");
    const Lp1 = breaks.length;
    if (Lp1 < 2)
      throw new RuntimeError("mkpp: breaks must have at least 2 elements");
    const L = Lp1 - 1;
    for (let i = 0; i < L; i++) {
      if (!(breaks[i] < breaks[i + 1]))
        throw new RuntimeError("mkpp: breaks must be strictly increasing");
    }

    let dim: number[] = [1];
    if (args.length >= 3) {
      dim = dimArgToVec(args[2]);
      if (dim.length === 0) dim = [1];
      for (const d of dim)
        if (!(d >= 0))
          throw new RuntimeError("mkpp: dim entries must be non-negative");
    }
    const dProd = dim.reduce((a, b) => a * b, 1);

    const coefsArg = args[1];
    let rows: number, cols: number, coefData: Float64Array;
    if (isRuntimeNumber(coefsArg) || isRuntimeLogical(coefsArg)) {
      rows = 1;
      cols = 1;
      coefData = allocFloat64Array([
        isRuntimeLogical(coefsArg) ? (coefsArg ? 1 : 0) : (coefsArg as number),
      ]);
    } else if (isRuntimeTensor(coefsArg)) {
      if (coefsArg.imag) throw complexUnsupported("mkpp")();
      [rows, cols] = tensorSize2D(coefsArg);
      coefData = allocFloat64Array(coefsArg.data);
    } else {
      throw new RuntimeError("mkpp: coefs must be numeric");
    }

    if (rows !== dProd * L)
      throw new RuntimeError(
        `mkpp: coefs has ${rows} rows; expected ${dProd * L} (prod(dim)*pieces)`
      );
    const k = cols;

    const breaksOut = allocFloat64Array(Lp1);
    for (let i = 0; i < Lp1; i++) breaksOut[i] = breaks[i];

    const fields = new Map<string, RuntimeValue>();
    fields.set("form", "pp");
    fields.set("breaks", RTV.tensor(breaksOut, [1, Lp1]));
    fields.set("coefs", RTV.tensor(coefData, [rows, cols]));
    fields.set("pieces", L);
    fields.set("order", k);
    if (dim.length === 1) {
      fields.set("dim", dim[0]);
    } else {
      fields.set("dim", RTV.tensor(allocFloat64Array(dim), [1, dim.length]));
    }
    return RTV.struct(fields);
  },
});

builtin({
  name: "ppval",
  nargin: 2,
  mfile: true,
  apply: args => {
    const pp = args[0];
    if (!isRuntimeStruct(pp))
      throw new RuntimeError("ppval: first argument must be a pp struct");
    const formVal = pp.fields.get("form");
    const formStr =
      formVal === undefined
        ? ""
        : isRuntimeChar(formVal)
          ? formVal.value
          : isRuntimeString(formVal)
            ? (formVal as string)
            : "";
    if (formStr !== "pp") throw new RuntimeError("ppval: pp.form must be 'pp'");

    const breaksRT = pp.fields.get("breaks");
    const coefsRT = pp.fields.get("coefs");
    const piecesRT = pp.fields.get("pieces");
    const orderRT = pp.fields.get("order");
    const dimRT = pp.fields.get("dim");
    if (
      breaksRT === undefined ||
      coefsRT === undefined ||
      piecesRT === undefined ||
      orderRT === undefined ||
      dimRT === undefined
    )
      throw new RuntimeError("ppval: pp struct missing required fields");

    const breaks = toFloatArray(breaksRT, "ppval");
    const L = Math.round(toNumber(piecesRT));
    const k = Math.round(toNumber(orderRT));
    const dim: number[] = dimArgToVec(dimRT);
    const dProd = dim.reduce((a, b) => a * b, 1);

    if (!isRuntimeTensor(coefsRT))
      throw new RuntimeError("ppval: pp.coefs must be a matrix");
    const coefs = coefsRT.data;
    const [rows, cols] = tensorSize2D(coefsRT);
    if (rows !== dProd * L || cols !== k)
      throw new RuntimeError(
        "ppval: pp.coefs shape inconsistent with pieces/order/dim"
      );

    const findPiece = (x: number): number => {
      if (x < breaks[1]) return 0;
      if (x >= breaks[L - 1]) return L - 1;
      let lo = 0,
        hi = L;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (breaks[mid] <= x) lo = mid;
        else hi = mid;
      }
      return lo;
    };

    const evalAt = (x: number, out: Float64Array, off: number) => {
      const i = findPiece(x);
      const dx = x - breaks[i];
      const baseRow = i * dProd;
      for (let j = 0; j < dProd; j++) {
        const r = baseRow + j;
        let result = coefs[0 * rows + r];
        for (let c = 1; c < k; c++) {
          result = result * dx + coefs[c * rows + r];
        }
        out[off + j] = result;
      }
    };

    const xq = args[1];
    const isScalarDim = dim.length === 1 && dim[0] === 1;

    if (isRuntimeNumber(xq) || isRuntimeLogical(xq)) {
      const xv = isRuntimeLogical(xq) ? (xq ? 1 : 0) : (xq as number);
      if (isScalarDim) {
        const out = allocFloat64Array(1);
        evalAt(xv, out, 0);
        return RTV.num(out[0]);
      }
      const out = allocFloat64Array(dProd);
      evalAt(xv, out, 0);
      const outShape = dim.length === 1 ? [dim[0], 1] : [...dim];
      return RTV.tensor(out, outShape);
    }

    if (hasImagValue(xq))
      throw builtinError(
        "MATLAB:chckxy:XQComplex",
        "The interpolation points must be real."
      );
    if (!isRuntimeTensor(xq))
      throw new RuntimeError("ppval: query points must be numeric");

    const N = xq.data.length;
    const xqShape = xq.shape.slice();

    if (isScalarDim) {
      const out = allocFloat64Array(N);
      for (let i = 0; i < N; i++) evalAt(xq.data[i], out, i);
      return RTV.tensor(out, xqShape);
    }

    const out = allocFloat64Array(dProd * N);
    for (let i = 0; i < N; i++) evalAt(xq.data[i], out, i * dProd);

    const xqIsRowVec = xqShape.length === 2 && xqShape[0] === 1;
    const xqIsColVec = xqShape.length === 2 && xqShape[1] === 1;
    let outShape: number[];
    if (xqIsRowVec || xqIsColVec) {
      outShape = [...dim, N];
    } else {
      outShape = [...dim, ...xqShape];
    }
    return RTV.tensor(out, outShape);
  },
});

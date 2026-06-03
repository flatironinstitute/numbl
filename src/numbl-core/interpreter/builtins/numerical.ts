/**
 * Numerical builtins for the interpreter system:
 * conv, deconv, polyval, polyfit, trapz, cumtrapz, gradient,
 * interp1, roots, poly, cov, corrcoef, accumarray.
 *
 * These are thin wrappers that delegate to the legacy implementations
 * (which are already well-tested) while registering as IBuiltins for
 * proper JIT type inference.
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
  isRuntimeString,
  isRuntimeFunction,
  isRuntimeStruct,
} from "../../runtime/types.js";
import { defineBuiltin } from "./types.js";
import type { JitType } from "../../jitTypes.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { linsolveLapack } from "../../helpers/linsolve.js";
import { allocFloat64Array, withScratch } from "../../runtime/alloc.js";

function toFloatArray(v: RuntimeValue): Float64Array {
  if (isRuntimeNumber(v)) return allocFloat64Array([v as number]);
  if (isRuntimeLogical(v)) return allocFloat64Array([v ? 1 : 0]);
  if (isRuntimeTensor(v)) return allocFloat64Array(v.data);
  throw new RuntimeError("Expected numeric argument");
}

function isColumnVector(v: RuntimeValue): boolean {
  if (isRuntimeTensor(v))
    return v.shape[0] > 1 && (v.shape.length < 2 || v.shape[1] === 1);
  return false;
}

function varargMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length < 1) return null;
  return [{ kind: "unknown" }];
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

defineBuiltin({
  name: "eps",
  cases: [
    // eps() — no args
    {
      match: argTypes => {
        if (argTypes.length !== 0) return null;
        return [{ kind: "number" }];
      },
      apply: () => DOUBLE_EPS,
    },
    // eps('double') or eps('single')
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "char" && argTypes[0].kind !== "string")
          return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        const s = isRuntimeChar(v)
          ? v.value
          : isRuntimeString(v)
            ? (v as string)
            : "";
        if (s === "double") return DOUBLE_EPS;
        if (s === "single") return SINGLE_EPS;
        throw new RuntimeError(`eps: unknown data type '${s}'`);
      },
    },
    // eps(x) — scalar number
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const k = argTypes[0].kind;
        if (k === "number" || k === "boolean") return [{ kind: "number" }];
        return null;
      },
      apply: args => {
        const v = args[0];
        const x = typeof v === "boolean" ? (v ? 1 : 0) : (v as number);
        return epsOfScalar(x);
      },
    },
    // eps(x) — tensor
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "tensor") return null;
        const a = argTypes[0];
        return [{ kind: "tensor" as const, isComplex: false, shape: a.shape }];
      },
      apply: args => {
        const v = args[0];
        if (!isRuntimeTensor(v))
          throw new RuntimeError("eps: expected numeric argument");
        const n = v.data.length;
        const out = allocFloat64Array(n);
        for (let i = 0; i < n; i++) out[i] = epsOfScalar(v.data[i]);
        return RTV.tensor(out, v.shape.slice());
      },
    },
  ],
});

// ── conv ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "conv",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const a = toFloatArray(args[0]);
        const b = toFloatArray(args[1]);
        const m = a.length;
        const n = b.length;
        const colVec = isColumnVector(args[0]);

        const fullLen = m + n - 1;
        const full = allocFloat64Array(fullLen);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            full[i + j] += a[i] * b[j];
          }
        }

        let shape: string = "full";
        if (args.length === 3) {
          const opt = args[2];
          if (isRuntimeChar(opt)) shape = opt.value.toLowerCase();
          else if (isRuntimeString(opt)) shape = (opt as string).toLowerCase();
          else throw new RuntimeError("conv: third argument must be a string");
        }

        let result: Float64Array;
        if (shape === "full") {
          result = full;
        } else if (shape === "same") {
          const start = Math.floor(n / 2);
          result = allocFloat64Array(m);
          for (let i = 0; i < m; i++) result[i] = full[start + i];
        } else if (shape === "valid") {
          const validLen = Math.max(m, n) - Math.min(m, n) + 1;
          const start = Math.min(m, n) - 1;
          result = allocFloat64Array(validLen);
          for (let i = 0; i < validLen; i++) result[i] = full[start + i];
        } else {
          throw new RuntimeError(`conv: unknown shape '${shape}'`);
        }

        const outShape: [number, number] = colVec
          ? [result.length, 1]
          : [1, result.length];
        return RTV.tensor(result, outShape);
      },
    },
  ],
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

defineBuiltin({
  name: "conv2",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 4) return null;
        return [{ kind: "unknown" }];
      },
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
    },
  ],
});

// ── deconv ───────────────────────────────────────────────────────────

defineBuiltin({
  name: "deconv",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 2) return null;
        const out: JitType = { kind: "unknown" };
        return nargout > 1 ? [out, out] : [out];
      },
      apply: (args, nargout) => {
        const b = toFloatArray(args[0]);
        const a = toFloatArray(args[1]);
        const nb = b.length;
        const na = a.length;
        const colVec = isColumnVector(args[0]);
        const mkShape = (n: number): [number, number] =>
          colVec ? [n, 1] : [1, n];

        if (na === 0 || a[0] === 0)
          throw new RuntimeError(
            "deconv: leading coefficient of divisor is zero"
          );

        const nq = nb - na + 1;
        if (nq <= 0) {
          if (nargout <= 1)
            return RTV.tensor(allocFloat64Array([0]), mkShape(1));
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
    },
  ],
});

// ── polyval ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "polyval",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const p = toFloatArray(args[0]);
        const x = args[1];
        const np = p.length;

        const horner = (xVal: number): number => {
          let result = p[0];
          for (let i = 1; i < np; i++) result = result * xVal + p[i];
          return result;
        };

        if (isRuntimeNumber(x)) return RTV.num(horner(x as number));

        if (isRuntimeTensor(x)) {
          const result = allocFloat64Array(x.data.length);
          for (let i = 0; i < x.data.length; i++) result[i] = horner(x.data[i]);
          return RTV.tensor(result, [...x.shape]);
        }

        throw new RuntimeError("polyval: second argument must be numeric");
      },
    },
  ],
});

// ── polyfit ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "polyfit",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 3) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const xArr = toFloatArray(args[0]);
        const yArr = toFloatArray(args[1]);
        const n = Math.round(toNumber(args[2]));
        const m = xArr.length;

        if (xArr.length !== yArr.length)
          throw new RuntimeError("polyfit: x and y must have the same length");
        if (n < 0)
          throw new RuntimeError("polyfit: degree must be non-negative");

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
        if (!X) throw new RuntimeError("polyfit: LAPACK bridge unavailable");

        const result = allocFloat64Array(ncols);
        for (let i = 0; i < ncols; i++) result[i] = X[i];
        return RTV.tensor(result, [1, ncols]);
      },
    },
  ],
});

// ── trapz ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "trapz",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        let x: Float64Array | null = null;
        let y: Float64Array;

        if (args.length === 1) {
          y = toFloatArray(args[0]);
        } else {
          x = toFloatArray(args[0]);
          y = toFloatArray(args[1]);
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
    },
  ],
});

// ── cumtrapz ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "cumtrapz",
  cases: [
    {
      match: varargMatch,
      apply: args => {
        let x: Float64Array | null = null;
        let y: Float64Array;
        const yArg = args.length === 1 ? args[0] : args[1];

        if (args.length === 1) {
          y = toFloatArray(args[0]);
        } else {
          x = toFloatArray(args[0]);
          y = toFloatArray(args[1]);
          if (x.length !== y.length)
            throw new RuntimeError(
              "cumtrapz: x and y must have the same length"
            );
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
    },
  ],
});

// ── gradient ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "gradient",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        const out: JitType = { kind: "unknown" };
        return nargout > 1 ? [out, out] : [out];
      },
      apply: (args, nargout) => {
        const f = args[0];

        let hScalar: number | null = 1;
        let hVec: Float64Array | null = null;
        if (args.length === 2) {
          const spacing = args[1];
          if (isRuntimeNumber(spacing)) {
            hScalar = spacing as number;
          } else if (isRuntimeTensor(spacing)) {
            hVec = allocFloat64Array(spacing.data);
            hScalar = null;
          } else {
            hScalar = toNumber(spacing);
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
            (data[len - 1] - data[len - 2]) /
            (coords[len - 1] - coords[len - 2]);
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
    },
  ],
});

// ── accumarray ───────────────────────────────────────────────────────

defineBuiltin({
  name: "accumarray",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const subs = Array.from(toFloatArray(args[0]), (x: number) =>
          Math.round(x)
        );
        const nSubs = subs.length;

        let vals: Float64Array;
        if (isRuntimeNumber(args[1])) {
          vals = allocFloat64Array(nSubs);
          vals.fill(args[1] as number);
        } else {
          vals = toFloatArray(args[1]);
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
            if ((szArg as number) > maxIdx)
              maxIdx = Math.round(szArg as number);
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
    },
  ],
});

// ── interp1 ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "interp1",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 3 || argTypes.length > 5) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const xArr = toFloatArray(args[0]);
        const yArr = toFloatArray(args[1]);
        const xqArg = args[2];
        const nn = xArr.length;

        if (xArr.length !== yArr.length)
          throw new RuntimeError("interp1: x and y must have the same length");

        let method = "linear";
        let doExtrap = false;
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
          }
        }

        const interpOne = (xq: number): number => {
          if (xq < xArr[0] || xq > xArr[nn - 1]) {
            if (!doExtrap) return NaN;
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

        if (isRuntimeNumber(xqArg)) return RTV.num(interpOne(xqArg as number));

        if (isRuntimeTensor(xqArg)) {
          const result = allocFloat64Array(xqArg.data.length);
          for (let i = 0; i < xqArg.data.length; i++)
            result[i] = interpOne(xqArg.data[i]);
          return RTV.tensor(result, [...xqArg.shape]);
        }

        throw new RuntimeError("interp1: query points must be numeric");
      },
    },
  ],
});

// ── roots ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "roots",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const p = toFloatArray(args[0]);
        const n = p.length - 1;

        if (n <= 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);

        const C = allocFloat64Array(n * n);
        for (let i = 0; i < n - 1; i++) C[(i + 1) * n + i] = 1;
        for (let i = 0; i < n; i++) C[i] = -p[i + 1] / p[0];

        const bridge = getEffectiveBridge("roots", "eig");
        if (bridge.eig) {
          const f64 = C instanceof Float64Array ? C : allocFloat64Array(C);
          const eigResult = bridge.eig(
            f64 as Float64Array,
            n,
            false,
            false,
            true
          );
          if (eigResult) {
            let hasComplex = false;
            for (let i = 0; i < n; i++) {
              if (Math.abs(eigResult.wi[i]) > 0) {
                hasComplex = true;
                break;
              }
            }
            if (hasComplex) {
              const realPart = allocFloat64Array(n);
              const imagPart = allocFloat64Array(n);
              for (let i = 0; i < n; i++) {
                realPart[i] = eigResult.wr[i];
                imagPart[i] = eigResult.wi[i];
              }
              return RTV.tensor(realPart, [n, 1], imagPart);
            }
            const result = allocFloat64Array(n);
            for (let i = 0; i < n; i++) result[i] = eigResult.wr[i];
            return RTV.tensor(result, [n, 1]);
          }
        }

        if (n === 1) {
          return RTV.tensor(allocFloat64Array([-p[1] / p[0]]), [1, 1]);
        }
        if (n === 2) {
          const a = p[0],
            b = p[1],
            c = p[2];
          const disc = b * b - 4 * a * c;
          if (disc >= 0) {
            return RTV.tensor(
              allocFloat64Array([
                (-b + Math.sqrt(disc)) / (2 * a),
                (-b - Math.sqrt(disc)) / (2 * a),
              ]),
              [2, 1]
            );
          }
          return RTV.tensor(
            allocFloat64Array([-b / (2 * a), -b / (2 * a)]),
            [2, 1],
            allocFloat64Array([
              Math.sqrt(-disc) / (2 * a),
              -Math.sqrt(-disc) / (2 * a),
            ])
          );
        }

        throw new RuntimeError(
          "roots: LAPACK required for polynomials of degree > 2"
        );
      },
    },
  ],
});

// ── poly ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "poly",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const A = args[0];

        if (isRuntimeNumber(A)) {
          return RTV.tensor(allocFloat64Array([1, -(A as number)]), [1, 2]);
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("poly: argument must be numeric");

        const [m, n] = tensorSize2D(A);

        if (m === 1 || n === 1) {
          const roots = toFloatArray(A);
          const nr = roots.length;
          let coeffs = allocFloat64Array(nr + 1);
          coeffs[0] = 1;
          for (let i = 0; i < nr; i++) {
            const newCoeffs = allocFloat64Array(nr + 1);
            newCoeffs[0] = coeffs[0];
            for (let j = 1; j <= i + 1; j++) {
              newCoeffs[j] = coeffs[j] - roots[i] * coeffs[j - 1];
            }
            coeffs = newCoeffs;
          }
          return RTV.tensor(coeffs, [1, nr + 1]);
        }

        if (m !== n) throw new RuntimeError("poly: matrix must be square");

        const bridge = getEffectiveBridge("poly", "eig");
        if (bridge.eig) {
          const f64 =
            A.data instanceof Float64Array ? A.data : allocFloat64Array(A.data);
          const eigResult = bridge.eig(
            f64 as Float64Array,
            n,
            false,
            false,
            true
          );
          if (eigResult) {
            const roots = allocFloat64Array(n);
            for (let i = 0; i < n; i++) roots[i] = eigResult.wr[i];
            let coeffs = allocFloat64Array(n + 1);
            coeffs[0] = 1;
            for (let i = 0; i < n; i++) {
              const newCoeffs = allocFloat64Array(n + 1);
              newCoeffs[0] = coeffs[0];
              for (let j = 1; j <= i + 1; j++) {
                newCoeffs[j] = coeffs[j] - roots[i] * coeffs[j - 1];
              }
              coeffs = newCoeffs;
            }
            return RTV.tensor(coeffs, [1, n + 1]);
          }
        }

        if (n === 2) {
          const a00 = A.data[0],
            a10 = A.data[1],
            a01 = A.data[m],
            a11 = A.data[m + 1];
          const tr = a00 + a11;
          const dt = a00 * a11 - a01 * a10;
          return RTV.tensor(allocFloat64Array([1, -tr, dt]), [1, 3]);
        }

        throw new RuntimeError(
          "poly: matrix input requires LAPACK for matrices > 2x2"
        );
      },
    },
  ],
});

// ── cov ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cov",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args =>
        withScratch(() => {
          if (
            args.length === 2 &&
            isRuntimeTensor(args[0]) &&
            isRuntimeTensor(args[1])
          ) {
            const x = toFloatArray(args[0]);
            const y = toFloatArray(args[1]);
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
          if (isRuntimeNumber(X)) return RTV.num(0);
          if (!isRuntimeTensor(X))
            throw new RuntimeError("cov: argument must be numeric");

          const [m, ncols] = tensorSize2D(X);

          if (m === 1 || ncols === 1) {
            const arr = toFloatArray(X);
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
                  (X.data[p * m + i] - means[p]) *
                  (X.data[q * m + i] - means[q]);
              }
              result[q * ncols + p] = s / (m - 1);
            }
          }
          return RTV.tensor(result, [ncols, ncols]);
        }),
    },
  ],
});

// ── corrcoef ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "corrcoef",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args =>
        withScratch(() => {
          let data: Float64Array;
          let m: number, ncols: number;

          if (args.length === 2) {
            const x = toFloatArray(args[0]);
            const y = toFloatArray(args[1]);
            const n = x.length;
            m = n;
            ncols = 2;
            data = allocFloat64Array(n * 2);
            for (let i = 0; i < n; i++) {
              data[i] = x[i];
              data[n + i] = y[i];
            }
          } else {
            const X = args[0];
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
                s +=
                  (data[p * m + i] - means[p]) * (data[q * m + i] - means[q]);
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
    },
  ],
});

// ── histc ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "histc",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        const out: JitType = { kind: "unknown" };
        return nargout > 1 ? [out, out] : [out];
      },
      apply: (args, nargout) => {
        const xArg = args[0];
        const edges = toFloatArray(args[1]);
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
          dim = Math.round(toNumber(args[2]));
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
    },
  ],
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

defineBuiltin({
  name: "mkpp",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        return [{ kind: "struct", fields: {} }];
      },
      apply: args => {
        const breaks = toFloatArray(args[0]);
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
            isRuntimeLogical(coefsArg)
              ? coefsArg
                ? 1
                : 0
              : (coefsArg as number),
          ]);
        } else if (isRuntimeTensor(coefsArg)) {
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
          fields.set(
            "dim",
            RTV.tensor(allocFloat64Array(dim), [1, dim.length])
          );
        }
        return RTV.struct(fields);
      },
    },
  ],
});

defineBuiltin({
  name: "ppval",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "unknown" }];
      },
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
        if (formStr !== "pp")
          throw new RuntimeError("ppval: pp.form must be 'pp'");

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

        const breaks = toFloatArray(breaksRT);
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
    },
  ],
});

/**
 * Array extras: colon, tril, triu, toeplitz, magic, bitwise ops,
 * coordinate transforms, and nchoosek.
 */

import { allocFloat64Array } from "../../runtime/alloc.js";
import {
  RTV,
  toNumber,
  toString,
  RuntimeError,
  mRange,
} from "../../runtime/index.js";
import {
  type RuntimeValue,
  type RuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeChar,
  RuntimeChar,
} from "../../runtime/types.js";
import { builtin } from "./builtin.js";
import { builtinError, invalidDataTypeError, mustBeRealError } from "./args.js";
import { imagAllZero } from "../../helpers/effectively-real.js";

// ── colon ────────────────────────────────────────────────────────────

/** A colon operand: a real numeric, char, or logical scalar. */
function colonOperand(v: RuntimeValue): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  const notRealScalar = () =>
    builtinError(
      "MATLAB:colon:operandsNotRealScalar",
      "Colon operands must be real scalars."
    );
  if (isRuntimeComplexNumber(v)) throw notRealScalar();
  if (isRuntimeChar(v)) {
    if (v.value.length !== 1) throw notRealScalar();
    return v.value.charCodeAt(0);
  }
  if (isRuntimeTensor(v)) {
    if (v.data.length !== 1 || (v.imag && v.imag[0] !== 0))
      throw notRealScalar();
    return v.data[0];
  }
  throw builtinError(
    "MATLAB:colon:inputsMustBeNumericCharLogical",
    "Inputs must be numeric, char, or logical."
  );
}

builtin({
  name: "colon",
  nargin: [2, 3],
  apply: args => {
    const ops = args.map(colonOperand);
    const result =
      ops.length === 2
        ? mRange(ops[0], 1, ops[1])
        : mRange(ops[0], ops[1], ops[2]);
    // colon('a','c') is the char row 'abc', as for the `:` operator.
    const isChar =
      isRuntimeChar(args[0]) || isRuntimeChar(args[args.length - 1]);
    if (isChar && isRuntimeTensor(result))
      return RTV.char(
        Array.from(result.data)
          .map(c => String.fromCharCode(c))
          .join("")
      );
    return result;
  },
});

// ── triu / tril ──────────────────────────────────────────────────────

function triPart(
  args: RuntimeValue[],
  keepFn: (i: number, j: number, k: number) => boolean,
  name: string
): RuntimeValue {
  let k = 0;
  if (args.length === 2) {
    const kArg = args[1];
    const kv =
      typeof kArg === "number"
        ? kArg
        : typeof kArg === "boolean"
          ? Number(kArg)
          : isRuntimeTensor(kArg) && kArg.data.length === 1 && !kArg.imag
            ? kArg.data[0]
            : NaN;
    if (!Number.isInteger(kv))
      throw builtinError(
        `MATLAB:${name}:kthDiagInputNotInteger`,
        "K-th diagonal input must be an integer scalar."
      );
    k = kv;
  }
  const v = args[0];
  if (isRuntimeNumber(v)) return keepFn(0, 0, k) ? v : RTV.num(0);
  if (isRuntimeComplexNumber(v)) return keepFn(0, 0, k) ? v : RTV.num(0);
  if (isRuntimeLogical(v)) return keepFn(0, 0, k) ? v : false;
  if (isRuntimeSparseMatrix(v)) {
    const isComplex = v.pi !== undefined;
    const irArr: number[] = [];
    const prArr: number[] = [];
    const piArr: number[] = [];
    const jc = new Int32Array(v.n + 1);
    for (let c = 0; c < v.n; c++) {
      jc[c] = irArr.length;
      for (let kk = v.jc[c]; kk < v.jc[c + 1]; kk++) {
        if (keepFn(v.ir[kk], c, k)) {
          irArr.push(v.ir[kk]);
          prArr.push(v.pr[kk]);
          if (isComplex) piArr.push(v.pi![kk]);
        }
      }
    }
    jc[v.n] = irArr.length;
    return RTV.sparseMatrix(
      v.m,
      v.n,
      new Int32Array(irArr),
      jc,
      allocFloat64Array(prArr),
      isComplex ? allocFloat64Array(piArr) : undefined
    );
  }
  if (isRuntimeChar(v))
    throw new RuntimeError(`${name}: char input is not supported`);
  if (!isRuntimeTensor(v)) throw invalidDataTypeError();
  if (v.shape.length > 2)
    throw builtinError(
      `MATLAB:${name}:firstInputMustBe2D`,
      "First input must be 2D."
    );
  const nrows = v.shape[0] ?? 1;
  const ncols = v.shape.length >= 2 ? v.shape[1] : 1;
  const data = allocFloat64Array(nrows * ncols);
  const idata = v.imag ? allocFloat64Array(nrows * ncols) : undefined;
  for (let j = 0; j < ncols; j++) {
    for (let i = 0; i < nrows; i++) {
      if (keepFn(i, j, k)) {
        const idx = i + j * nrows;
        data[idx] = v.data[idx];
        if (idata && v.imag) idata[idx] = v.imag[idx];
      }
    }
  }
  // MATLAB narrows a result whose imaginary parts are all zero.
  const im = idata && idata.some(x => x !== 0) ? idata : undefined;
  return RTV.tensor(data, [nrows, ncols], im);
}

builtin({
  name: "triu",
  nargin: [1, 2],
  apply: args => triPart(args, (i, j, k) => j - i >= k, "triu"),
});

builtin({
  name: "tril",
  nargin: [1, 2],
  apply: args => triPart(args, (i, j, k) => i - j >= -k, "tril"),
});

// ── toeplitz ─────────────────────────────────────────────────────────

builtin({
  name: "toeplitz",
  nargin: [1, 2],
  mfile: true,
  apply: args => {
    function vecData(v: RuntimeValue): {
      re: Float64Array;
      im: Float64Array | undefined;
      len: number;
    } {
      if (isRuntimeNumber(v))
        return {
          re: allocFloat64Array([v as number]),
          im: undefined,
          len: 1,
        };
      if (isRuntimeComplexNumber(v))
        return {
          re: allocFloat64Array([v.re]),
          im: allocFloat64Array([v.im]),
          len: 1,
        };
      if (isRuntimeTensor(v))
        return { re: v.data, im: v.imag, len: v.data.length };
      throw builtinError(
        "MATLAB:toeplitz:nonNumericInputs",
        "Inputs must be numeric."
      );
    }

    let colRe: Float64Array,
      colIm: Float64Array | undefined,
      colLen: number,
      rowRe: Float64Array,
      rowIm: Float64Array | undefined,
      rowLen: number;

    if (args.length === 1) {
      const { re, im, len } = vecData(args[0]);
      rowRe = re;
      rowIm = im;
      rowLen = len;
      colRe = allocFloat64Array(re);
      colLen = len;
      if (im) {
        const ci = allocFloat64Array(im);
        for (let k = 1; k < len; k++) ci[k] = -im[k];
        colIm = ci;
      }
    } else {
      const c = vecData(args[0]);
      const r = vecData(args[1]);
      colRe = c.re;
      colIm = c.im;
      colLen = c.len;
      rowRe = r.re;
      rowIm = r.im;
      rowLen = r.len;
    }

    const m = colLen;
    const n = rowLen;
    const isComplex = colIm !== undefined || rowIm !== undefined;
    const data = allocFloat64Array(m * n);
    const idata = isComplex ? allocFloat64Array(m * n) : undefined;

    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        const idx = i + j * m;
        const diag = i - j;
        if (diag >= 0) {
          data[idx] = colRe[diag];
          if (idata) idata[idx] = colIm ? colIm[diag] : 0;
        } else {
          const k = -diag;
          data[idx] = rowRe[k];
          if (idata) idata[idx] = rowIm ? rowIm[k] : 0;
        }
      }
    }

    return RTV.tensor(data, [m, n], idata);
  },
});

// ── magic ────────────────────────────────────────────────────────────

builtin({
  name: "magic",
  nargin: 1,
  mfile: true,
  apply: args => {
    // MATLAB: n = floor(real(double(n))); n < 1 gives [], and magic(2)
    // is [1 3; 4 2].
    const a = args[0];
    let nRaw: number;
    if (typeof a === "number") nRaw = a;
    else if (typeof a === "boolean") nRaw = a ? 1 : 0;
    else if (isRuntimeComplexNumber(a)) nRaw = a.re;
    else if (isRuntimeTensor(a) && a.data.length === 1) nRaw = a.data[0];
    else if (isRuntimeChar(a) && a.value.length === 1)
      nRaw = a.value.charCodeAt(0);
    else if (isRuntimeTensor(a) || isRuntimeChar(a))
      throw builtinError(
        "MATLAB:validation:IncompatibleSize",
        "Invalid argument at position 1. Value must be a scalar."
      );
    else throw invalidDataTypeError();
    const n = Math.floor(nRaw);
    if (!(n >= 1)) return RTV.tensor(allocFloat64Array(0), [0, 0]);
    if (n === 1) return 1;
    if (n === 2) return RTV.tensor(allocFloat64Array([1, 4, 3, 2]), [2, 2]);

    const M = new Array<number>(n * n).fill(0);
    const set = (r: number, c: number, v: number) => {
      M[c * n + r] = v;
    };
    const get = (r: number, c: number) => M[c * n + r];

    if (n % 2 === 1) {
      let i = 0;
      let j = Math.floor(n / 2);
      for (let k = 1; k <= n * n; k++) {
        set(i, j, k);
        const ni = (i - 1 + n) % n;
        const nj = (j + 1) % n;
        if (get(ni, nj) !== 0) {
          i = (i + 1) % n;
        } else {
          i = ni;
          j = nj;
        }
      }
    } else if (n % 4 === 0) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          set(i, j, i * n + j + 1);
        }
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const ii = i % 4;
          const jj = j % 4;
          if (
            ((ii === 0 || ii === 3) && (jj === 0 || jj === 3)) ||
            ((ii === 1 || ii === 2) && (jj === 1 || jj === 2))
          ) {
            set(i, j, n * n + 1 - get(i, j));
          }
        }
      }
    } else {
      const p = n / 2;
      const sub = new Array<number>(p * p).fill(0);
      const sset = (r: number, c: number, v: number) => {
        sub[c * p + r] = v;
      };
      const sget = (r: number, c: number) => sub[c * p + r];

      let si = 0;
      let sj = Math.floor(p / 2);
      for (let k = 1; k <= p * p; k++) {
        sset(si, sj, k);
        const ni = (si - 1 + p) % p;
        const nj = (sj + 1) % p;
        if (sget(ni, nj) !== 0) {
          si = (si + 1) % p;
        } else {
          si = ni;
          sj = nj;
        }
      }

      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          const v = sget(i, j);
          set(i, j, v);
          set(i, j + p, v + 2 * p * p);
          set(i + p, j, v + 3 * p * p);
          set(i + p, j + p, v + p * p);
        }
      }

      const k = Math.floor((n - 2) / 4);
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < k; j++) {
          if (j === 0) {
            if (i === Math.floor(p / 2)) {
              const tmp = get(i, 1);
              set(i, 1, get(i + p, 1));
              set(i + p, 1, tmp);
            } else {
              const tmp = get(i, 0);
              set(i, 0, get(i + p, 0));
              set(i + p, 0, tmp);
            }
          } else {
            const tmp = get(i, j);
            set(i, j, get(i + p, j));
            set(i + p, j, tmp);
          }
        }
        for (let j = n - k + 2; j < n; j++) {
          const tmp = get(i, j);
          set(i, j, get(i + p, j));
          set(i + p, j, tmp);
        }
      }
    }

    return RTV.tensor(allocFloat64Array(M), [n, n]);
  },
});

// ── Bitwise ops ──────────────────────────────────────────────────────

/** MATLAB's error identifiers for each bit function: the id prefix, the
 *  suffix for complex operands, the suffix and message for non-numeric
 *  operands. */
const BIT_IDS: Record<string, [string, string, string, string]> = {
  bitand: [
    "Bitand",
    "complexInputs",
    "operandsNotNumeric",
    "Operands to BIT Ops must be numeric.",
  ],
  bitor: [
    "Bitor",
    "complexInputs",
    "operandsNotNumeric",
    "Operands to BIT Ops must be numeric.",
  ],
  bitxor: [
    "bitxor",
    "complexInputs",
    "operandsNotNumeric",
    "Operands to BIT Ops must be numeric.",
  ],
  bitshift: [
    "bitshift",
    "complexArgs",
    "numericArg",
    "Inputs must be numeric.",
  ],
  bitget: ["bitget", "complexArgs", "numericArg", "Inputs must be numeric."],
  bitset: ["bitset", "complexArgs", "numericArg", "Inputs must be numeric."],
};

/** A bit-op operand: a real number or a real tensor. Logical values count
 *  as 0/1; complex (with a nonzero imaginary part), char, and non-numeric
 *  values raise MATLAB's errors. */
function bitOperand(v: RuntimeValue, name: string): number | RuntimeTensor {
  const [prefix, cplx, nonNum, nonNumMsg] = BIT_IDS[name];
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  const complexErr = () =>
    builtinError(`MATLAB:${prefix}:${cplx}`, "Inputs must be real.");
  if (isRuntimeComplexNumber(v)) {
    if (v.im !== 0) throw complexErr();
    return v.re;
  }
  if (isRuntimeTensor(v)) {
    if (v.imag) {
      if (!imagAllZero(v.imag)) throw complexErr();
      return RTV.tensor(v.data, v.shape) as RuntimeTensor;
    }
    return v;
  }
  throw builtinError(`MATLAB:${prefix}:${nonNum}`, nonNumMsg);
}

const INTEGER_TYPE_NAMES = [
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
];

/** Reject the optional ASSUMEDTYPE argument: an invalid name raises
 *  MATLAB's error; a valid one is not supported (numbl has no integer
 *  types to range-check against). */
function rejectAssumedType(v: RuntimeValue, name: string): never {
  const prefix = BIT_IDS[name][0];
  if (
    !(isRuntimeChar(v) || isRuntimeString(v)) ||
    !INTEGER_TYPE_NAMES.includes(toString(v))
  )
    throw builtinError(
      `MATLAB:${prefix}:unknownTypeString`,
      "ASSUMEDTYPE must be an integer type name."
    );
  throw new RuntimeError(`${name}: the ASSUMEDTYPE argument is not supported`);
}

function bitwiseOp(
  aArg: RuntimeValue,
  bArg: RuntimeValue,
  op: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  const a = bitOperand(aArg, name);
  const b = bitOperand(bArg, name);
  if (typeof a === "number" && typeof b === "number") {
    return RTV.num(op(Math.round(a), Math.round(b)));
  }
  if (typeof a !== "number" && typeof b === "number") {
    const bv = Math.round(b);
    const result = allocFloat64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = op(Math.round(a.data[i]), bv);
    }
    return RTV.tensor(result, [...a.shape]);
  }
  if (typeof a === "number" && typeof b !== "number") {
    const av = Math.round(a);
    const result = allocFloat64Array(b.data.length);
    for (let i = 0; i < b.data.length; i++) {
      result[i] = op(av, Math.round(b.data[i]));
    }
    return RTV.tensor(result, [...b.shape]);
  }
  const at = a as RuntimeTensor;
  const bt = b as RuntimeTensor;
  if (at.data.length !== bt.data.length)
    throw new RuntimeError(`${name}: arrays must be the same size`);
  const result = allocFloat64Array(at.data.length);
  for (let i = 0; i < at.data.length; i++) {
    result[i] = op(Math.round(at.data[i]), Math.round(bt.data[i]));
  }
  return RTV.tensor(result, [...at.shape]);
}

function defineBitBinary(
  name: string,
  op: (a: number, b: number) => number
): void {
  builtin({
    name,
    nargin: [2, 3],
    apply: args => {
      if (args.length === 3) rejectAssumedType(args[2], name);
      return bitwiseOp(args[0], args[1], op, name);
    },
  });
}

defineBitBinary("bitand", (a, b) => a & b);
defineBitBinary("bitor", (a, b) => a | b);
defineBitBinary("bitxor", (a, b) => a ^ b);
defineBitBinary("bitshift", (a, k) => (k >= 0 ? a << k : a >>> -k));

// bitget(A, bit) returns bit number `bit` (1-based, 1 = LSB) of each
// element of A. MATLAB supports vector `bit` with broadcasting; we match
// the scalar-or-same-shape rules used by the other bitwise ops.
defineBitBinary("bitget", (a, b) => {
  if (b < 1) return 0;
  // Use BigInt for bit positions ≥ 32 to avoid JS integer overflow
  if (b <= 31) return (a >>> (b - 1)) & 1;
  const bi = BigInt(a);
  return Number((bi >> BigInt(b - 1)) & 1n);
});

// bitset(A, bit)     → set bit to 1
// bitset(A, bit, v)  → set bit to v (nonzero means 1)
builtin({
  name: "bitset",
  nargin: [2, 4],
  apply: args => {
    if (args.length === 4) rejectAssumedType(args[3], "bitset");
    let v = 1;
    if (args.length === 3) {
      const vArg = args[2];
      if (isRuntimeChar(vArg) || isRuntimeString(vArg))
        rejectAssumedType(vArg, "bitset");
      const vv = bitOperand(vArg, "bitset");
      if (typeof vv !== "number") {
        if (vv.data.length !== 1)
          throw new RuntimeError("bitset: array V is not supported");
        v = Math.round(vv.data[0]);
      } else v = Math.round(vv);
    }
    return bitwiseOp(
      args[0],
      args[1],
      (a, b) => {
        if (b < 1) return a;
        if (b <= 31) {
          const mask = 1 << (b - 1);
          return v ? a | mask : a & ~mask;
        }
        const bi = BigInt(a);
        const mask = 1n << BigInt(b - 1);
        const out = v ? bi | mask : bi & ~mask;
        return Number(out);
      },
      "bitset"
    );
  },
});

// ── Coordinate transforms ────────────────────────────────────────────

/** A coordinate-transform input as a real double array (MATLAB requires
 *  double or single; numbl has only double). */
function coordOperand(v: RuntimeValue): RuntimeTensor {
  if (typeof v === "number")
    return RTV.tensor(allocFloat64Array([v]), [1, 1]) as RuntimeTensor;
  if (isRuntimeComplexNumber(v)) {
    if (v.im !== 0) throw mustBeRealError();
    return RTV.tensor(allocFloat64Array([v.re]), [1, 1]) as RuntimeTensor;
  }
  if (isRuntimeTensor(v) && v.cls !== "logical") {
    if (v.imag) {
      if (!imagAllZero(v.imag)) throw mustBeRealError();
      return RTV.tensor(v.data, v.shape) as RuntimeTensor;
    }
    return v;
  }
  throw builtinError(
    "MATLAB:math:mustBeFloat",
    "Invalid data type. Argument must be double or single."
  );
}

/** Apply `fn` element-wise over `ins` with MATLAB implicit expansion,
 *  producing `nOut` results of the broadcast shape (numbers when every
 *  input is scalar). */
function broadcastMap(
  ins: RuntimeTensor[],
  nOut: number,
  fn: (...vals: number[]) => number[]
): RuntimeValue[] {
  const ndim = Math.max(...ins.map(t => t.shape.length));
  const dimOf = (t: RuntimeTensor, d: number) =>
    d < t.shape.length ? t.shape[d] : 1;
  const outShape: number[] = [];
  for (let d = 0; d < ndim; d++) {
    let s = 1;
    for (const t of ins) {
      const td = dimOf(t, d);
      if (td === 1) continue;
      if (s !== 1 && s !== td)
        throw builtinError(
          "MATLAB:sizeDimensionsMustMatch",
          "Arrays have incompatible sizes for this operation."
        );
      s = td;
    }
    outShape.push(s);
  }
  let len = 1;
  for (const s of outShape) len *= s;
  const allScalar = ins.every(t => t.data.length === 1);
  const outs = Array.from({ length: nOut }, () => allocFloat64Array(len));
  const sub = new Array<number>(ndim).fill(0);
  const vals = new Array<number>(ins.length);
  for (let i = 0; i < len; i++) {
    for (let j = 0; j < ins.length; j++) {
      const t = ins[j];
      let idx = 0;
      let stride = 1;
      for (let d = 0; d < ndim; d++) {
        const td = dimOf(t, d);
        if (td !== 1) idx += sub[d] * stride;
        stride *= td;
      }
      vals[j] = t.data[idx];
    }
    const r = fn(...vals);
    for (let k = 0; k < nOut; k++) outs[k][i] = r[k];
    for (let d = 0; d < ndim; d++) {
      if (++sub[d] < outShape[d]) break;
      sub[d] = 0;
    }
  }
  return outs.map(d => (allScalar ? d[0] : RTV.tensor(d, outShape.slice())));
}

/** Define a coordinate transform. `nIn` inputs map through `fn` to `nOut`
 *  outputs; when `passZ`, an optional extra input z passes through
 *  unchanged as the last output (cart2pol / pol2cart). */
function coordTransform(
  name: string,
  nIn: number,
  nOut: number,
  passZ: boolean,
  fn: (...vals: number[]) => number[]
): void {
  builtin({
    name,
    nargin: passZ ? [nIn, nIn + 1] : nIn,
    nargout: 3,
    mfile: true,
    apply: (args, nargout) => {
      const ins = args.slice(0, nIn).map(coordOperand);
      const outs = broadcastMap(ins, nOut, fn);
      if (passZ && args.length > nIn) {
        coordOperand(args[nIn]);
        outs.push(args[nIn]);
      }
      if (nargout <= 1) return outs[0];
      return outs.slice(0, nargout);
    },
  });
}

coordTransform("cart2sph", 3, 3, false, (x, y, z) => {
  const hypotxy = Math.sqrt(x * x + y * y);
  return [
    Math.atan2(y, x),
    Math.atan2(z, hypotxy),
    Math.sqrt(x * x + y * y + z * z),
  ];
});

coordTransform("sph2cart", 3, 3, false, (az, el, r) => {
  const rcosel = r * Math.cos(el);
  return [rcosel * Math.cos(az), rcosel * Math.sin(az), r * Math.sin(el)];
});

coordTransform("cart2pol", 2, 2, true, (x, y) => [
  Math.atan2(y, x),
  Math.sqrt(x * x + y * y),
]);

coordTransform("pol2cart", 2, 2, true, (theta, rho) => [
  rho * Math.cos(theta),
  rho * Math.sin(theta),
]);

// ── nchoosek ───────────────────────────────────────────────────────────

/** Binomial coefficient n!/((n-k)!k!) via the multiplicative formula. */
function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < k; i++) {
    result = (result * (n - i)) / (i + 1);
  }
  return Math.round(result);
}

/** Invoke cb on each k-combination of indices 0..m-1, in lexicographic
 *  order (matching MATLAB's row order). The same `idx` array is reused
 *  across calls, so callers must read it before returning. */
function forEachCombination(
  m: number,
  k: number,
  cb: (idx: number[]) => void
): void {
  if (k < 0 || k > m) return;
  const idx = Array.from({ length: k }, (_, i) => i);
  for (;;) {
    cb(idx);
    let i = k - 1;
    while (i >= 0 && idx[i] === m - k + i) i--;
    if (i < 0) break;
    idx[i]++;
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
}

builtin({
  name: "nchoosek",
  nargin: 2,
  mfile: true,
  apply: args => {
    const v = args[0];
    const kArg = args[1];
    const k =
      typeof kArg === "number"
        ? kArg
        : typeof kArg === "boolean"
          ? Number(kArg)
          : isRuntimeTensor(kArg) && kArg.data.length === 1 && !kArg.imag
            ? kArg.data[0]
            : NaN;
    if (!Number.isInteger(k) || k < 0)
      throw builtinError(
        "MATLAB:nchoosek:InvalidArg2",
        "The second input has to be a non-negative integer."
      );

    // numel == 1 (a MATLAB scalar) → binomial coefficient b = nchoosek(n,k).
    const numel = isRuntimeTensor(v)
      ? v.data.length
      : isRuntimeChar(v)
        ? v.value.length
        : 1;
    if (numel === 1) {
      if (
        isRuntimeComplexNumber(v) ||
        (isRuntimeTensor(v) && v.imag && v.imag[0] !== 0)
      )
        throw new RuntimeError("nchoosek: complex N is not supported");
      const n = Math.round(toNumber(v));
      if (!Number.isFinite(n) || n < 0)
        throw new RuntimeError(
          "nchoosek: N must be a nonnegative integer scalar"
        );
      return RTV.num(binomial(n, k));
    }

    // Otherwise C = nchoosek(v,k): all combinations of vector v.
    const rows = binomial(numel, k);

    if (isRuntimeChar(v)) {
      const chars = v.value;
      let out = "";
      forEachCombination(numel, k, idx => {
        for (let c = 0; c < k; c++) out += chars[idx[c]];
      });
      return new RuntimeChar(out, [rows, k]);
    }

    if (!isRuntimeTensor(v))
      throw new RuntimeError("nchoosek: V must be a vector");

    const re = allocFloat64Array(rows * k);
    const im = v.imag ? allocFloat64Array(rows * k) : undefined;
    let r = 0;
    forEachCombination(numel, k, idx => {
      for (let c = 0; c < k; c++) {
        re[c * rows + r] = v.data[idx[c]];
        if (im) im[c * rows + r] = v.imag![idx[c]];
      }
      r++;
    });
    const result = RTV.tensor(re, [rows, k], im, v.cls);
    return result;
  },
});

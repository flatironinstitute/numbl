/**
 * The element-wise kernel layer: binary arithmetic, logical-result binary
 * ops and unary math over dense operands, with N-d implicit expansion.
 *
 * Callers classify their values first (char to codes, sparse densified,
 * logical to 0/1, result class) and then call one entry point here. An
 * operand is a real scalar (a JS number) or a dense array
 * `{data, imag?, shape}` (a `RuntimeTensor` is one as it is). A result is
 * always a fresh dense array; `imag` is absent when the result is real, and
 * a complex result whose imaginary parts are all zero is returned as real.
 *
 * The ops with a native kernel (`OpBin` codes below `NATIVE_BIN_OP_COUNT`,
 * comparisons below `NATIVE_CMP_OP_COUNT`, and unary math) go through
 * `tensorOps`, so the native addon serves them when it is loaded, for
 * same-shape and scalar-array operands. Broadcast operands and the other
 * ops run the loops in this file. Results are identical either way.
 */

import { tensorOps } from "./dispatch.js";
import {
  OpBin,
  OpCmp,
  OpUnary,
  NATIVE_BIN_OP_COUNT,
  NATIVE_CMP_OP_COUNT,
} from "./opCodes.js";
import { broadcastShape, forEachBroadcastRow } from "./broadcast.js";
import { cdivResult, complexDivide } from "./complexDivide.js";

/** A dense operand. `RuntimeTensor` satisfies this interface. */
export interface DenseOperand {
  readonly data: Float64Array;
  readonly imag?: Float64Array;
  readonly shape: readonly number[];
}

/** A kernel operand: a real scalar or a dense array. */
export type Operand = number | DenseOperand;

/** A kernel result. `imag` is absent when the result is real. */
export interface ElemResult {
  data: Float64Array;
  imag?: Float64Array;
  shape: number[];
}

// ── Scalar formulas ─────────────────────────────────────────────────────

function modScalar(a: number, b: number): number {
  if (b === 0) return a;
  let r = a % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return r;
}

/** `op` on two real elements. POW here is `Math.pow`: the caller decides
 *  first whether a negative base makes the result complex. */
export function realElem(op: number, x: number, y: number): number {
  switch (op) {
    case OpBin.ADD:
      return x + y;
    case OpBin.SUB:
      return x - y;
    case OpBin.MUL:
      return x * y;
    case OpBin.DIV:
      return x / y;
    case OpBin.POW:
      return Math.pow(x, y);
    case OpBin.MIN:
      return x !== x ? y : y !== y ? x : Math.min(x, y);
    case OpBin.MAX:
      return x !== x ? y : y !== y ? x : Math.max(x, y);
    case OpBin.MOD:
      return modScalar(x, y);
    case OpBin.REM:
      return x % y;
    case OpBin.ATAN2:
      return Math.atan2(x, y);
    case OpBin.HYPOT:
      return Math.hypot(x, y);
    default:
      throw new Error(`elementwise: unknown binary op ${op}`);
  }
}

/** cos(pi * x) with exact results at integer and half-integer x. */
function cospi(x: number): number {
  if (Number.isInteger(x)) return (x & 1) === 0 ? 1 : -1;
  if (Number.isInteger(x * 2)) return 0;
  return Math.cos(Math.PI * x);
}

/** sin(pi * x) with exact results at integer and half-integer x. */
function sinpi(x: number): number {
  if (Number.isInteger(x)) return 0;
  const twice = x * 2;
  if (Number.isInteger(twice)) {
    // x = n + 0.5; sin(pi*(n+0.5)) = (-1)^n.
    return (((twice - 1) / 2) & 1) === 0 ? 1 : -1;
  }
  return Math.sin(Math.PI * x);
}

// The complex formulas write their result here (no allocation per element).
let cRe = 0;
let cIm = 0;

/** Complex division (see complexDivide.ts). */
function complexDiv(aRe: number, aIm: number, bRe: number, bIm: number) {
  complexDivide(aRe, aIm, bRe, bIm);
  cRe = cdivResult.re;
  cIm = cdivResult.im;
}

/** Principal-branch z^w. A pure-real negative base with a real exponent
 *  uses cospi/sinpi so the phase is exact at integer and half-integer
 *  exponents (MATLAB: real((-4)^0.5) == 0 exactly). */
function complexPow(aRe: number, aIm: number, bRe: number, bIm: number) {
  if (aIm === 0 && bIm === 0 && aRe < 0) {
    const mag = Math.pow(-aRe, bRe);
    cRe = mag * cospi(bRe);
    cIm = mag * sinpi(bRe);
    return;
  }
  const r = Math.hypot(aRe, aIm);
  if (r === 0) {
    // 0^w: 1 for w = 0, 0 for a positive real part, else Inf.
    cIm = 0;
    cRe = bRe === 0 && bIm === 0 ? 1 : bRe > 0 ? 0 : Infinity;
    return;
  }
  const theta = Math.atan2(aIm, aRe);
  const lnR = Math.log(r);
  const newRe = bRe * lnR - bIm * theta;
  const newIm = bRe * theta + bIm * lnR;
  const expR = Math.exp(newRe);
  cRe = expR * Math.cos(newIm);
  cIm = expR * Math.sin(newIm);
}

/** Complex min/max: by magnitude, ties broken by angle; picks `b` when
 *  neither is better. */
function complexPick(
  max: boolean,
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
) {
  const absA = Math.sqrt(aRe * aRe + aIm * aIm);
  const absB = Math.sqrt(bRe * bRe + bIm * bIm);
  let pickA: boolean;
  if (absA !== absB) pickA = max ? absA > absB : absA < absB;
  else {
    const angA = Math.atan2(aIm, aRe);
    const angB = Math.atan2(bIm, bRe);
    pickA = max ? angA > angB : angA < angB;
  }
  cRe = pickA ? aRe : bRe;
  cIm = pickA ? aIm : bIm;
}

/** `op` on two complex elements; the result is in (cRe, cIm). */
function complexElem(
  op: number,
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
): void {
  switch (op) {
    case OpBin.ADD:
      cRe = aRe + bRe;
      cIm = aIm + bIm;
      return;
    case OpBin.SUB:
      cRe = aRe - bRe;
      cIm = aIm - bIm;
      return;
    case OpBin.MUL:
      cRe = aRe * bRe - aIm * bIm;
      cIm = aRe * bIm + aIm * bRe;
      return;
    case OpBin.DIV:
      complexDiv(aRe, aIm, bRe, bIm);
      return;
    case OpBin.POW:
      complexPow(aRe, aIm, bRe, bIm);
      return;
    case OpBin.MIN:
      complexPick(false, aRe, aIm, bRe, bIm);
      return;
    case OpBin.MAX:
      complexPick(true, aRe, aIm, bRe, bIm);
      return;
    default:
      throw new Error(`elementwise: op ${op} has no complex form`);
  }
}

/** Complex times or divided by real, without promoting the real operand to
 *  complex: each part of `c` is scaled on its own, and a zero part of a
 *  number with a nonzero other part stays exactly 0 (promoting would turn
 *  Inf or NaN in `x` into NaN parts). An element with a zero imaginary
 *  part is scaled as a real number (MATLAB: [0 1i] .* [Inf 1] has NaN
 *  first). `realLeft` is for `x ./ c`. */
function mixedElem(
  op: number,
  re: number,
  im: number,
  x: number,
  realLeft: boolean
): void {
  if (op === OpBin.MUL) {
    if (im === 0) {
      cRe = re * x;
      cIm = 0;
    } else {
      cRe = re === 0 ? 0 : re * x;
      cIm = im * x;
    }
  } else if (!realLeft) {
    if (im === 0) {
      cRe = re / x;
      cIm = 0;
    } else {
      cRe = re === 0 ? 0 : re / x;
      cIm = im / x;
    }
  } else if (im === 0) {
    cRe = x / re;
    cIm = 0;
  } else {
    complexDiv(x, 0, re, im);
  }
}

/**
 * `op` on two scalars. `aComplex` / `bComplex` say whether an operand is
 * complex-typed (it has an imaginary part, possibly zero). The rules are
 * the array rules of `elementwiseBinary`: for MUL and DIV an operand with a
 * zero imaginary part counts as real, and real times complex scales each
 * part (for scalars, unless that gives a real zero or NaN); a negative real base with a non-integer real exponent gives a
 * complex power. The returned object is reused by the next call.
 */
export function binaryScalar(
  op: number,
  aRe: number,
  aIm: number,
  aComplex: boolean,
  bRe: number,
  bIm: number,
  bComplex: boolean
): { re: number; im: number } {
  if (op === OpBin.MUL || op === OpBin.DIV) {
    aComplex = aComplex && aIm !== 0;
    bComplex = bComplex && bIm !== 0;
    if (aComplex !== bComplex) {
      if (aComplex) mixedElem(op, aRe, aIm, bRe, false);
      else mixedElem(op, bRe, bIm, aRe, true);
      // A real zero or NaN result is recomputed with the complex formula,
      // which gives MATLAB's +0 for 0 .* (-1.5-0.5i).
      if (cIm !== 0 || (cRe !== 0 && cRe === cRe)) return scalarOut();
    }
  }
  if (aComplex || bComplex) {
    complexElem(op, aRe, aIm, bRe, bIm);
    return scalarOut();
  }
  if (op === OpBin.POW && aRe < 0 && !Number.isInteger(bRe)) {
    complexPow(aRe, 0, bRe, 0);
    return scalarOut();
  }
  scalarResult.re = realElem(op, aRe, bRe);
  scalarResult.im = 0;
  return scalarResult;
}

const scalarResult = { re: 0, im: 0 };
function scalarOut(): { re: number; im: number } {
  scalarResult.re = cRe;
  scalarResult.im = cIm;
  return scalarResult;
}

// ── Operand helpers ─────────────────────────────────────────────────────

const SCALAR_SHAPE: readonly number[] = [1, 1];

function numelOf(x: Operand): number {
  return typeof x === "number" ? 1 : x.data.length;
}

function shapeOf(x: Operand): readonly number[] {
  return typeof x === "number" ? SCALAR_SHAPE : x.shape;
}

function imagOf(x: Operand): Float64Array | undefined {
  return typeof x === "number" ? undefined : x.imag;
}

/** The real data of an operand (a number becomes a 1-element array). */
function reOf(x: Operand): Float64Array {
  return typeof x === "number" ? Float64Array.of(x) : x.data;
}

function anyNonzero(v: Float64Array): boolean {
  for (let i = 0; i < v.length; i++) if (v[i] !== 0) return true;
  return false;
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** The output shape of a binary op: a scalar operand takes the other's
 *  shape, otherwise the broadcast shape. */
function outShapeOf(a: Operand, b: Operand): number[] {
  const sa = shapeOf(a);
  const sb = shapeOf(b);
  if (numelOf(a) === 1 && sa.length === 2 && sa[0] === 1) return [...sb];
  if (numelOf(b) === 1 && sb.length === 2 && sb[0] === 1) return [...sa];
  if (sameShape(sa, sb)) return [...sa];
  return broadcastShape(sa, sb);
}

function numelOfShape(shape: readonly number[]): number {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

/** The result with an all-zero imaginary part dropped. */
function complexResult(
  re: Float64Array,
  im: Float64Array,
  shape: number[]
): ElemResult {
  return anyNonzero(im) ? { data: re, imag: im, shape } : { data: re, shape };
}

// ── Real binary ─────────────────────────────────────────────────────────

/** out[io+k] = a[ia+k*sa] op b[ib+k*sb] for k < n. */
function realRow(
  op: number,
  n: number,
  a: Float64Array,
  ia: number,
  sa: number,
  b: Float64Array,
  ib: number,
  sb: number,
  out: Float64Array,
  io: number
): void {
  const end = io + n;
  switch (op) {
    case OpBin.ADD:
      for (let o = io; o < end; o++, ia += sa, ib += sb) out[o] = a[ia] + b[ib];
      return;
    case OpBin.SUB:
      for (let o = io; o < end; o++, ia += sa, ib += sb) out[o] = a[ia] - b[ib];
      return;
    case OpBin.MUL:
      for (let o = io; o < end; o++, ia += sa, ib += sb) out[o] = a[ia] * b[ib];
      return;
    case OpBin.DIV:
      for (let o = io; o < end; o++, ia += sa, ib += sb) out[o] = a[ia] / b[ib];
      return;
    case OpBin.POW:
      if (sb === 0 && b[ib] === 2) {
        // Math.pow(x, 2) is x * x; the multiply is faster.
        for (let o = io; o < end; o++, ia += sa) out[o] = a[ia] * a[ia];
      } else {
        for (let o = io; o < end; o++, ia += sa, ib += sb)
          out[o] = Math.pow(a[ia], b[ib]);
      }
      return;
    case OpBin.MIN:
      for (let o = io; o < end; o++, ia += sa, ib += sb) {
        const x = a[ia];
        const y = b[ib];
        out[o] = x !== x ? y : y !== y ? x : Math.min(x, y);
      }
      return;
    case OpBin.MAX:
      for (let o = io; o < end; o++, ia += sa, ib += sb) {
        const x = a[ia];
        const y = b[ib];
        out[o] = x !== x ? y : y !== y ? x : Math.max(x, y);
      }
      return;
    case OpBin.MOD:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = modScalar(a[ia], b[ib]);
      return;
    case OpBin.REM:
      for (let o = io; o < end; o++, ia += sa, ib += sb) out[o] = a[ia] % b[ib];
      return;
    case OpBin.ATAN2:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = Math.atan2(a[ia], b[ib]);
      return;
    case OpBin.HYPOT:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = Math.hypot(a[ia], b[ib]);
      return;
    default:
      throw new Error(`elementwise: unknown binary op ${op}`);
  }
}

function hasNegative(x: Operand): boolean {
  if (typeof x === "number") return x < 0;
  const d = x.data;
  for (let i = 0; i < d.length; i++) if (d[i] < 0) return true;
  return false;
}

function allIntegers(x: Operand): boolean {
  if (typeof x === "number") return Number.isInteger(x);
  const d = x.data;
  for (let i = 0; i < d.length; i++) if (!Number.isInteger(d[i])) return false;
  return true;
}

function realBinary(op: number, a: Operand, b: Operand): ElemResult {
  // A negative base with a non-integer exponent anywhere makes x.^y
  // complex; every element then uses the complex formula.
  if (op === OpBin.POW && hasNegative(a) && !allIntegers(b))
    return complexBinary(op, a, b);
  const shape = outShapeOf(a, b);
  const n = numelOfShape(shape);
  const out = new Float64Array(n);
  if (n === 0) return { data: out, shape };
  const na = numelOf(a);
  const nb = numelOf(b);
  const native = op < NATIVE_BIN_OP_COUNT;
  if (na === n && nb === n) {
    // Same shape (or both scalars).
    const ad = reOf(a);
    const bd = reOf(b);
    if (native) tensorOps.realBinaryElemwise(op, n, ad, bd, out);
    else realRow(op, n, ad, 0, 1, bd, 0, 1, out, 0);
  } else if (na === 1 || nb === 1) {
    const left = na === 1;
    const s = left ? scalarOf(a) : scalarOf(b);
    const arr = left ? (b as DenseOperand).data : (a as DenseOperand).data;
    if (native) tensorOps.realScalarBinaryElemwise(op, n, s, arr, left, out);
    else {
      const sv = Float64Array.of(s);
      if (left) realRow(op, n, sv, 0, 0, arr, 0, 1, out, 0);
      else realRow(op, n, arr, 0, 1, sv, 0, 0, out, 0);
    }
  } else {
    const ad = reOf(a);
    const bd = reOf(b);
    forEachBroadcastRow(
      shapeOf(a),
      shapeOf(b),
      shape,
      (ia, sa, ib, sb, io, len) =>
        realRow(op, len, ad, ia, sa, bd, ib, sb, out, io)
    );
  }
  return { data: out, shape };
}

function scalarOf(x: Operand): number {
  return typeof x === "number" ? x : x.data[0];
}

// ── Complex binary ──────────────────────────────────────────────────────

/** Complex `op` with broadcasting, through the per-element formulas.
 *  A missing imaginary part reads as 0. */
function complexLoop(
  op: number,
  a: Operand,
  b: Operand,
  shape: number[],
  n: number
): ElemResult {
  const ar = reOf(a);
  const ai = imagOf(a);
  const br = reOf(b);
  const bi = imagOf(b);
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const row = (
    ia: number,
    sa: number,
    ib: number,
    sb: number,
    io: number,
    len: number
  ) => {
    for (let o = io; o < io + len; o++, ia += sa, ib += sb) {
      complexElem(op, ar[ia], ai ? ai[ia] : 0, br[ib], bi ? bi[ib] : 0);
      outRe[o] = cRe;
      outIm[o] = cIm;
    }
  };
  const na = numelOf(a);
  const nb = numelOf(b);
  if (na === n && nb === n) row(0, 1, 0, 1, 0, n);
  else if (na === 1) row(0, 0, 0, 1, 0, n);
  else if (nb === 1) row(0, 1, 0, 0, 0, n);
  else forEachBroadcastRow(shapeOf(a), shapeOf(b), shape, row);
  return complexResult(outRe, outIm, shape);
}

/** Complex `op` where at least one operand has an imaginary part.
 *  ADD..DIV on same-shape or scalar-array operands use the native kernels;
 *  scalar pairs and broadcasts use the per-element formulas (for DIV,
 *  Smith's algorithm). */
function complexBinary(op: number, a: Operand, b: Operand): ElemResult {
  const shape = outShapeOf(a, b);
  const n = numelOfShape(shape);
  const na = numelOf(a);
  const nb = numelOf(b);
  if (op >= NATIVE_BIN_OP_COUNT || n === 0 || (na === 1 && nb === 1))
    return complexLoop(op, a, b, shape, n);
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  if (na === n && nb === n) {
    tensorOps.complexBinaryElemwise(
      op,
      n,
      reOf(a),
      imagOf(a) ?? null,
      reOf(b),
      imagOf(b) ?? null,
      outRe,
      outIm
    );
  } else if (na === 1 || nb === 1) {
    const left = na === 1;
    const s = left ? a : b;
    const t = (left ? b : a) as DenseOperand;
    const sIm = imagOf(s);
    tensorOps.complexScalarBinaryElemwise(
      op,
      n,
      scalarOf(s),
      sIm ? sIm[0] : 0,
      t.data,
      t.imag ?? null,
      left,
      outRe,
      outIm
    );
  } else {
    return complexLoop(op, a, b, shape, n);
  }
  return complexResult(outRe, outIm, shape);
}

/** Complex times or divided by real (`c` complex, `r` real). */
function mixedBinary(
  op: number,
  c: DenseOperand,
  r: Operand,
  realLeft: boolean
): ElemResult {
  const shape = realLeft ? outShapeOf(r, c) : outShapeOf(c, r);
  const n = numelOfShape(shape);
  const cr = c.data;
  const ci = c.imag as Float64Array;
  const rd = reOf(r);
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const row = (
    ic: number,
    sc: number,
    ir: number,
    sr: number,
    io: number,
    len: number
  ) => {
    for (let o = io; o < io + len; o++, ic += sc, ir += sr) {
      mixedElem(op, cr[ic], ci[ic], rd[ir], realLeft);
      outRe[o] = cRe;
      outIm[o] = cIm;
    }
  };
  const nc = cr.length;
  const nr = rd.length;
  if (nc === n && nr === n) row(0, 1, 0, 1, 0, n);
  else if (nc === 1) row(0, 0, 0, 1, 0, n);
  else if (nr === 1) row(0, 1, 0, 0, 0, n);
  else forEachBroadcastRow(c.shape, shapeOf(r), shape, row);
  return complexResult(outRe, outIm, shape);
}

/** A complex operand as a dense one (a number has no imaginary part, so
 *  this is only called on dense operands). */
function dense(x: Operand): DenseOperand {
  return x as DenseOperand;
}

// ── Binary entry point ──────────────────────────────────────────────────

/**
 * Element-wise `a op b` (`op` an `OpBin` code) with implicit expansion.
 * Raises MATLAB's size-mismatch error for incompatible shapes. MOD, REM,
 * ATAN2 and HYPOT are real-only; the caller rejects complex input.
 */
export function elementwiseBinary(
  op: number,
  a: Operand,
  b: Operand
): ElemResult {
  const ai = imagOf(a);
  const bi = imagOf(b);
  if (!ai && !bi) return realBinary(op, a, b);
  if (op === OpBin.MUL || op === OpBin.DIV) {
    // A complex operand whose imaginary part is all zero counts as real.
    const aC = ai !== undefined && anyNonzero(ai);
    const bC = bi !== undefined && anyNonzero(bi);
    if (!aC && !bC) return realBinary(op, realView(a), realView(b));
    if (!aC) return mixedBinary(op, dense(b), realView(a), true);
    if (!bC) return mixedBinary(op, dense(a), realView(b), false);
  }
  return complexBinary(op, a, b);
}

/** An operand without its imaginary part. */
function realView(x: Operand): Operand {
  if (typeof x === "number" || x.imag === undefined) return x;
  return { data: x.data, shape: x.shape };
}

// ── Logical-result binary ───────────────────────────────────────────────

/** out[io+k] = (a[ia+k*sa] op b[ib+k*sb]) ? 1 : 0 for k < n. */
function cmpRow(
  op: number,
  n: number,
  a: Float64Array,
  ia: number,
  sa: number,
  b: Float64Array,
  ib: number,
  sb: number,
  out: Float64Array,
  io: number
): void {
  const end = io + n;
  switch (op) {
    case OpCmp.EQ:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] === b[ib] ? 1 : 0;
      return;
    case OpCmp.NE:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] !== b[ib] ? 1 : 0;
      return;
    case OpCmp.LT:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] < b[ib] ? 1 : 0;
      return;
    case OpCmp.LE:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] <= b[ib] ? 1 : 0;
      return;
    case OpCmp.GT:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] > b[ib] ? 1 : 0;
      return;
    case OpCmp.GE:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] >= b[ib] ? 1 : 0;
      return;
    case OpCmp.AND:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] !== 0 && b[ib] !== 0 ? 1 : 0;
      return;
    case OpCmp.OR:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = a[ia] !== 0 || b[ib] !== 0 ? 1 : 0;
      return;
    case OpCmp.XOR:
      for (let o = io; o < end; o++, ia += sa, ib += sb)
        out[o] = (a[ia] !== 0) !== (b[ib] !== 0) ? 1 : 0;
      return;
    default:
      throw new Error(`elementwise: unknown comparison op ${op}`);
  }
}

/** `op` (an `OpCmp` code) on two scalars, with the rules of
 *  `elementwiseCompare`. */
export function compareScalar(
  op: number,
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
): boolean {
  switch (op) {
    case OpCmp.EQ:
      return aRe === bRe && aIm === bIm;
    case OpCmp.NE:
      return aRe !== bRe || aIm !== bIm;
    case OpCmp.LT:
      return aRe < bRe;
    case OpCmp.LE:
      return aRe <= bRe;
    case OpCmp.GT:
      return aRe > bRe;
    case OpCmp.GE:
      return aRe >= bRe;
    case OpCmp.AND:
      return (aRe !== 0 || aIm !== 0) && (bRe !== 0 || bIm !== 0);
    case OpCmp.OR:
      return aRe !== 0 || aIm !== 0 || bRe !== 0 || bIm !== 0;
    case OpCmp.XOR:
      return (aRe !== 0 || aIm !== 0) !== (bRe !== 0 || bIm !== 0);
    default:
      throw new Error(`elementwise: unknown comparison op ${op}`);
  }
}

/** A complex operand as its nonzero mask (for AND, OR, XOR). */
function nonzeroMask(x: Operand): Operand {
  if (typeof x === "number" || x.imag === undefined) return x;
  const re = x.data;
  const im = x.imag;
  const m = new Float64Array(re.length);
  for (let i = 0; i < re.length; i++) m[i] = re[i] !== 0 || im[i] !== 0 ? 1 : 0;
  return { data: m, shape: x.shape };
}

/**
 * Element-wise `a op b` for an `OpCmp` code, with implicit expansion. The
 * result is 0/1 (the caller makes it logical). `==` and `~=` compare
 * complex values; the ordering ops compare real parts only (as MATLAB
 * does); AND, OR and XOR test for nonzero values.
 */
export function elementwiseCompare(
  op: number,
  a: Operand,
  b: Operand
): ElemResult {
  if (op === OpCmp.AND || op === OpCmp.OR || op === OpCmp.XOR) {
    a = nonzeroMask(a);
    b = nonzeroMask(b);
  }
  const shape = outShapeOf(a, b);
  const n = numelOfShape(shape);
  const out = new Float64Array(n);
  if (n === 0) return { data: out, shape };
  const na = numelOf(a);
  const nb = numelOf(b);
  const ai = imagOf(a);
  const bi = imagOf(b);
  const native = op < NATIVE_CMP_OP_COUNT;
  if ((ai || bi) && (op === OpCmp.EQ || op === OpCmp.NE)) {
    const ar = reOf(a);
    const br = reOf(b);
    if (na === n && nb === n) {
      tensorOps.complexComparison(op, n, ar, ai ?? null, br, bi ?? null, out);
    } else if (na === 1 || nb === 1) {
      const left = na === 1;
      const sIm = left ? ai : bi;
      tensorOps.complexScalarComparison(
        op,
        n,
        left ? ar[0] : br[0],
        sIm ? sIm[0] : 0,
        left ? br : ar,
        (left ? bi : ai) ?? null,
        left,
        out
      );
    } else {
      forEachBroadcastRow(
        shapeOf(a),
        shapeOf(b),
        shape,
        (ia, sa, ib, sb, io, len) => {
          for (let o = io; o < io + len; o++, ia += sa, ib += sb)
            out[o] = compareScalar(
              op,
              ar[ia],
              ai ? ai[ia] : 0,
              br[ib],
              bi ? bi[ib] : 0
            )
              ? 1
              : 0;
        }
      );
    }
    return { data: out, shape };
  }
  if (na === n && nb === n) {
    const ad = reOf(a);
    const bd = reOf(b);
    if (native) tensorOps.realComparison(op, n, ad, bd, out);
    else cmpRow(op, n, ad, 0, 1, bd, 0, 1, out, 0);
  } else if (na === 1 || nb === 1) {
    const left = na === 1;
    const s = left ? scalarOf(a) : scalarOf(b);
    const arr = left ? reOf(b) : reOf(a);
    if (native) tensorOps.realScalarComparison(op, n, s, arr, left, out);
    else {
      const sv = Float64Array.of(s);
      if (left) cmpRow(op, n, sv, 0, 0, arr, 0, 1, out, 0);
      else cmpRow(op, n, arr, 0, 1, sv, 0, 0, out, 0);
    }
  } else {
    const ad = reOf(a);
    const bd = reOf(b);
    forEachBroadcastRow(
      shapeOf(a),
      shapeOf(b),
      shape,
      (ia, sa, ib, sb, io, len) =>
        cmpRow(op, len, ad, ia, sa, bd, ib, sb, out, io)
    );
  }
  return { data: out, shape };
}

// ── Generic and unary ───────────────────────────────────────────────────

/** Element-wise `fn(a, b)` on real operands with implicit expansion, for
 *  builtins whose per-element function has no op code. */
export function mapBinary(
  fn: (x: number, y: number) => number,
  a: Operand,
  b: Operand
): ElemResult {
  const shape = outShapeOf(a, b);
  const n = numelOfShape(shape);
  const out = new Float64Array(n);
  const ad = reOf(a);
  const bd = reOf(b);
  forEachBroadcastRow(
    shapeOf(a),
    shapeOf(b),
    shape,
    (ia, sa, ib, sb, io, len) => {
      for (let o = io; o < io + len; o++, ia += sa, ib += sb)
        out[o] = fn(ad[ia], bd[ib]);
    }
  );
  return { data: out, shape };
}

/** Element-wise `op` (an `OpUnary` code) on a dense operand. ABS of a
 *  complex operand is real; the other ops keep the imaginary part. */
export function elementwiseUnary(op: number, a: DenseOperand): ElemResult {
  const n = a.data.length;
  const shape = [...a.shape];
  const out = new Float64Array(n);
  if (!a.imag) {
    tensorOps.realUnaryElemwise(op, n, a.data, out);
    return { data: out, shape };
  }
  if (op === OpUnary.ABS) {
    tensorOps.complexAbs(n, a.data, a.imag, out);
    return { data: out, shape };
  }
  const outIm = new Float64Array(n);
  tensorOps.complexUnaryElemwise(op, n, a.data, a.imag, out, outIm);
  return { data: out, imag: outIm, shape };
}

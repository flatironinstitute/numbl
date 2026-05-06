/**
 * Tensor operation helpers for JIT-compiled code.
 * Handles element-wise binary/unary ops, comparison ops, concat, and
 * tensor coercion for struct-array field access.
 */

import { RuntimeTensor } from "../../../runtime/types.js";
import type { RuntimeComplexNumber } from "../../../runtime/types.js";
import { re, im, mkc, cAdd, cSub, cMul, cDiv } from "./jitHelpersComplex.js";
import { tensorOps, OpRealBin } from "../../../ops/index.js";
import { allocFloat64Array } from "./alloc.js";
// Real/complex binary op codes are aligned (ADD=0, SUB=1, MUL=2, DIV=3) so
// the same OpRealBin value is passed to tensorOps.complexBinaryElemwise too.

// ── Type checks ────────────────────────────────────────────────────────

function isTensor(v: unknown): v is RuntimeTensor {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  );
}

function isComplex(v: unknown): v is RuntimeComplexNumber {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeComplexNumber).kind === "complex_number"
  );
}

// ── Tensor creation ────────────────────────────────────────────────────

export function makeTensor(
  data: Float64Array,
  imag: Float64Array | undefined,
  shape: number[]
): RuntimeTensor {
  const s = [...shape];
  while (s.length > 2 && s[s.length - 1] === 1) s.pop();
  return new RuntimeTensor(data, s, imag);
}

// ── Output-buffer reuse (dest-hint) ────────────────────────────────────
//
// `tAdd` / `tSub` / etc. accept a `dest` hint from the JIT codegen so the
// helpers can later return a pre-existing buffer instead of allocating a
// fresh one. Currently disabled: without a uniqueness signal at the
// helper boundary we can't prove `dest` isn't aliased, so the helpers
// always allocate. The hooks remain so this can be re-enabled by the
// upcoming memory-management work without a codegen change.

/* eslint-disable @typescript-eslint/no-unused-vars */
function reuseRealBuffer(_dest: unknown, _n: number): Float64Array | undefined {
  return undefined;
}

function reuseComplexBuffers(
  _dest: unknown,
  _n: number
): { outRe: Float64Array; outIm: Float64Array } | undefined {
  return undefined;
}
/* eslint-enable @typescript-eslint/no-unused-vars */

/** Rewrap a real result: reuse dest's wrapper if we reused its buffer. */
function finalizeReal(
  out: Float64Array,
  shape: number[],
  dest: unknown,
  isLogical: boolean
): RuntimeTensor {
  if (
    typeof dest === "object" &&
    dest !== null &&
    (dest as RuntimeTensor).data === out
  ) {
    const t = dest as RuntimeTensor;
    t.shape = shape;
    t.imag = undefined;
    t._isLogical = isLogical ? true : undefined;
    return t;
  }
  const t = makeTensor(out, undefined, shape);
  if (isLogical) t._isLogical = true;
  return t;
}

/** Split-complex finalize that optionally reuses dest's wrapper. */
function finalizeSplitReused(
  outRe: Float64Array,
  outIm: Float64Array | undefined,
  shape: number[],
  dest: unknown
): RuntimeTensor {
  let realOnly = true;
  if (outIm) {
    for (let i = 0; i < outIm.length; i++) {
      if (outIm[i] !== 0) {
        realOnly = false;
        break;
      }
    }
  }
  if (
    typeof dest === "object" &&
    dest !== null &&
    (dest as RuntimeTensor).data === outRe
  ) {
    const t = dest as RuntimeTensor;
    t.shape = shape;
    t.imag = realOnly ? undefined : outIm;
    t._isLogical = undefined;
    return t;
  }
  return realOnly
    ? makeTensor(outRe, undefined, shape)
    : makeTensor(outRe, outIm, shape);
}

// ── Element-wise tensor binary operations ──────────────────────────────

type ScalarVal = number | RuntimeComplexNumber;

export function tensorBinaryOp(
  a: unknown,
  b: unknown,
  realOp: (x: number, y: number) => number,
  complexOp: (a: ScalarVal, b: ScalarVal) => ScalarVal
): RuntimeTensor | ScalarVal {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    if (at.data.length !== bt.data.length)
      throw new Error("Matrix dimensions must agree.");
    const n = at.data.length;
    const aHasImag = !!at.imag;
    const bHasImag = !!bt.imag;

    if (!aHasImag && !bHasImag) {
      const out = allocFloat64Array(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bt.data[i]);
      return makeTensor(out, undefined, at.shape.slice());
    }
    const outR = allocFloat64Array(n);
    const outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const av: ScalarVal = aHasImag
        ? mkc(at.data[i], at.imag![i])
        : at.data[i];
      const bv: ScalarVal = bHasImag
        ? mkc(bt.data[i], bt.imag![i])
        : bt.data[i];
      const r = complexOp(av, bv);
      outR[i] = re(r);
      outI[i] = im(r);
    }
    return makeTensor(outR, outI, at.shape.slice());
  }

  if (aIsT) {
    const at = a as RuntimeTensor;
    const n = at.data.length;
    const aHasImag = !!at.imag;
    const bIsC = isComplex(b);

    if (!aHasImag && !bIsC) {
      const bv = b as number;
      const out = allocFloat64Array(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bv);
      return makeTensor(out, undefined, at.shape.slice());
    }
    const outR = allocFloat64Array(n);
    const outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const av: ScalarVal = aHasImag
        ? mkc(at.data[i], at.imag![i])
        : at.data[i];
      const r = complexOp(av, b as ScalarVal);
      outR[i] = re(r);
      outI[i] = im(r);
    }
    return makeTensor(outR, outI, at.shape.slice());
  }

  if (bIsT) {
    const bt = b as RuntimeTensor;
    const n = bt.data.length;
    const bHasImag = !!bt.imag;
    const aIsC = isComplex(a);

    if (!bHasImag && !aIsC) {
      const av = a as number;
      const out = allocFloat64Array(n);
      for (let i = 0; i < n; i++) out[i] = realOp(av, bt.data[i]);
      return makeTensor(out, undefined, bt.shape.slice());
    }
    const outR = allocFloat64Array(n);
    const outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const bv: ScalarVal = bHasImag
        ? mkc(bt.data[i], bt.imag![i])
        : bt.data[i];
      const r = complexOp(a as ScalarVal, bv);
      outR[i] = re(r);
      outI[i] = im(r);
    }
    return makeTensor(outR, outI, bt.shape.slice());
  }

  throw new Error("JIT tensor binary: unexpected argument types");
}

// ── Element-wise tensor comparison ─────────────────────────────────────

export function tensorCompareOp(
  a: unknown,
  b: unknown,
  cmp: (x: number, y: number) => boolean
): RuntimeTensor {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    if (at.data.length !== bt.data.length)
      throw new Error("Matrix dimensions must agree.");
    const n = at.data.length;
    const out = allocFloat64Array(n);
    for (let i = 0; i < n; i++) out[i] = cmp(at.data[i], bt.data[i]) ? 1 : 0;
    const r = makeTensor(out, undefined, at.shape.slice());
    r._isLogical = true;
    return r;
  }
  // Coerce a scalar bool/number operand with `+` so `tensor === false` (used
  // for ==/~=) doesn't trivially return all-zero/all-one due to JS strict
  // equality treating booleans and numbers as different types.
  if (aIsT) {
    const at = a as RuntimeTensor;
    const n = at.data.length;
    const bv = +(b as number);
    const out = allocFloat64Array(n);
    for (let i = 0; i < n; i++) out[i] = cmp(at.data[i], bv) ? 1 : 0;
    const r = makeTensor(out, undefined, at.shape.slice());
    r._isLogical = true;
    return r;
  }
  if (!bIsT) {
    throw new Error(
      "JIT tensor compare: at least one operand must be a tensor"
    );
  }
  const bt = b as RuntimeTensor;
  const n = bt.data.length;
  const av = +(a as number);
  const out = allocFloat64Array(n);
  for (let i = 0; i < n; i++) out[i] = cmp(av, bt.data[i]) ? 1 : 0;
  const r = makeTensor(out, undefined, bt.shape.slice());
  r._isLogical = true;
  return r;
}

// ── Element-wise tensor unary ──────────────────────────────────────────

const UNARY_OP_CODE = new Map<(x: number) => number, number>([
  [Math.exp, 0],
  [Math.log, 1],
  [Math.log2, 2],
  [Math.log10, 3],
  [Math.sqrt, 4],
  [Math.abs, 5],
  [Math.floor, 6],
  [Math.ceil, 7],
  [Math.round, 8],
  [Math.trunc, 9],
  [Math.sin, 10],
  [Math.cos, 11],
  [Math.tan, 12],
  [Math.asin, 13],
  [Math.acos, 14],
  [Math.atan, 15],
  [Math.sinh, 16],
  [Math.cosh, 17],
  [Math.tanh, 18],
  [Math.sign, 19],
]);

const OP_ABS = 5;

export function tensorUnary(
  dest: unknown,
  a: RuntimeTensor,
  fn: (x: number) => number
): RuntimeTensor {
  const n = a.data.length;
  const opCode = UNARY_OP_CODE.get(fn);
  if (opCode !== undefined && a.data instanceof Float64Array) {
    // Real input → real output: existing fast path.
    if (!a.imag) {
      const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
      tensorOps.realUnaryElemwise(opCode, n, a.data, out);
      return finalizeReal(out, a.shape.slice(), dest, false);
    }
    // Complex input: abs → real output (complexAbs); otherwise complex
    // output (complexUnaryElemwise). Both native where available.
    if (a.imag instanceof Float64Array) {
      if (opCode === OP_ABS) {
        const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
        tensorOps.complexAbs(n, a.data, a.imag, out);
        return finalizeReal(out, a.shape.slice(), dest, false);
      }
      const reuse = reuseComplexBuffers(dest, n);
      const outRe = reuse ? reuse.outRe : allocFloat64Array(n);
      const outIm = reuse ? reuse.outIm : allocFloat64Array(n);
      tensorOps.complexUnaryElemwise(opCode, n, a.data, a.imag, outRe, outIm);
      return finalizeSplitReused(outRe, outIm, a.shape.slice(), dest);
    }
  }
  // Slow fallback: scalar closure. Only reachable for non-op-coded `fn`s
  // on real tensors (the old real-only slow path).
  const out = allocFloat64Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(a.data[i]);
  return makeTensor(out, undefined, a.shape.slice());
}

export function tensorNeg(dest: unknown, a: RuntimeTensor): RuntimeTensor {
  const n = a.data.length;
  if (!a.imag && a.data instanceof Float64Array) {
    const destBuf = reuseRealBuffer(dest, n);
    const outR = destBuf ?? allocFloat64Array(n);
    const aData = a.data;
    for (let i = 0; i < n; i++) outR[i] = -aData[i];
    return finalizeReal(outR, a.shape.slice(), dest, false);
  }
  // Complex negation: reuse split buffers when possible.
  if (a.data instanceof Float64Array && a.imag instanceof Float64Array) {
    const reuse = reuseComplexBuffers(dest, n);
    const outR = reuse ? reuse.outRe : allocFloat64Array(n);
    const aData = a.data;
    const aImag = a.imag;
    for (let i = 0; i < n; i++) outR[i] = -aData[i];
    let outI: Float64Array | undefined;
    if (aImag) {
      const outI = reuse ? reuse.outIm : allocFloat64Array(n);
      for (let i = 0; i < n; i++) outI[i] = -aImag[i];
    }
    return finalizeSplitReused(outR, outI, a.shape.slice(), dest);
  }
  const outR = allocFloat64Array(n);
  for (let i = 0; i < n; i++) outR[i] = -a.data[i];
  let outI: Float64Array | undefined;
  if (a.imag) {
    outI = allocFloat64Array(n);
    for (let i = 0; i < n; i++) outI[i] = -a.imag[i];
  }
  return makeTensor(outR, outI, a.shape.slice());
}

// ── double() fast path ────────────────────────────────────────────────
//
// `double(x)` is an identity for non-logical numeric tensors. For logical
// tensors we conservatively allocate a fresh copy with the flag cleared.

export function tDouble(v: unknown): unknown {
  if (typeof v === "number" || typeof v === "boolean") return +v;
  if (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  ) {
    const t = v as RuntimeTensor;
    if (!t._isLogical) return t;
    const newData = allocFloat64Array(t.data.length);
    newData.set(t.data as Float64Array);
    return {
      kind: "tensor",
      data: newData,
      shape: t.shape.slice(),
    } as RuntimeTensor;
  }
  return v;
}

// ── Flat-sum fast path (real vectors) ─────────────────────────────────
//
// `sum(t)` on a vector-shaped real tensor collapses to a scalar. The
// interpreter builtin handles every case (nd arrays, 'all', 'omitnan',
// dim arg, sparse) which is overkill for the hot-loop column-vector
// case. Bail to the generic path for anything non-trivial.

export function tSum(v: unknown): unknown {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  ) {
    const t = v as RuntimeTensor;
    if (t.imag) return undefined; // complex → defer
    const s = t.shape;
    const isVector =
      s.length === 1 ||
      (s.length === 2 && (s[0] === 1 || s[1] === 1)) ||
      s.length === 0;
    if (!isVector) return undefined; // matrix → defer
    const data = t.data;
    const n = data.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += data[i];
    return sum;
  }
  return undefined;
}

// ── Vertical concat growth ─────────────────────────────────────────────

export function vconcatGrow1r(base: unknown, v: number): RuntimeTensor {
  const bt = base as RuntimeTensor;
  const baseLen = bt.data.length;
  if (baseLen === 0) {
    const out = allocFloat64Array(1);
    out[0] = v;
    return makeTensor(out, undefined, [1, 1]);
  }
  const s = bt.shape;
  const isColOrScalar =
    (s.length === 2 && s[1] === 1) ||
    (s.length === 1 && s[0] >= 1) ||
    (s.length === 2 && s[0] === 1 && s[1] === 1);
  if (!isColOrScalar) {
    throw new Error(
      "Dimensions of arrays being concatenated are not consistent."
    );
  }
  const out = allocFloat64Array(baseLen + 1);
  out.set(bt.data);
  out[baseLen] = v;
  return makeTensor(out, undefined, [baseLen + 1, 1]);
}

// ── Copy-on-write unsharing ────────────────────────────────────────────

/** Unconditionally allocate a fresh wrapper + buffer for `t`. The caller
 *  can mutate the result without affecting any other alias of the input.
 *  Used by the JIT path on entry to a write-target hoisted alias; the
 *  runtime indexed-store path uses the sweep instead. */
export function unshare(t: unknown): RuntimeTensor {
  const tt = t as RuntimeTensor;
  const newData = allocFloat64Array(tt.data.length);
  newData.set(tt.data);
  let newImag: Float64Array | undefined;
  if (tt.imag) {
    newImag = allocFloat64Array(tt.imag.length);
    newImag.set(tt.imag);
  }
  return new RuntimeTensor(
    newData,
    tt.shape.slice(),
    newImag,
    tt._isLogical || undefined
  );
}

// ── Scalar → 1x1 tensor coercion ──────────────────────────────────────

export function asTensor(v: unknown): RuntimeTensor {
  if (typeof v === "number") {
    return makeTensor(allocFloat64Array([v]), undefined, [1, 1]);
  }
  if (typeof v === "boolean") {
    return makeTensor(allocFloat64Array([v ? 1 : 0]), undefined, [1, 1]);
  }
  return v as RuntimeTensor;
}

// ── Tensor binary op wrappers (re-exported via jitHelpers) ─────────────
//
// Fast path: real Float64 tensor-tensor or tensor-scalar → native tensorOps.
// Fall back to the generic JS-closure path for complex / size mismatch.

function fastBinaryOp(
  dest: unknown,
  a: unknown,
  b: unknown,
  opCode: number,
  realOp: (x: number, y: number) => number,
  complexOp: (a: ScalarVal, b: ScalarVal) => ScalarVal
): RuntimeTensor | ScalarVal {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  // ── tensor–tensor ─────────────────────────────────────────────────────
  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    if (
      at.data instanceof Float64Array &&
      bt.data instanceof Float64Array &&
      (!at.imag || at.imag instanceof Float64Array) &&
      (!bt.imag || bt.imag instanceof Float64Array) &&
      at.data.length === bt.data.length
    ) {
      const n = at.data.length;
      if (!at.imag && !bt.imag) {
        const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
        tensorOps.realBinaryElemwise(opCode, n, at.data, bt.data, out);
        return finalizeReal(out, at.shape.slice(), dest, false);
      }
      const complexReuse = reuseComplexBuffers(dest, n);
      const outRe = complexReuse ? complexReuse.outRe : allocFloat64Array(n);
      const outIm = complexReuse ? complexReuse.outIm : allocFloat64Array(n);
      tensorOps.complexBinaryElemwise(
        opCode,
        n,
        at.data,
        (at.imag as Float64Array | undefined) ?? null,
        bt.data,
        (bt.imag as Float64Array | undefined) ?? null,
        outRe,
        outIm
      );
      return finalizeSplitReused(outRe, outIm, at.shape.slice(), dest);
    }
  }
  // ── tensor–scalar (right) ─────────────────────────────────────────────
  if (aIsT && typeof b === "number") {
    const at = a as RuntimeTensor;
    if (
      at.data instanceof Float64Array &&
      (!at.imag || at.imag instanceof Float64Array)
    ) {
      const n = at.data.length;
      if (!at.imag) {
        const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
        tensorOps.realScalarBinaryElemwise(opCode, n, b, at.data, false, out);
        return finalizeReal(out, at.shape.slice(), dest, false);
      }
      const complexReuse = reuseComplexBuffers(dest, n);
      const outRe = complexReuse ? complexReuse.outRe : allocFloat64Array(n);
      const outIm = complexReuse ? complexReuse.outIm : allocFloat64Array(n);
      tensorOps.complexScalarBinaryElemwise(
        opCode,
        n,
        b,
        0,
        at.data,
        at.imag as Float64Array,
        false,
        outRe,
        outIm
      );
      return finalizeSplitReused(outRe, outIm, at.shape.slice(), dest);
    }
  }
  // ── scalar–tensor (left) ──────────────────────────────────────────────
  if (bIsT && typeof a === "number") {
    const bt = b as RuntimeTensor;
    if (
      bt.data instanceof Float64Array &&
      (!bt.imag || bt.imag instanceof Float64Array)
    ) {
      const n = bt.data.length;
      if (!bt.imag) {
        const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
        tensorOps.realScalarBinaryElemwise(opCode, n, a, bt.data, true, out);
        return finalizeReal(out, bt.shape.slice(), dest, false);
      }
      const complexReuse = reuseComplexBuffers(dest, n);
      const outRe = complexReuse ? complexReuse.outRe : allocFloat64Array(n);
      const outIm = complexReuse ? complexReuse.outIm : allocFloat64Array(n);
      tensorOps.complexScalarBinaryElemwise(
        opCode,
        n,
        a,
        0,
        bt.data,
        bt.imag as Float64Array,
        true,
        outRe,
        outIm
      );
      return finalizeSplitReused(outRe, outIm, bt.shape.slice(), dest);
    }
  }
  // ── tensor–complex scalar variants ────────────────────────────────────
  if (aIsT && isComplex(b)) {
    const at = a as RuntimeTensor;
    const bc = b as RuntimeComplexNumber;
    if (
      at.data instanceof Float64Array &&
      (!at.imag || at.imag instanceof Float64Array)
    ) {
      const n = at.data.length;
      const complexReuse = reuseComplexBuffers(dest, n);
      const outRe = complexReuse ? complexReuse.outRe : allocFloat64Array(n);
      const outIm = complexReuse ? complexReuse.outIm : allocFloat64Array(n);
      tensorOps.complexScalarBinaryElemwise(
        opCode,
        n,
        bc.re,
        bc.im,
        at.data,
        (at.imag as Float64Array | undefined) ?? null,
        false,
        outRe,
        outIm
      );
      return finalizeSplitReused(outRe, outIm, at.shape.slice(), dest);
    }
  }
  if (bIsT && isComplex(a)) {
    const bt = b as RuntimeTensor;
    const ac = a as RuntimeComplexNumber;
    if (
      bt.data instanceof Float64Array &&
      (!bt.imag || bt.imag instanceof Float64Array)
    ) {
      const n = bt.data.length;
      const complexReuse = reuseComplexBuffers(dest, n);
      const outRe = complexReuse ? complexReuse.outRe : allocFloat64Array(n);
      const outIm = complexReuse ? complexReuse.outIm : allocFloat64Array(n);
      tensorOps.complexScalarBinaryElemwise(
        opCode,
        n,
        ac.re,
        ac.im,
        bt.data,
        (bt.imag as Float64Array | undefined) ?? null,
        true,
        outRe,
        outIm
      );
      return finalizeSplitReused(outRe, outIm, bt.shape.slice(), dest);
    }
  }
  return tensorBinaryOp(a, b, realOp, complexOp);
}

export const tAdd = (dest: unknown, a: unknown, b: unknown) =>
  fastBinaryOp(dest, a, b, OpRealBin.ADD, (x, y) => x + y, cAdd);
export const tSub = (dest: unknown, a: unknown, b: unknown) =>
  fastBinaryOp(dest, a, b, OpRealBin.SUB, (x, y) => x - y, cSub);
export const tMul = (dest: unknown, a: unknown, b: unknown) =>
  fastBinaryOp(dest, a, b, OpRealBin.MUL, (x, y) => x * y, cMul);
export const tDiv = (dest: unknown, a: unknown, b: unknown) =>
  fastBinaryOp(dest, a, b, OpRealBin.DIV, (x, y) => x / y, cDiv);
export const tPow = (_dest: unknown, a: unknown, b: unknown) =>
  tensorBinaryOp(a, b, Math.pow, () => {
    throw new Error("JIT tPow: complex power not supported");
  });

function fastCompareOp(
  dest: unknown,
  a: unknown,
  b: unknown,
  opCode: number,
  cmp: (x: number, y: number) => boolean
): RuntimeTensor {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);
  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    if (
      at.data instanceof Float64Array &&
      bt.data instanceof Float64Array &&
      at.data.length === bt.data.length
    ) {
      const n = at.data.length;
      const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
      tensorOps.realComparison(opCode, n, at.data, bt.data, out);
      return finalizeReal(out, at.shape.slice(), dest, true);
    }
  }
  if (aIsT && typeof b === "number") {
    const at = a as RuntimeTensor;
    if (at.data instanceof Float64Array) {
      const n = at.data.length;
      const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
      tensorOps.realScalarComparison(opCode, n, b, at.data, false, out);
      return finalizeReal(out, at.shape.slice(), dest, true);
    }
  }
  if (bIsT && typeof a === "number") {
    const bt = b as RuntimeTensor;
    if (bt.data instanceof Float64Array) {
      const n = bt.data.length;
      const out = reuseRealBuffer(dest, n) ?? allocFloat64Array(n);
      tensorOps.realScalarComparison(opCode, n, a, bt.data, true, out);
      return finalizeReal(out, bt.shape.slice(), dest, true);
    }
  }
  return tensorCompareOp(a, b, cmp);
}

export const tEq = (dest: unknown, a: unknown, b: unknown) =>
  fastCompareOp(dest, a, b, 0, (x, y) => x === y);
export const tNeq = (dest: unknown, a: unknown, b: unknown) =>
  fastCompareOp(dest, a, b, 1, (x, y) => x !== y);
export const tLt = (dest: unknown, a: unknown, b: unknown) =>
  fastCompareOp(dest, a, b, 2, (x, y) => x < y);
export const tLe = (dest: unknown, a: unknown, b: unknown) =>
  fastCompareOp(dest, a, b, 3, (x, y) => x <= y);
export const tGt = (dest: unknown, a: unknown, b: unknown) =>
  fastCompareOp(dest, a, b, 4, (x, y) => x > y);
export const tGe = (dest: unknown, a: unknown, b: unknown) =>
  fastCompareOp(dest, a, b, 5, (x, y) => x >= y);

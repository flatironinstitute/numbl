/**
 * Runtime helpers for JIT-compiled tensor and complex operations.
 * Passed as `$h` to generated functions.
 */

import {
  FloatXArray,
  type FloatXArrayType,
  type RuntimeTensor,
  type RuntimeComplexNumber,
  type RuntimeFunction,
} from "../../runtime/types.js";

// ── Type checks ─────────────────────────────────────────────────────────

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

function re(v: unknown): number {
  if (typeof v === "number") return v;
  return (v as RuntimeComplexNumber).re;
}

function im(v: unknown): number {
  if (typeof v === "number") return 0;
  return (v as RuntimeComplexNumber).im;
}

function mkc(r: number, i: number): number | RuntimeComplexNumber {
  if (i === 0) return r;
  return { kind: "complex_number", re: r, im: i };
}

// ── Tensor creation ─────────────────────────────────────────────────────

function makeTensor(
  data: FloatXArrayType,
  imag: FloatXArrayType | undefined,
  shape: number[]
): RuntimeTensor {
  // Strip trailing singleton dimensions (always keep minimum 2D)
  const s = [...shape];
  while (s.length > 2 && s[s.length - 1] === 1) s.pop();
  const t: RuntimeTensor = { kind: "tensor", data, shape: s, _rc: 1 };
  if (imag) t.imag = imag;
  return t;
}

// ── Complex scalar operations ───────────────────────────────────────────

function cAdd(a: unknown, b: unknown): number | RuntimeComplexNumber {
  return mkc(re(a) + re(b), im(a) + im(b));
}

function cSub(a: unknown, b: unknown): number | RuntimeComplexNumber {
  return mkc(re(a) - re(b), im(a) - im(b));
}

function cMul(a: unknown, b: unknown): number | RuntimeComplexNumber {
  const ar = re(a),
    ai = im(a),
    br = re(b),
    bi = im(b);
  return mkc(ar * br - ai * bi, ar * bi + ai * br);
}

function cDiv(a: unknown, b: unknown): number | RuntimeComplexNumber {
  const ar = re(a),
    ai = im(a),
    br = re(b),
    bi = im(b);
  const d = br * br + bi * bi;
  return mkc((ar * br + ai * bi) / d, (ai * br - ar * bi) / d);
}

function cNeg(a: unknown): number | RuntimeComplexNumber {
  return mkc(-re(a), -im(a));
}

function cConj(a: unknown): number | RuntimeComplexNumber {
  return mkc(re(a), -im(a));
}

function cAngle(a: unknown): number {
  return Math.atan2(im(a), re(a));
}

function mod(a: number, b: number): number {
  if (b === 0) return a;
  return ((a % b) + b) % b;
}

// ── Element-wise tensor operations (real and complex) ───────────────────

type ScalarVal = number | RuntimeComplexNumber;

function tensorBinaryOp(
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
    const n = at.data.length;
    const aHasImag = !!at.imag;
    const bHasImag = !!bt.imag;

    if (!aHasImag && !bHasImag) {
      // Both real
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bt.data[i]);
      return makeTensor(out, undefined, at.shape.slice());
    }
    // At least one complex
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
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
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(at.data[i], bv);
      return makeTensor(out, undefined, at.shape.slice());
    }
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
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
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realOp(av, bt.data[i]);
      return makeTensor(out, undefined, bt.shape.slice());
    }
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
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

function tensorCompareOp(
  a: unknown,
  b: unknown,
  cmp: (x: number, y: number) => boolean
): RuntimeTensor {
  const aIsT = isTensor(a);
  const bIsT = isTensor(b);

  if (aIsT && bIsT) {
    const at = a as RuntimeTensor;
    const bt = b as RuntimeTensor;
    const n = at.data.length;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = cmp(at.data[i], bt.data[i]) ? 1 : 0;
    return makeTensor(out, undefined, at.shape.slice());
  }
  if (aIsT) {
    const at = a as RuntimeTensor;
    const n = at.data.length;
    const bv = b as number;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = cmp(at.data[i], bv) ? 1 : 0;
    return makeTensor(out, undefined, at.shape.slice());
  }
  const bt = b as RuntimeTensor;
  const n = bt.data.length;
  const av = a as number;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = cmp(av, bt.data[i]) ? 1 : 0;
  return makeTensor(out, undefined, bt.shape.slice());
}

function tensorUnary(
  a: RuntimeTensor,
  fn: (x: number) => number
): RuntimeTensor {
  const n = a.data.length;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = fn(a.data[i]);
  return makeTensor(out, undefined, a.shape.slice());
}

function tensorNeg(a: RuntimeTensor): RuntimeTensor {
  const n = a.data.length;
  const outR = new FloatXArray(n);
  for (let i = 0; i < n; i++) outR[i] = -a.data[i];
  let outI: FloatXArrayType | undefined;
  if (a.imag) {
    outI = new FloatXArray(n);
    for (let i = 0; i < n; i++) outI[i] = -a.imag[i];
  }
  return makeTensor(outR, outI, a.shape.slice());
}

// ── Tensor indexing ─────────────────────────────────────────────────────

function idx1(base: unknown, i: number): unknown {
  if (isTensor(base)) {
    const idx = Math.round(i) - 1;
    if (idx < 0 || idx >= base.data.length)
      throw new Error("Index exceeds array bounds");
    if (base.imag !== undefined) {
      const imVal = base.imag[idx];
      return imVal === 0 ? base.data[idx] : mkc(base.data[idx], imVal);
    }
    return base.data[idx];
  }
  // Scalar indexing: x(1) = x
  if (typeof base === "number") {
    if (Math.round(i) !== 1) throw new Error("Index exceeds array bounds");
    return base;
  }
  if (isComplex(base)) {
    if (Math.round(i) !== 1) throw new Error("Index exceeds array bounds");
    return base;
  }
  throw new Error("JIT index: unsupported base type");
}

function idx2(base: unknown, ri: number, ci: number): unknown {
  if (isTensor(base)) {
    const s = base.shape;
    const rows = s.length === 0 ? 1 : s.length === 1 ? 1 : s[0];
    const cols = s.length === 0 ? 1 : s.length === 1 ? s[0] : s[1];
    const r = Math.round(ri) - 1;
    const c = Math.round(ci) - 1;
    if (r < 0 || r >= rows || c < 0 || c >= cols)
      throw new Error("Index exceeds array bounds");
    const lin = c * rows + r;
    if (base.imag !== undefined) {
      const imVal = base.imag[lin];
      return imVal === 0 ? base.data[lin] : mkc(base.data[lin], imVal);
    }
    return base.data[lin];
  }
  throw new Error("JIT index: unsupported base type for 2D indexing");
}

function idxN(base: unknown, indices: number[]): unknown {
  if (isTensor(base)) {
    const s = base.shape;
    let lin = 0;
    let stride = 1;
    for (let k = 0; k < indices.length; k++) {
      const dimSize = k < s.length ? s[k] : 1;
      const sub = Math.round(indices[k]) - 1;
      if (sub < 0 || sub >= dimSize)
        throw new Error("Index exceeds array bounds");
      lin += sub * stride;
      stride *= dimSize;
    }
    if (base.imag !== undefined) {
      const imVal = base.imag[lin];
      return imVal === 0 ? base.data[lin] : mkc(base.data[lin], imVal);
    }
    return base.data[lin];
  }
  throw new Error("JIT index: unsupported base type for N-D indexing");
}

// ── Specialized real-tensor index helpers ───────────────────────────────
//
// These are emitted by the JIT codegen when the call site has statically
// proven the base is a real-valued tensor (no imag part). They skip:
//   * the isTensor runtime check
//   * the imag !== undefined check
//   * Math.round (callers in tight loops always pass integer indices)
//   * the array allocation per call that idxN does
//
// V8 inlines these into the caller's hot loop, so each access compiles
// down to a Float64Array load with one comparison for the bounds check.
// Bounds check uses `>>> 0` so negative indices wrap to a huge unsigned
// value which fails the `< length` test in one comparison.

function bce(): never {
  throw new Error("Index exceeds array bounds");
}

function idx1r(base: RuntimeTensor, i: number): number {
  const idx = (i - 1) | 0;
  if (idx >>> 0 >= base.data.length) bce();
  return base.data[idx];
}

function idx2r(base: RuntimeTensor, ri: number, ci: number): number {
  const s = base.shape;
  const rows = s.length >= 2 ? s[0] : s.length === 1 ? 1 : 1;
  const r = (ri - 1) | 0;
  const c = (ci - 1) | 0;
  const lin = c * rows + r;
  if (lin >>> 0 >= base.data.length) bce();
  return base.data[lin];
}

function idx3r(
  base: RuntimeTensor,
  i1: number,
  i2: number,
  i3: number
): number {
  const s = base.shape;
  const d0 = s[0];
  const d1 = s.length >= 2 ? s[1] : 1;
  const k0 = (i1 - 1) | 0;
  const k1 = (i2 - 1) | 0;
  const k2 = (i3 - 1) | 0;
  const lin = k2 * d0 * d1 + k1 * d0 + k0;
  if (lin >>> 0 >= base.data.length) bce();
  return base.data[lin];
}

// ── Hoisted-base index helpers ──────────────────────────────────────────
//
// These take the tensor's `data` Float64Array, its `length`, and its
// dimension sizes as separate scalar arguments. The JIT codegen hoists
// these reads ONCE at the top of a loop function for any tensor input
// that's not assigned within the loop body, then passes them per call.
//
// Why this is faster than the plain idx*r helpers: V8 can inline the
// helper, but each call still pays per-call property loads on `base`
// (`.data`, `.shape[0]`, `.data.length`) because the helper takes a
// generic Object parameter and V8 can't always prove `base` is the same
// object across iterations. By hoisting those four loads to local
// variables before the loop, the per-iter cost drops from a property
// chain to four register reads.
//
// Measured on stage 03 of jit-benchmarks (npts=10000, nrect=2000, ~80M
// inner-loop tensor reads): plain helper = 144ms, hoisted helper = 75ms.

function idx1r_h(data: FloatXArrayType, len: number, i: number): number {
  const idx = (i - 1) | 0;
  if (idx >>> 0 >= len) bce();
  return data[idx];
}

function idx2r_h(
  data: FloatXArrayType,
  len: number,
  rows: number,
  ri: number,
  ci: number
): number {
  const r = (ri - 1) | 0;
  const c = (ci - 1) | 0;
  const lin = c * rows + r;
  if (lin >>> 0 >= len) bce();
  return data[lin];
}

function idx3r_h(
  data: FloatXArrayType,
  len: number,
  d0: number,
  d1: number,
  i1: number,
  i2: number,
  i3: number
): number {
  const k0 = (i1 - 1) | 0;
  const k1 = (i2 - 1) | 0;
  const k2 = (i3 - 1) | 0;
  const lin = k2 * d0 * d1 + k1 * d0 + k0;
  if (lin >>> 0 >= len) bce();
  return data[lin];
}

// ── Scalar-write helpers ────────────────────────────────────────────────
//
// Mirror the hoisted read helpers. Used by the loop JIT for `t(i) = v`
// inside a hot loop, where the base tensor has already been unshared once
// at the top of the loop function (see `unshare` below), so each write is
// a direct Float64Array store with a single unsigned bounds check.

function set1r_h(
  data: FloatXArrayType,
  len: number,
  i: number,
  v: number
): void {
  const idx = (i - 1) | 0;
  if (idx >>> 0 >= len) bce();
  data[idx] = v;
}

function set2r_h(
  data: FloatXArrayType,
  len: number,
  rows: number,
  ri: number,
  ci: number,
  v: number
): void {
  const r = (ri - 1) | 0;
  const c = (ci - 1) | 0;
  const lin = c * rows + r;
  if (lin >>> 0 >= len) bce();
  data[lin] = v;
}

function set3r_h(
  data: FloatXArrayType,
  len: number,
  d0: number,
  d1: number,
  i1: number,
  i2: number,
  i3: number,
  v: number
): void {
  const k0 = (i1 - 1) | 0;
  const k1 = (i2 - 1) | 0;
  const k2 = (i3 - 1) | 0;
  const lin = k2 * d0 * d1 + k1 * d0 + k0;
  if (lin >>> 0 >= len) bce();
  data[lin] = v;
}

// ── Range slice write helper ────────────────────────────────────────────
//
// Mirror of the scalar-write helpers for the linear range pattern
//   dst(a:b) = src(c:d)
// where both dst and src are real tensors and the indices are 1-based
// MATLAB ranges. The two ranges must have the same length. Bounds checks
// use the same `>>> 0` unsigned trick as the scalar helpers.
//
// Used by the loop JIT for the chunkie grow-and-copy pattern
//   out_pt(1:nout_max) = tmp_pt(1:nout_max)
// where the dst tensor was just freshly allocated by `zeros(...)` and
// has been re-hoisted via the per-Assign refresh path.
//
// Implementation uses TypedArray.prototype.set with a subarray view of
// the source. ECMAScript guarantees TypedArray#set on a TypedArray source
// handles overlapping memory by cloning the source first, so this is
// safe even when dst and src are the same buffer.

function setRange1r_h(
  dstData: FloatXArrayType,
  dstLen: number,
  dstStart: number,
  dstEnd: number,
  srcData: FloatXArrayType,
  srcLen: number,
  srcStart: number,
  srcEnd: number
): void {
  const dStart = (dstStart - 1) | 0;
  const dEnd = (dstEnd - 1) | 0;
  const sStart = (srcStart - 1) | 0;
  const sEnd = (srcEnd - 1) | 0;
  const dN = dEnd - dStart + 1;
  const sN = sEnd - sStart + 1;
  if (dN !== sN) {
    throw new Error(
      "Unable to perform assignment because the indices on the left side are not compatible with the size of the right side."
    );
  }
  if (dN <= 0) return;
  // Bounds checks: start and end must be in-range. We check start with
  // unsigned >= len which catches negative AND too-large in one branch.
  if (dStart >>> 0 >= dstLen) bce();
  if (dEnd >>> 0 >= dstLen) bce();
  if (sStart >>> 0 >= srcLen) bce();
  if (sEnd >>> 0 >= srcLen) bce();
  dstData.set(srcData.subarray(sStart, sEnd + 1), dStart);
}

// ── Unshare (COW unsharing at loop entry) ───────────────────────────────
//
// Called once at the top of a JIT'd loop for each tensor parameter that's
// written to inside the body. Returns the same tensor if `_rc <= 1`
// (meaning no one else holds a reference, so in-place mutation is safe).
// Otherwise decrements the old tensor's `_rc` and returns a fresh copy
// with `_rc === 1`. The caller (JIT'd function) then reassigns the local
// parameter to the returned tensor and hoists `.data` / `.length` / shape
// dims from it — all subsequent writes go through the hoisted alias.
//
// This matches the `base._rc > 1` dance in `storeIntoTensor` in
// runtime/indexing.ts but pulls it out of the inner loop.

function unshare(t: unknown): RuntimeTensor {
  const tt = t as RuntimeTensor;
  if (tt._rc <= 1) return tt;
  tt._rc--;
  const newData = new FloatXArray(tt.data);
  const newImag = tt.imag ? new FloatXArray(tt.imag) : undefined;
  const copy: RuntimeTensor = {
    kind: "tensor",
    data: newData,
    shape: tt.shape.slice(),
    _rc: 1,
  };
  if (newImag) copy.imag = newImag;
  if (tt._isLogical) copy._isLogical = tt._isLogical;
  return copy;
}

// ── Complex truthiness ──────────────────────────────────────────────────

function cTruthy(v: unknown): boolean {
  if (typeof v === "number") return v !== 0;
  const c = v as RuntimeComplexNumber;
  return c.re !== 0 || c.im !== 0;
}

// ── Exported helpers object ─────────────────────────────────────────────

export const jitHelpers = {
  // Complex truthiness
  cTruthy,

  // Complex scalar ops
  cAdd,
  cSub,
  cMul,
  cDiv,
  cNeg,
  cConj,
  cAngle,

  // Scalar math
  mod,

  // Tensor binary ops (handles real + complex + mixed)
  tAdd: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x + y, cAdd),
  tSub: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x - y, cSub),
  tMul: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x * y, cMul),
  tDiv: (a: unknown, b: unknown) => tensorBinaryOp(a, b, (x, y) => x / y, cDiv),

  // Tensor power (element-wise)
  tPow: (a: unknown, b: unknown) =>
    tensorBinaryOp(a, b, Math.pow, () => {
      // Complex power not supported in JIT — should not reach here
      throw new Error("JIT tPow: complex power not supported");
    }),

  // Tensor comparisons (real only, returns logical tensor)
  tEq: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x === y),
  tNeq: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x !== y),
  tLt: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x < y),
  tLe: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x <= y),
  tGt: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x > y),
  tGe: (a: unknown, b: unknown) => tensorCompareOp(a, b, (x, y) => x >= y),

  // Tensor unary
  tNeg: tensorNeg,

  // Tensor math (real only for now)
  tSin: (a: RuntimeTensor) => tensorUnary(a, Math.sin),
  tCos: (a: RuntimeTensor) => tensorUnary(a, Math.cos),
  tTan: (a: RuntimeTensor) => tensorUnary(a, Math.tan),
  tAsin: (a: RuntimeTensor) => tensorUnary(a, Math.asin),
  tAcos: (a: RuntimeTensor) => tensorUnary(a, Math.acos),
  tAtan: (a: RuntimeTensor) => tensorUnary(a, Math.atan),
  tSinh: (a: RuntimeTensor) => tensorUnary(a, Math.sinh),
  tCosh: (a: RuntimeTensor) => tensorUnary(a, Math.cosh),
  tTanh: (a: RuntimeTensor) => tensorUnary(a, Math.tanh),
  tSqrt: (a: RuntimeTensor) => tensorUnary(a, Math.sqrt),
  tAbs: (a: RuntimeTensor) => tensorUnary(a, Math.abs),
  tFloor: (a: RuntimeTensor) => tensorUnary(a, Math.floor),
  tCeil: (a: RuntimeTensor) => tensorUnary(a, Math.ceil),
  tRound: (a: RuntimeTensor) => tensorUnary(a, Math.round),
  tFix: (a: RuntimeTensor) => tensorUnary(a, Math.trunc),
  tExp: (a: RuntimeTensor) => tensorUnary(a, Math.exp),
  tLog: (a: RuntimeTensor) => tensorUnary(a, Math.log),
  tLog2: (a: RuntimeTensor) => tensorUnary(a, Math.log2),
  tLog10: (a: RuntimeTensor) => tensorUnary(a, Math.log10),
  tSign: (a: RuntimeTensor) => tensorUnary(a, Math.sign),

  // Tensor literal construction
  mkTensor: (data: number[], shape: number[]) =>
    makeTensor(new FloatXArray(data), undefined, shape),
  mkTensorC: (reData: number[], imData: number[], shape: number[]) =>
    makeTensor(new FloatXArray(reData), new FloatXArray(imData), shape),

  // Tensor indexing (generic — handles any base type, real or complex)
  idx1,
  idx2,
  idxN,

  // Specialized real-tensor index helpers (emitted when call site can
  // prove the base is a real tensor — see emitIndex in jitCodegen.ts).
  idx1r,
  idx2r,
  idx3r,

  // Hoisted-base variants — emitted when the loop JIT has lifted the
  // tensor's .data / .length / .shape[0] / .shape[1] reads to local
  // variables at the function entry.
  idx1r_h,
  idx2r_h,
  idx3r_h,

  // Scalar-write helpers (real tensors, hoisted base). Counterparts to
  // idx{1,2,3}r_h used by the loop JIT for scalar indexed assignment.
  set1r_h,
  set2r_h,
  set3r_h,

  // Range-slice write helper (real tensors, hoisted bases for both dst
  // and src). Used by the loop JIT for `dst(a:b) = src(c:d)`.
  setRange1r_h,

  // Unshare a tensor for COW write. Returns `t` if _rc <= 1, else a fresh copy.
  unshare,

  // Scalar accessors
  re,
  im,

  // User function call with call frame tracking
  callUser: (
    rt: {
      pushCallFrame: (name: string) => void;
      popCallFrame: () => void;
      pushCleanupScope: () => void;
      popAndRunCleanups: (callFn: (fn: RuntimeFunction) => void) => void;
      dispatch: (name: string, nargout: number, args: unknown[]) => unknown;
      annotateError: (e: unknown) => void;
    },
    name: string,
    fn: (...args: unknown[]) => unknown,
    ...args: unknown[]
  ) => {
    rt.pushCallFrame(name);
    rt.pushCleanupScope();
    try {
      return fn(...args);
    } catch (e) {
      rt.annotateError(e);
      throw e;
    } finally {
      rt.popAndRunCleanups((cfn: RuntimeFunction) => {
        if (cfn.jsFn) {
          if (cfn.jsFnExpectsNargout) cfn.jsFn(0);
          else cfn.jsFn();
        } else {
          rt.dispatch(cfn.name, 0, []);
        }
      });
      rt.popCallFrame();
    }
  },

  // IBuiltin apply functions (populated by buildJitHelpers)
} as Record<string, unknown>;

import {
  buildIBuiltinHelpers,
  setDynamicRegisterHook,
} from "../builtins/index.js";
Object.assign(jitHelpers, buildIBuiltinHelpers());

// Hook: when a dynamic IBuiltin is registered, add its ib_* entry to jitHelpers
import type { IBuiltin } from "../builtins/index.js";
import { inferJitType as _ijt } from "../builtins/index.js";
setDynamicRegisterHook((b: IBuiltin) => {
  const h = jitHelpers as Record<string, unknown>;
  h[`ib_${b.name}`] = (...args: unknown[]) => {
    const pe = h._profileEnter as (...a: unknown[]) => void;
    const pl = h._profileLeave as (...a: unknown[]) => void;
    pe("builtin:jit:" + b.name);
    const rtArgs = args as import("../../runtime/types.js").RuntimeValue[];
    const argTypes = rtArgs.map(_ijt);
    const res = b.resolve(argTypes, 1);
    if (!res) {
      pl();
      throw new Error(`JIT ib_${b.name}: resolve failed`);
    }
    const result = res.apply(rtArgs, 1);
    pl();
    return typeof result === "boolean" ? (result ? 1 : 0) : result;
  };
});

/**
 * Build a per-runtime jitHelpers map for an execution that has .numbl.js
 * user functions. The result is a snapshot of the global jitHelpers
 * extended with `ib_<name>` entries (single-output fast path) for each js
 * user function and an overridden `ibcall` (multi-output) that consults
 * the js user function map first before falling back to the global registry.
 *
 * The result is built in a single object-literal expression (via spread)
 * so V8 gives the new object a fresh, stable hidden class even when the
 * source `jitHelpers` is in dictionary mode (which it is, because it gets
 * mutated by the dynamic-builtin-register hook every time a Runtime is
 * constructed). The codegen emits direct property accesses like
 * `$h.idx2r(...)` from inside hot loops, and V8 can only inline through
 * those accesses if `$h` has a stable shape — without this snapshot the
 * scalar tensor loops were ~3.5× slower.
 */
export function buildPerRuntimeJitHelpers(
  jsUserFunctions: ReadonlyMap<string, IBuiltin>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Record<string, any> {
  // Forward-declare h so the per-user-function closures below can reference
  // it; the closures only call into h at call time, by which point the
  // single object-literal assignment to h has run. ESLint can't see that
  // we're using a `let` purely for the temporal-dead-zone hoisting.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, prefer-const
  let h: Record<string, any>;

  // Build user-fn ib_ entries into a fresh container; these get spread into
  // h below in a single object-literal expression.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const userIbEntries: Record<string, any> = {};
  for (const [name, ib] of jsUserFunctions) {
    userIbEntries[`ib_${name}`] = (...args: unknown[]) => {
      const pe = h._profileEnter as (...a: unknown[]) => void;
      const pl = h._profileLeave as (...a: unknown[]) => void;
      pe("jsUserFunction:jit:" + name);
      const rtArgs = args as import("../../runtime/types.js").RuntimeValue[];
      const argTypes = rtArgs.map(_ijt);
      const res = ib.resolve(argTypes, 1);
      if (!res) {
        pl();
        throw new Error(`JIT ib_${name}: resolve failed`);
      }
      const result = res.apply(rtArgs, 1);
      pl();
      return typeof result === "boolean" ? (result ? 1 : 0) : result;
    };
  }

  // Build the multi-output `ibcall` override referencing the js-user-fn
  // map. Done as a separate value so we can include it in the same
  // object-literal expression that builds h.
  const origIbcall = (jitHelpers as Record<string, unknown>).ibcall as (
    name: unknown,
    nargout: unknown,
    ...args: unknown[]
  ) => unknown;
  const userIbcall = (
    name: unknown,
    nargout: unknown,
    ...args: unknown[]
  ): unknown => {
    const ib = jsUserFunctions.get(name as string);
    if (!ib) return origIbcall(name, nargout, ...args);
    const pe = h._profileEnter as (...a: unknown[]) => void;
    const pl = h._profileLeave as (...a: unknown[]) => void;
    pe("jsUserFunction:jit:" + (name as string));
    const rtArgs = args as import("../../runtime/types.js").RuntimeValue[];
    const argTypes = rtArgs.map(_ijt);
    const res = ib.resolve(argTypes, nargout as number);
    if (!res) {
      pl();
      throw new Error(`JIT ibcall: resolve failed for ${name}`);
    }
    const result = res.apply(rtArgs, nargout as number);
    pl();
    if (Array.isArray(result)) {
      return result.map(v => (typeof v === "boolean" ? (v ? 1 : 0) : v));
    }
    return [typeof result === "boolean" ? (result ? 1 : 0) : result];
  };

  // Single object-literal assignment — fresh hidden class, all keys present
  // up front. The `ibcall: userIbcall` field comes after the spread so it
  // overrides the spread's value for that key without changing the shape.
  h = {
    ...(jitHelpers as Record<string, unknown>),
    ...userIbEntries,
    ibcall: userIbcall,
  };

  return h;
}

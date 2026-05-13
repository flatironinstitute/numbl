/**
 * JIT runtime helpers assembly point.
 *
 * This module assembles the $h helpers object from focused sub-modules:
 * - jitHelpersComplex.ts — complex number arithmetic
 * - jitHelpersIndex.ts  — tensor indexing (generic, fast-path, hoisted)
 * - jitHelpersTensor.ts — tensor binary/unary ops, concat, COW
 *
 * The $h object is passed to every JIT-compiled function. V8 inline
 * caches require a stable hidden class, so buildPerRuntimeJitHelpers()
 * creates a fresh snapshot via a single spread expression.
 */

import {
  type RuntimeTensor,
  type RuntimeFunction,
  RuntimeStruct,
  RuntimeCell,
  type RuntimeValue,
} from "../../../runtime/types.js";
import { isShared } from "../../../runtime/refcount.js";

import {
  getTicTime,
  setTicTime,
} from "../../../interpreter/builtins/time-system.js";

import {
  mTranspose,
  mConjugateTranspose,
  mMul,
  mAdd,
  mSub,
  mElemMul,
  mElemDiv,
  mElemPow,
  mEqual,
  mNotEqual,
  mLess,
  mLessEqual,
  mGreater,
  mGreaterEqual,
} from "../../../helpers/arithmetic.js";

// Re-export sub-modules for direct import where needed
export {
  re,
  im,
  mkc,
  cAdd,
  cSub,
  cMul,
  cDiv,
  cNeg,
  cConj,
  cAngle,
  cTruthy,
} from "./jitHelpersComplex.js";

export {
  bce,
  idx1,
  idx2,
  idxN,
  idx1r,
  idx2r,
  idx3r,
  idx1r_h,
  idx2r_h,
  idx3r_h,
  set1r_h,
  set2r_h,
  set3r_h,
  setRange1r_h,
  setCol2r_h,
  subarrayCopy1r,
  subarrayCopy1rRow,
} from "./jitHelpersIndex.js";

export {
  makeTensor,
  tensorBinaryOp,
  tensorCompareOp,
  tensorUnary,
  tensorNeg,
  vconcatGrow1r,
  unshare,
  asTensor,
  tDouble,
  tSum,
  tAdd,
  tSub,
  tMul,
  tDiv,
  tPow,
  tEq,
  tNeq,
  tLt,
  tLe,
  tGt,
  tGe,
} from "./jitHelpersTensor.js";

// ── Imports for assembly ───────────────────────────────────────────────

import {
  re,
  im,
  cAdd,
  cSub,
  cMul,
  cDiv,
  cNeg,
  cConj,
  cAngle,
  cTruthy,
} from "./jitHelpersComplex.js";

import {
  idx1,
  idx2,
  idxN,
  idx1r,
  idx2r,
  idx3r,
  idx1r_h,
  idx2r_h,
  idx3r_h,
  set1r_h,
  set2r_h,
  set3r_h,
  setRange1r_h,
  setCol2r_h,
  subarrayCopy1r,
  subarrayCopy1rRow,
} from "./jitHelpersIndex.js";

import {
  makeTensor,
  tensorNeg,
  vconcatGrow1r,
  unshare,
  asTensor,
  tDouble,
  tSum,
  tAdd,
  tSub,
  tMul,
  tDiv,
  tPow,
  tEq,
  tNeq,
  tLt,
  tLe,
  tGt,
  tGe,
  tensorUnary,
} from "./jitHelpersTensor.js";

export { JitBailToInterpreter } from "./jitHelpersIndex.js";

// ── Function handle return type verification ─────────────────────────

/**
 * Thrown when a function handle called from JIT code returns a type
 * different from what the JIT expected (determined by probing at compile
 * time). The loop runner catches this and falls back to interpretation.
 */
export class JitFuncHandleBailError extends Error {
  constructor(
    public readonly fnName: string,
    public readonly expectedType: string,
    public readonly actualType: string
  ) {
    super(
      `JIT bail: function handle '${fnName}' returned '${actualType}' ` +
        `but JIT expected '${expectedType}'. Falling back to interpreter. ` +
        `(This is unusual — function handles in JIT loops are expected to ` +
        `return a consistent type.)`
    );
    this.name = "JitFuncHandleBailError";
  }
}

/** Map a runtime value to its JIT type tag string for the bail check. */
function typeTagOf(value: unknown): string {
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "number"; // MATLAB treats logical as numeric
  if (typeof value === "object" && value !== null) {
    const kind = (value as { kind?: string }).kind;
    if (kind === "tensor") return "tensor";
    if (kind === "complex_number") return "complex_or_number";
    if (kind === "struct") return "struct";
    if (kind === "cell") return "cell";
    if (kind === "char") return "char";
  }
  if (typeof value === "string") return "string";
  return "unknown";
}

// ── Scalar helpers ─────────────────────────────────────────────────────

function mod(a: number, b: number): number {
  if (b === 0) return a;
  let r = a % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return r;
}

// ── Call-frame ceremony ────────────────────────────────────────────────

/** Runtime surface used by the JIT call helpers. Kept narrow so the
 *  JIT helpers don't depend on the full Runtime class. */
interface CallRt {
  pushCallFrame: (name: string) => void;
  popCallFrame: () => void;
  pushCleanupScope: () => void;
  popAndRunCleanups: (callFn: (fn: RuntimeFunction) => void) => void;
  dispatch: (name: string, nargout: number, args: unknown[]) => unknown;
  annotateError: (e: unknown) => void;
}

/** Cleanup callback shared by every call helper that pops a frame:
 *  invoke the cleanup fn directly when it has a jsFn closure,
 *  otherwise dispatch through the runtime. */
function runCleanup(rt: CallRt, cfn: RuntimeFunction): void {
  if (cfn.jsFn) {
    if (cfn.jsFnExpectsNargout) cfn.jsFn(0);
    else cfn.jsFn();
  } else {
    rt.dispatch(cfn.name, 0, []);
  }
}

/** Run `body` inside a pushed call frame + cleanup scope, annotating
 *  any thrown error with the current call context before rethrowing.
 *  Wraps the push/try/catch/finally/pop dance shared by every helper
 *  that dispatches through the runtime. */
function withCallFrame<T>(rt: CallRt, name: string, body: () => T): T {
  rt.pushCallFrame(name);
  rt.pushCleanupScope();
  try {
    return body();
  } catch (e) {
    rt.annotateError(e);
    throw e;
  } finally {
    rt.popAndRunCleanups(cfn => runCleanup(rt, cfn));
    rt.popCallFrame();
  }
}

// ── Assembled helpers object ───────────────────────────────────────────

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

  // tic/toc: set/read the global tic timer from JIT'd code.
  __tic: (): number => {
    const t = performance.now();
    setTicTime(t);
    return t / 1000;
  },
  __toc: (): number => {
    return performance.now() / 1000 - getTicTime() / 1000;
  },

  // Scalar math
  mod,
  // MATLAB-semantics round (half-away-from-zero). Math.round rounds
  // half toward +Inf, which disagrees on negative half-values
  // (round(-1.5) is -2 in MATLAB, -1 in JS). Used by the fused JS
  // codegen; the non-fused interpreter path already routes through the
  // math.ts matlabRound helper.
  round: (x: number) => Math.sign(x) * Math.round(Math.abs(x)),

  // Tensor binary ops
  tAdd,
  tSub,
  tMul,
  tDiv,
  tPow,

  // Tensor comparisons
  tEq,
  tNeq,
  tLt,
  tLe,
  tGt,
  tGe,

  // Tensor unary
  tNeg: tensorNeg,
  tTranspose: mTranspose,
  tCTranspose: mConjugateTranspose,
  __mtimes: mMul,

  // Broadcasting-aware arithmetic (used by bsxfun folding)
  __mAdd: mAdd,
  __mSub: mSub,
  __mElemMul: mElemMul,
  __mElemDiv: mElemDiv,
  __mElemPow: mElemPow,

  // Broadcasting-aware comparisons. The same-shape JIT fast path lives
  // in $h.tensorCompareOp and rejects mismatched shapes; mismatched
  // shapes fall through here to the broadcasting helpers from the
  // interpreter's arithmetic module.
  __mEqual: mEqual,
  __mNotEqual: mNotEqual,
  __mLess: mLess,
  __mLessEqual: mLessEqual,
  __mGreater: mGreater,
  __mGreaterEqual: mGreaterEqual,

  // Tensor math (real only). `dest` is the previous value of the LHS
  // variable (or undefined); when it's a rc==1 Float64 tensor of matching
  // length, the op writes into its buffer in place.
  tSin: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.sin),
  tCos: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.cos),
  tTan: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.tan),
  tAsin: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.asin),
  tAcos: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.acos),
  tAtan: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.atan),
  tSinh: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.sinh),
  tCosh: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.cosh),
  tTanh: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.tanh),
  tSqrt: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.sqrt),
  tAbs: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.abs),
  tFloor: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.floor),
  tCeil: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.ceil),
  tRound: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.round),
  tFix: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.trunc),
  tExp: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.exp),
  tLog: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.log),
  tLog2: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.log2),
  tLog10: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.log10),
  tSign: (dest: unknown, a: RuntimeTensor) => tensorUnary(dest, a, Math.sign),

  // Fast paths for common scalar-producing / identity builtins.
  tDouble,
  tSum,

  // x(:) — column-vector linearization. Produces a fresh [N, 1] tensor
  // with the same column-major data order as the source. The source's
  // underlying buffer IS already column-major, so this is a fixed-cost
  // copy (defensive — letting the result share buffers with the source
  // would require coordinated rc bookkeeping that the helper layer
  // doesn't have access to).
  __colonAll: (base: RuntimeTensor): RuntimeTensor => {
    const N = base.data.length;
    const out = allocFloat64Array(N);
    out.set(base.data);
    let imag: typeof base.imag = undefined;
    if (base.imag) {
      imag = allocFloat64Array(N);
      imag.set(base.imag);
    }
    return makeTensor(out, imag, [N, 1]);
  },

  // [a; b] — 2-arg vertical concat. Stacks `a` above `b` along axis 0;
  // both must have matching column counts. Pure scalar concat is handled
  // by the existing TensorLiteral path; this helper covers the common
  // tensor-of-tensors case (e.g. chunkie's `r = [(xs(:)).'; (ys(:)).']`).
  __vertcat2: (a: RuntimeTensor, b: RuntimeTensor): RuntimeTensor => {
    const aRows = a.shape[0] ?? 1;
    const aCols = a.shape.length >= 2 ? a.shape[1] : a.data.length;
    const bRows = b.shape[0] ?? 1;
    const bCols = b.shape.length >= 2 ? b.shape[1] : b.data.length;
    if (aCols !== bCols) {
      throw new Error(
        `Dimensions of arrays being concatenated are not consistent.`
      );
    }
    const totalRows = aRows + bRows;
    const cols = aCols;
    const N = totalRows * cols;
    const isComplex = !!(a.imag || b.imag);
    const out = allocFloat64Array(N);
    const imag = isComplex ? allocFloat64Array(N) : undefined;
    // Each output column is contiguous in column-major storage, so use
    // TypedArray.set with subarray views to copy aRows + bRows elements
    // per column in two helper calls (V8 inlines memmove).
    for (let c = 0; c < cols; c++) {
      const dstColBase = c * totalRows;
      out.set(a.data.subarray(c * aRows, (c + 1) * aRows), dstColBase);
      out.set(b.data.subarray(c * bRows, (c + 1) * bRows), dstColBase + aRows);
      if (imag) {
        if (a.imag) {
          imag.set(a.imag.subarray(c * aRows, (c + 1) * aRows), dstColBase);
        }
        if (b.imag) {
          imag.set(
            b.imag.subarray(c * bRows, (c + 1) * bRows),
            dstColBase + aRows
          );
        }
      }
    }
    return makeTensor(out, imag, [totalRows, cols]);
  },

  // [a, b] — 2-arg horizontal concat. Concatenates `a` and `b` along
  // axis 1; both must have matching row counts.
  __horzcat2: (a: RuntimeTensor, b: RuntimeTensor): RuntimeTensor => {
    const aRows = a.shape[0] ?? 1;
    const aCols = a.shape.length >= 2 ? a.shape[1] : a.data.length;
    const bRows = b.shape[0] ?? 1;
    const bCols = b.shape.length >= 2 ? b.shape[1] : b.data.length;
    if (aRows !== bRows) {
      throw new Error(
        `Dimensions of arrays being concatenated are not consistent.`
      );
    }
    const rows = aRows;
    const totalCols = aCols + bCols;
    const N = rows * totalCols;
    const isComplex = !!(a.imag || b.imag);
    const out = allocFloat64Array(N);
    const imag = isComplex ? allocFloat64Array(N) : undefined;
    // Column-major: a's columns come first, b's columns come after.
    out.set(a.data, 0);
    out.set(b.data, aCols * rows);
    if (imag) {
      if (a.imag) imag.set(a.imag, 0);
      if (b.imag) imag.set(b.imag, aCols * rows);
    }
    return makeTensor(out, imag, [rows, totalCols]);
  },

  // Extract a row or column slice from a 2D tensor.
  // colonPos=0 → column slice (fix col, vary row): A(:, fixedIdx)
  // colonPos=1 → row slice (fix row, vary col): A(fixedIdx, :)
  // Preserves complex parts: a complex base produces a complex slice;
  // a real base produces a real slice. Without this, JIT'ing
  //   vals(:, jj)
  // on a complex `vals` (e.g. chunkie adapgausskerneval's scratch
  // buffer) is blocked at lowering time.
  __extractSlice2d: (
    base: RuntimeTensor,
    fixedIdx: number,
    colonPos: number
  ): RuntimeTensor => {
    const d = base.data;
    const dim = base.imag;
    const d0 = base.shape[0]; // number of rows (column-major stride)
    const d1 = base.shape[1]; // number of cols
    const fi = Math.round(fixedIdx) - 1; // 0-based fixed index
    if (colonPos === 0) {
      // Column slice: A(:, fi) — contiguous in column-major, length d0.
      const out = allocFloat64Array(d0);
      const offset = fi * d0;
      for (let i = 0; i < d0; i++) out[i] = d[offset + i];
      let outIm: Float64Array | undefined;
      if (dim) {
        outIm = allocFloat64Array(d0);
        for (let i = 0; i < d0; i++) outIm[i] = dim[offset + i];
      }
      return makeTensor(out, outIm, [d0, 1]);
    } else {
      // Row slice: A(fi, :) — strided in column-major, length d1.
      const out = allocFloat64Array(d1);
      for (let j = 0; j < d1; j++) out[j] = d[j * d0 + fi];
      let outIm: Float64Array | undefined;
      if (dim) {
        outIm = allocFloat64Array(d1);
        for (let j = 0; j < d1; j++) outIm[j] = dim[j * d0 + fi];
      }
      return makeTensor(out, outIm, [1, d1]);
    }
  },

  // Page-slice write into a 3D tensor: base(:, :, k) = rhs.
  // Copies rhs.data (a 2D tensor) into base.data at page offset (k-1)*d0*d1.
  // If rhs is complex but base is real, promotes base to complex in-place
  // — the caller's env/type map must have been updated to match. If rhs is
  // real but base is complex, clears the page's imag part to 0.
  __writePage3d: (
    base: RuntimeTensor,
    k: number,
    rhs: RuntimeTensor
  ): RuntimeTensor => {
    const d0 = base.shape[0];
    const d1 = base.shape[1];
    const pageSize = d0 * d1;
    const pageOffset = (Math.round(k) - 1) * pageSize;
    if (rhs.imag && !base.imag) {
      base.imag = allocFloat64Array(base.data.length);
    }
    const rdata = rhs.data;
    for (let i = 0; i < pageSize; i++) base.data[pageOffset + i] = rdata[i];
    if (base.imag) {
      if (rhs.imag) {
        const rimag = rhs.imag;
        for (let i = 0; i < pageSize; i++) base.imag[pageOffset + i] = rimag[i];
      } else {
        for (let i = 0; i < pageSize; i++) base.imag[pageOffset + i] = 0;
      }
    }
    return base;
  },

  // Tensor literal construction
  mkTensor: (data: number[], shape: number[]) =>
    makeTensor(allocFloat64Array(data), undefined, shape),
  mkTensorC: (reData: number[], imData: number[], shape: number[]) =>
    makeTensor(allocFloat64Array(reData), allocFloat64Array(imData), shape),

  // Struct construction + field write (stage 22)
  //
  // Used by AssignMember codegen to (re)initialize a variable as a
  // fresh empty struct (for the MATLAB `s = []; s.f = v` idiom or a
  // write-only local) and to set a field on an existing struct.
  structNew_h: (): RuntimeStruct => new RuntimeStruct(new Map()),
  structSetField_h: (
    rt: import("../../../runtime/refcount.js").RefcountRuntime,
    s: RuntimeStruct,
    field: string,
    value: RuntimeValue
  ): void => {
    s.bindField(rt, field, value);
  },
  // Clone a struct so subsequent mutations don't leak back to the
  // caller. Mirrors `cowCopy(struct)`: the constructor increfs every
  // field, so the new struct owns its child refs independently of the
  // source. Called once at function entry for any struct param that
  // is a target of AssignMember.
  structUnshare_h: (s: unknown): unknown => {
    if (s !== null && typeof s === "object") {
      const rs = s as RuntimeStruct;
      if (rs.kind === "struct") {
        return new RuntimeStruct(new Map(rs.fields));
      }
    }
    return s;
  },

  // Tensor indexing (generic)
  idx1,
  idx2,
  idxN,

  // Specialized real-tensor index helpers
  idx1r,
  idx2r,
  idx3r,

  // Hoisted-base variants
  idx1r_h,
  idx2r_h,
  idx3r_h,

  // Scalar-write helpers
  set1r_h,
  set2r_h,
  set3r_h,

  // Range-slice write helper
  setRange1r_h,

  // Column slice write helper
  setCol2r_h,

  // Range-slice read helpers (stage 21).
  // Column variant (default for column vectors / matrices under linear
  // indexing) and row variant (row-vector sources preserve orientation).
  subarrayCopy1r,
  subarrayCopy1rRow,

  // Vertical concat growth
  vconcatGrow1r,

  // Copy-on-write unshare
  unshare,

  // Scalar → tensor coercion
  asTensor,

  // Tensor indexing with a tensor index.
  // Supports both numeric (1-based) indices and logical masks (indexed
  // element-by-element, returning values where mask != 0).
  __tensorIndex: (base: RuntimeTensor, idx: RuntimeTensor): RuntimeTensor => {
    const bd = base.data;
    const bl = bd.length;
    if (idx._isLogical) {
      const n = idx.data.length;
      const out: number[] = [];
      for (let i = 0; i < n; i++) {
        if (idx.data[i] !== 0) {
          if (i >= bl) throw new Error("Index exceeds array bounds");
          out.push(bd[i] as number);
        }
      }
      // MATLAB convention: row vector bases stay rows, everything else
      // (column vectors, matrices, higher-dim tensors) collapses to a column.
      const isRowVector = base.shape.length === 2 && base.shape[0] === 1;
      const shape: [number, number] = isRowVector
        ? [1, out.length]
        : [out.length, 1];
      return makeTensor(allocFloat64Array(out), undefined, shape);
    }
    const n = idx.data.length;
    const result = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      const k = Math.round(idx.data[i] as number) - 1;
      if (k < 0 || k >= bl) throw new Error("Index exceeds array bounds");
      result[i] = bd[k];
    }
    return makeTensor(result, undefined, idx.shape);
  },

  // Cell array read/write
  __cellRead: (cell: unknown, idx: number): unknown => {
    const c = cell as { kind: "cell"; data: unknown[]; shape: number[] };
    const k = Math.round(idx) - 1;
    if (k < 0 || k >= c.data.length)
      throw new Error("Index exceeds cell bounds");
    return c.data[k];
  },

  __cellWrite: (
    rt: import("../../../runtime/refcount.js").RefcountRuntime,
    cell: unknown,
    idx: number,
    value: unknown
  ): unknown => {
    const orig = cell as RuntimeCell;
    const k = Math.round(idx) - 1;
    if (k < 0) throw new Error("Index exceeds cell bounds");
    // Refcount-driven COW: copy the cell only if it's shared with
    // another holder; otherwise mutate in place. `bindElement` does
    // the incref-new / decref-old bookkeeping so the new value is
    // properly tracked and the displaced slot's old value is released.
    const target = isShared(orig)
      ? new RuntimeCell(orig.data.slice(), [...orig.shape])
      : orig;
    if (k >= target.data.length) {
      // Auto-grow: append empty tensors, then update shape (row by
      // default; preserve column-vector orientation if base was one).
      const oldLen = target.data.length;
      while (target.data.length <= k) {
        const empty = makeTensor(allocFloat64Array(0), undefined, [0, 0]);
        empty.incref();
        target.data.push(empty);
      }
      const newLen = target.data.length;
      if (newLen !== oldLen) {
        const isColVec = target.shape[0] > 1 && target.shape[1] === 1;
        target.shape = isColVec ? [newLen, 1] : [1, newLen];
      }
    }
    target.bindElement(rt, k, value as RuntimeValue);
    return target;
  },

  // Horizontal concatenation with row-count validation
  __horzcat: (...args: unknown[]): unknown => {
    const parts: number[] = [];
    let nRows = -1;
    for (const a of args) {
      if (typeof a === "number") {
        if (nRows === -1) nRows = 1;
        else if (nRows !== 1)
          throw new Error(
            "Dimensions of arrays being concatenated are not consistent."
          );
        parts.push(a);
      } else if (typeof a === "boolean") {
        if (nRows === -1) nRows = 1;
        else if (nRows !== 1)
          throw new Error(
            "Dimensions of arrays being concatenated are not consistent."
          );
        parts.push(a ? 1 : 0);
      } else if (
        typeof a === "object" &&
        a !== null &&
        (a as { kind?: string }).kind === "tensor"
      ) {
        const t = a as RuntimeTensor;
        if (t.data.length === 0) continue;
        const tRows = t.shape.length >= 2 ? t.shape[0] : 1;
        if (nRows === -1) nRows = tRows;
        else if (nRows !== tRows)
          throw new Error(
            "Dimensions of arrays being concatenated are not consistent."
          );
        for (let i = 0; i < t.data.length; i++) parts.push(t.data[i] as number);
      } else {
        throw new Error("__horzcat: unsupported element type");
      }
    }
    if (parts.length === 0) {
      return makeTensor(allocFloat64Array(0), undefined, [0, 0]);
    }
    if (nRows <= 0) nRows = 1;
    const nCols = parts.length / nRows;
    return makeTensor(allocFloat64Array(parts), undefined, [nRows, nCols]);
  },

  // Scalar accessors
  re,
  im,

  // User function call with call frame tracking.
  callUser: (
    rt: CallRt,
    name: string,
    fn: (...args: unknown[]) => unknown,
    ...args: unknown[]
  ) => {
    return withCallFrame(rt, name, () => fn(...args));
  },

  // Indirect call through a function handle variable. Used by the JIT for
  // loops that call anonymous functions or named function references, e.g.
  // kern(srcinfo, targinfo) in chunkie's adapgausskerneval. The function
  // handle is passed as a JIT input variable; this helper invokes its
  // jsFn closure (or falls back to rt.dispatch for non-closure handles).
  //
  // `expectedType` is the type tag that the JIT lowering inferred at
  // compile time (via probing). After every call, the result is checked
  // against this tag. On mismatch a JitFuncHandleBailError is thrown,
  // causing the loop runner to fall back to the interpreter.
  //
  // Fast path: for handles with jsFn, skip call frame push/pop. This
  // matters in hot loops where per-call frame management dominates.
  // The slow path (dispatch) still uses full call frame tracking.
  callFuncHandle: (
    rt: CallRt,
    fn: RuntimeFunction,
    expectedType: string,
    ...args: unknown[]
  ) => {
    const result = fn.jsFn
      ? fn.jsFnExpectsNargout
        ? fn.jsFn(1, ...args)
        : fn.jsFn(...args)
      : withCallFrame(rt, fn.name, () => rt.dispatch(fn.name, 1, args));
    const actualType = typeTagOf(result);
    if (actualType !== expectedType) {
      throw new JitFuncHandleBailError(fn.name, expectedType, actualType);
    }
    return result;
  },

  // Stage 24: soft-bail user function call. Dispatches the named
  // function through the interpreter's runtime. Used when the JIT
  // couldn't lower the callee's body (tensor arithmetic, matrix
  // multiply, etc.) but the enclosing loop is otherwise JIT-friendly.
  // Mirrors callFuncHandle's slow path + return-type verification.
  //
  // The full call-frame + cleanup dance is deliberate: user functions
  // can have cleanup handlers, throw errors, and maintain persistent
  // state — all of which need frame tracking for correct semantics.
  callUserFunc: (
    rt: CallRt,
    name: string,
    expectedType: string,
    ...args: unknown[]
  ) => {
    const result = withCallFrame(rt, name, () => rt.dispatch(name, 1, args));
    const actualType = typeTagOf(result);
    if (actualType !== expectedType) {
      throw new JitFuncHandleBailError(name, expectedType, actualType);
    }
    return result;
  },

  // Multi-output function-handle dispatch: `[a, b, c] = fhandle(args...)`.
  // Returns an array of nargout values (the JIT codegen unpacks).
  // jsFnExpectsNargout is true for anonymous handles (interpreterExec
  // fixes that flag); the slow dispatch path handles named handles
  // whose jsFn isn't set up. The caller is expected to have probed
  // return types; mismatches surface as a JitFuncHandleBailError.
  callFuncHandleMulti: (
    rt: CallRt,
    fn: RuntimeFunction,
    nargout: number,
    ...args: unknown[]
  ): unknown[] => {
    const result = fn.jsFn
      ? fn.jsFnExpectsNargout
        ? fn.jsFn(nargout, ...args)
        : fn.jsFn(...args)
      : withCallFrame(rt, fn.name, () => rt.dispatch(fn.name, nargout, args));
    return padMultiResult(result, nargout);
  },

  // Soft-bail user-call with multiple outputs. Mirrors callUserFunc but
  // returns an array per nargout for JIT to unpack into LHS vars.
  callUserFuncMulti: (
    rt: CallRt,
    name: string,
    nargout: number,
    ...args: unknown[]
  ): unknown[] => {
    const result = withCallFrame(rt, name, () =>
      rt.dispatch(name, nargout, args)
    );
    return padMultiResult(result, nargout);
  },
} as Record<string, unknown>;

/** Normalize a multi-output dispatch result to a length-nargout array.
 *  Anonymous-fn jsFns return an Array directly when nargout > 1; named
 *  dispatch may return a single value when nargout==1. Pad with undef
 *  if the function under-supplied. */
function padMultiResult(result: unknown, nargout: number): unknown[] {
  if (Array.isArray(result)) {
    return result.length >= nargout
      ? result
      : [...result, ...Array(nargout - result.length).fill(undefined)];
  }
  return [result, ...Array(nargout - 1).fill(undefined)];
}

// ── IBuiltin integration ───────────────────────────────────────────────

import {
  buildIBuiltinHelpers,
  setDynamicRegisterHook,
} from "../../../interpreter/builtins/index.js";
Object.assign(jitHelpers, buildIBuiltinHelpers());

import type { IBuiltin } from "../../../interpreter/builtins/index.js";
import { inferJitType as _ijt } from "../../../interpreter/builtins/index.js";
import { allocFloat64Array } from "./alloc.js";
setDynamicRegisterHook((b: IBuiltin) => {
  const h = jitHelpers as Record<string, unknown>;
  h[`ib_${b.name}`] = (...args: unknown[]) => {
    const pe = h._profileEnter as (...a: unknown[]) => void;
    const pl = h._profileLeave as (...a: unknown[]) => void;
    pe("builtin:jit:" + b.name);
    const rtArgs = args as import("../../../runtime/types.js").RuntimeValue[];
    const argTypes = rtArgs.map(_ijt);
    const res = b.resolve(argTypes, 1);
    if (!res) {
      pl();
      throw new Error(`JIT ib_${b.name}: resolve failed`);
    }
    const result = res.apply(rtArgs, 1);
    pl();
    // Preserve JS booleans as logicals — converting to 0/1 would strip the
    // `class == "logical"` signal for builtins like `isnan`, `isempty`,
    // `logical`, `any`, `all`, etc.
    return result;
  };
});

// ── Per-runtime helpers builder ────────────────────────────────────────

/**
 * Build a per-runtime jitHelpers snapshot. Uses a single spread expression
 * for a fresh V8 hidden class (critical for inline-cache performance).
 */
export function buildPerRuntimeJitHelpers(
  jsUserFunctions: ReadonlyMap<string, IBuiltin>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Record<string, any> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, prefer-const
  let h: Record<string, any>;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const userIbEntries: Record<string, any> = {};
  for (const [name, ib] of jsUserFunctions) {
    userIbEntries[`ib_${name}`] = (...args: unknown[]) => {
      const pe = h._profileEnter as (...a: unknown[]) => void;
      const pl = h._profileLeave as (...a: unknown[]) => void;
      pe("jsUserFunction:jit:" + name);
      const rtArgs = args as import("../../../runtime/types.js").RuntimeValue[];
      const argTypes = rtArgs.map(_ijt);
      const res = ib.resolve(argTypes, 1);
      if (!res) {
        pl();
        throw new Error(`JIT ib_${name}: resolve failed`);
      }
      const result = res.apply(rtArgs, 1);
      pl();
      return result;
    };
  }

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
    const rtArgs = args as import("../../../runtime/types.js").RuntimeValue[];
    const argTypes = rtArgs.map(_ijt);
    const res = ib.resolve(argTypes, nargout as number);
    if (!res) {
      pl();
      throw new Error(`JIT ibcall: resolve failed for ${name}`);
    }
    const result = res.apply(rtArgs, nargout as number);
    pl();
    if (Array.isArray(result)) return result;
    return [result];
  };

  h = {
    ...(jitHelpers as Record<string, unknown>),
    ...userIbEntries,
    ibcall: userIbcall,
  };

  return h;
}

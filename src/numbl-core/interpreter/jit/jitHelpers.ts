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
  FloatXArray,
  type RuntimeTensor,
  type RuntimeFunction,
  type RuntimeStruct,
  type RuntimeValue,
} from "../../runtime/types.js";

import { uninitFloat64 } from "../../runtime/alloc.js";
import { getTicTime, setTicTime } from "../builtins/time-system.js";

import {
  mTranspose,
  mConjugateTranspose,
  mMul,
  mAdd,
  mSub,
  mElemMul,
  mElemDiv,
  mElemPow,
} from "../../helpers/arithmetic.js";

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
} from "./jitHelpersIndex.js";

export {
  makeTensor,
  tensorBinaryOp,
  tensorCompareOp,
  tensorUnary,
  tensorNeg,
  vconcatGrow1r,
  unshare,
  shareTensor,
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
} from "./jitHelpersIndex.js";

import {
  makeTensor,
  tensorNeg,
  vconcatGrow1r,
  unshare,
  shareTensor,
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

  // Fused-loop helpers (raw Float64Array allocation + wrapping)
  uninit: uninitFloat64,
  wrapF64: (data: Float64Array, shape: number[]) =>
    makeTensor(data, undefined, shape.slice()),

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

  // Var-to-var share: bumps _rc so the aliased tensor is no longer
  // eligible for in-place buffer reuse by later ops.
  shareTensor,

  // Fast paths for common scalar-producing / identity builtins.
  tDouble,
  tSum,

  // Extract a row or column slice from a 2D tensor as a real tensor.
  // colonPos=0 → column slice (fix col, vary row): A(:, fixedIdx)
  // colonPos=1 → row slice (fix row, vary col): A(fixedIdx, :)
  __extractSlice2d: (
    base: RuntimeTensor,
    fixedIdx: number,
    colonPos: number,
    sliceLen: number
  ): RuntimeTensor => {
    const d = base.data;
    const d0 = base.shape[0]; // number of rows (column-major stride)
    const fi = Math.round(fixedIdx) - 1; // 0-based fixed index
    const out = new FloatXArray(sliceLen);
    if (colonPos === 0) {
      // Column slice: A(:, fi) — contiguous in column-major
      const offset = fi * d0;
      for (let i = 0; i < sliceLen; i++) out[i] = d[offset + i];
      return makeTensor(out, undefined, [sliceLen, 1]);
    } else {
      // Row slice: A(fi, :) — strided in column-major
      for (let j = 0; j < sliceLen; j++) out[j] = d[j * d0 + fi];
      return makeTensor(out, undefined, [1, sliceLen]);
    }
  },

  // Tensor literal construction
  mkTensor: (data: number[], shape: number[]) =>
    makeTensor(new FloatXArray(data), undefined, shape),
  mkTensorC: (reData: number[], imData: number[], shape: number[]) =>
    makeTensor(new FloatXArray(reData), new FloatXArray(imData), shape),

  // Struct construction + field write (stage 22)
  //
  // Used by AssignMember codegen to (re)initialize a variable as a
  // fresh empty struct (for the MATLAB `s = []; s.f = v` idiom or a
  // write-only local) and to set a field on an existing struct.
  structNew_h: (): RuntimeStruct => ({
    kind: "struct",
    fields: new Map(),
  }),
  structSetField_h: (
    s: RuntimeStruct,
    field: string,
    value: RuntimeValue
  ): void => {
    s.fields.set(field, value);
  },
  // Clone a struct's fields map so subsequent mutations don't leak
  // back to the caller. Mirrors the interpreter's `setRTValueField`
  // copy-on-write semantics for struct params. Called once at function
  // entry for any struct param that is a target of AssignMember.
  structUnshare_h: (s: unknown): unknown => {
    if (s !== null && typeof s === "object") {
      const rs = s as RuntimeStruct;
      if (rs.kind === "struct") {
        return { kind: "struct", fields: new Map(rs.fields) };
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

  // Range-slice read helper (stage 21)
  subarrayCopy1r,

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
      return makeTensor(new FloatXArray(out), undefined, shape);
    }
    const n = idx.data.length;
    const result = new FloatXArray(n);
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

  __cellWrite: (cell: unknown, idx: number, value: unknown): unknown => {
    let c = cell as {
      kind: "cell";
      data: unknown[];
      shape: number[];
      _rc: number;
    };
    const k = Math.round(idx) - 1;
    if (k < 0 || k >= c.data.length)
      throw new Error("Index exceeds cell bounds");
    // COW: if the cell wrapper is shared (rc > 1, typically because it was
    // passed in as a function arg), copy before mutating so the caller's
    // cell is unaffected.
    if (c._rc > 1) {
      c._rc--;
      c = {
        kind: "cell",
        data: c.data.slice(),
        shape: c.shape.slice(),
        _rc: 1,
      };
    }
    c.data[k] = value;
    return c;
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
      return makeTensor(new FloatXArray(0), undefined, [0, 0]);
    }
    if (nRows <= 0) nRows = 1;
    const nCols = parts.length / nRows;
    return makeTensor(new FloatXArray(parts), undefined, [nRows, nCols]);
  },

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
    rt: {
      pushCallFrame: (name: string) => void;
      popCallFrame: () => void;
      pushCleanupScope: () => void;
      popAndRunCleanups: (callFn: (fn: RuntimeFunction) => void) => void;
      dispatch: (name: string, nargout: number, args: unknown[]) => unknown;
      annotateError: (e: unknown) => void;
    },
    fn: RuntimeFunction,
    expectedType: string,
    ...args: unknown[]
  ) => {
    let result: unknown;
    // Fast path: direct jsFn call without frame overhead
    if (fn.jsFn) {
      if (fn.jsFnExpectsNargout) result = fn.jsFn(1, ...args);
      else result = fn.jsFn(...args);
    } else {
      // Slow path: dispatch through runtime with full call frame tracking
      rt.pushCallFrame(fn.name);
      rt.pushCleanupScope();
      try {
        result = rt.dispatch(fn.name, 1, args);
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
    }
    // Verify the result type matches what the JIT expected
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
    rt: {
      pushCallFrame: (name: string) => void;
      popCallFrame: () => void;
      pushCleanupScope: () => void;
      popAndRunCleanups: (callFn: (fn: RuntimeFunction) => void) => void;
      dispatch: (name: string, nargout: number, args: unknown[]) => unknown;
      annotateError: (e: unknown) => void;
    },
    name: string,
    expectedType: string,
    ...args: unknown[]
  ) => {
    let result: unknown;
    rt.pushCallFrame(name);
    rt.pushCleanupScope();
    try {
      result = rt.dispatch(name, 1, args);
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
    const actualType = typeTagOf(result);
    if (actualType !== expectedType) {
      throw new JitFuncHandleBailError(name, expectedType, actualType);
    }
    return result;
  },
} as Record<string, unknown>;

// ── IBuiltin integration ───────────────────────────────────────────────

import {
  buildIBuiltinHelpers,
  setDynamicRegisterHook,
} from "../builtins/index.js";
Object.assign(jitHelpers, buildIBuiltinHelpers());

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
      const rtArgs = args as import("../../runtime/types.js").RuntimeValue[];
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
    const rtArgs = args as import("../../runtime/types.js").RuntimeValue[];
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

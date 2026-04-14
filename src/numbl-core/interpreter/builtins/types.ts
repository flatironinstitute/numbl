/**
 * IBuiltin interface, registry, and shared helpers for interpreter builtins.
 */

import type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeComplexNumber,
} from "../../runtime/types.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeTensor,
  type RuntimeChar,
} from "../../runtime/types.js";
import { tensorOps } from "../../ops/index.js";
import { RTV } from "../../runtime/constructors.js";
import { uninitFloat64, uninitFloatX } from "../../runtime/alloc.js";
import {
  type JitType,
  signFromNumber,
  isNonneg,
  unifyJitTypes,
} from "../jit/jitTypes.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";

// ── IBuiltin interface ──────────────────────────────────────────────────

export interface IBuiltinResolution {
  outputTypes: JitType[];
  apply: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[];
}

export interface BuiltinHelp {
  signatures: string[];
  description: string;
}

export interface IBuiltin {
  name: string;
  help?: BuiltinHelp;
  /** Given input JIT types + nargout, return output types and a specialized apply, or null. */
  resolve: (argTypes: JitType[], nargout: number) => IBuiltinResolution | null;
  /** Optional fast-path JS code emission for JIT. Return null to fall back to $h.ib_<name>. */
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null;
}

// ── Registry ────────────────────────────────────────────────────────────

const registry = new Map<string, IBuiltin>();

export function getIBuiltin(name: string): IBuiltin | undefined {
  return registry.get(name);
}

export function registerIBuiltin(b: IBuiltin): void {
  if (registry.has(b.name)) {
    throw new Error(`registerIBuiltin: duplicate registration for '${b.name}'`);
  }
  registry.set(b.name, b);
}

/** Callback invoked when a dynamic IBuiltin is registered, so jitHelpers can be updated. */
let _onDynamicRegister: ((b: IBuiltin) => void) | null = null;

/** Set a callback for dynamic IBuiltin registration (called by jitHelpers setup). */
export function setDynamicRegisterHook(
  hook: ((b: IBuiltin) => void) | null
): void {
  _onDynamicRegister = hook;
}

/** Register a dynamic IBuiltin (e.g. .js user functions), replacing any
 *  existing entry with the same name without error. */
export function registerDynamicIBuiltin(b: IBuiltin): void {
  registry.set(b.name, b);
  _onDynamicRegister?.(b);
}

export function unregisterIBuiltin(name: string): void {
  registry.delete(name);
}

export function getAllIBuiltinNames(): string[] {
  return Array.from(registry.keys());
}

const helpRegistry = new Map<string, BuiltinHelp>();

export function getIBuiltinHelp(name: string): BuiltinHelp | undefined {
  return registry.get(name)?.help ?? helpRegistry.get(name);
}

export function registerBuiltinHelp(name: string, help: BuiltinHelp): void {
  helpRegistry.set(name, help);
}

/** Probe an IBuiltin to determine its nargin (number of input args).
 *  Tries resolve with 1, 2, 3 number-type args to find the accepted count. */
export function getIBuiltinNargin(name: string): number | undefined {
  const ib = registry.get(name);
  if (!ib) return undefined;
  const num: JitType = { kind: "number" };
  // Find the smallest arg count that resolves
  for (let n = 1; n <= 10; n++) {
    if (ib.resolve(Array(n).fill(num), 1)) return n;
  }
  return undefined;
}

// ── Infer JitType from a runtime value ──────────────────────────────────

export function inferJitType(value: unknown): JitType {
  if (typeof value === "boolean") return { kind: "boolean", value };
  if (typeof value === "number") {
    const sign = signFromNumber(value);
    return { kind: "number", exact: value, ...(sign ? { sign } : {}) };
  }
  if (typeof value === "string") {
    return { kind: "string", value };
  }
  if (isRuntimeChar(value as RuntimeChar)) {
    return { kind: "char", value: (value as RuntimeChar).value };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "complex_number"
  ) {
    return { kind: "complex_or_number" };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "tensor"
  ) {
    const t = value as RuntimeTensor;
    const shape = t.shape.length >= 2 ? t.shape.slice() : [1, ...t.shape];
    if (t.imag) return { kind: "tensor", isComplex: true, shape };
    return {
      kind: "tensor",
      isComplex: false,
      shape,
      ...(t._isLogical ? { isLogical: true } : {}),
    };
  }
  if (isRuntimeStruct(value as RuntimeValue)) {
    const s = value as import("../../runtime/types.js").RuntimeStruct;
    const fields: Record<string, JitType> = {};
    for (const [k, v] of s.fields) {
      // Nested struct_array fields (e.g. `T.nodes`) get a real
      // `struct_array` JitType so the JIT loop lowering can chain
      // through `T.nodes(i).leaf`. Top-level struct_array values stay
      // as `unknown` below to preserve the existing dispatch behavior
      // — the old inferJitType returned unknown for struct arrays
      // and many builtin `match` functions only accept
      // struct/class_instance/unknown.
      if (isRuntimeStructArray(v)) {
        fields[k] = inferStructArrayType(
          v as import("../../runtime/types.js").RuntimeStructArray
        );
      } else {
        fields[k] = inferJitType(v);
      }
    }
    return { kind: "struct", fields };
  }
  if (isRuntimeSparseMatrix(value as RuntimeValue)) {
    const sp = value as import("../../runtime/types.js").RuntimeSparseMatrix;
    return { kind: "sparse_matrix", isComplex: !!sp.pi, m: sp.m, n: sp.n };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "cell"
  ) {
    const c = value as import("../../runtime/types.js").RuntimeCell;
    return { kind: "cell", shape: c.shape.slice() };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "dictionary"
  ) {
    return { kind: "dictionary" };
  }
  if (isRuntimeClassInstance(value as RuntimeValue)) {
    const ci = value as import("../../runtime/types.js").RuntimeClassInstance;
    const fields: Record<string, JitType> = {};
    for (const [k, v] of ci.fields) {
      fields[k] = inferJitType(v);
    }
    return {
      kind: "class_instance",
      className: ci.className,
      isHandleClass: ci.isHandleClass,
      fields,
    };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "function"
  ) {
    return { kind: "function_handle" };
  }
  return { kind: "unknown" };
}

/**
 * Infer a `struct_array` JitType from a runtime struct array. Walks
 * every element, collects per-field JIT types, and unifies them across
 * elements. Used exclusively from the `struct` branch of `inferJitType`
 * — top-level struct array values still return `{kind: "unknown"}` to
 * preserve compatibility with builtin dispatch functions that reject
 * struct_array kinds.
 */
function inferStructArrayType(
  sa: import("../../runtime/types.js").RuntimeStructArray
): JitType {
  if (sa.elements.length === 0) {
    return { kind: "struct_array", length: 0 };
  }
  const elemFields: Record<string, JitType> = {};
  const first = sa.elements[0];
  for (const [k, v] of first.fields) {
    elemFields[k] = inferJitType(v);
  }
  for (let i = 1; i < sa.elements.length; i++) {
    const el = sa.elements[i];
    for (const k of Object.keys(elemFields)) {
      const v = el.fields.get(k);
      if (v === undefined) {
        delete elemFields[k];
        continue;
      }
      const before = elemFields[k];
      const vType = inferJitType(v);
      elemFields[k] = unifyStructArrayFieldTypes(before, vType);
    }
  }
  return {
    kind: "struct_array",
    elemFields,
    length: sa.elements.length,
  };
}

/**
 * Unify two element types for a struct array field. Unlike generic
 * `unifyJitTypes`, this widens mixed real-tensor / numeric-scalar
 * combinations to `tensor[?x?]` instead of bailing to `unknown`.
 *
 * Rationale: chunkie (and many MATLAB libraries) store 1-element vector
 * fields as bare numeric scalars (e.g. `T.nodes(i).xi = 87` when a leaf
 * has a single point) while other elements hold genuine row-vector
 * tensors. Treating the whole field as `unknown` bails the JIT on
 * `T.nodes(i).xi` reads even though the pattern is semantically sound
 * — a scalar is a 1x1 tensor in MATLAB. The JIT codegen for
 * `StructArrayMemberRead` with a tensor leaf wraps the element read in
 * `$h.asTensor(...)` so a runtime scalar is promoted to a 1x1 tensor
 * before the tensor-read fast paths see it.
 */
function unifyStructArrayFieldTypes(a: JitType, b: JitType): JitType {
  // Try the normal unification first — it handles same-kind merges
  // correctly (number+number, tensor+tensor, etc.)
  const normal = unifyJitTypes(a, b);
  if (normal.kind !== "unknown") return normal;
  // Normal unify returned unknown — this happens for mixed
  // tensor-vs-scalar combinations. In MATLAB, a scalar is a 1x1
  // tensor, so we widen to tensor[?x?] when one side is a real tensor
  // and the other is a numeric scalar.
  const isRealTensor = (t: JitType) =>
    t.kind === "tensor" && t.isComplex !== true;
  const isNumScalar = (t: JitType) =>
    t.kind === "number" || t.kind === "boolean";
  if (
    (isRealTensor(a) && isNumScalar(b)) ||
    (isNumScalar(a) && isRealTensor(b))
  ) {
    return { kind: "tensor", isComplex: false };
  }
  return normal;
}

/** Coerce JS booleans to 0/1 so JIT code never sees boolean values. */
function coerceBooleans(v: unknown): unknown {
  if (typeof v === "boolean") return v ? 1 : 0;
  return v;
}

/** Build the ib_* entries for the jitHelpers object.
 *  The returned object also has _profileEnter/_profileLeave hooks (no-ops by default)
 *  that the runtime replaces when profiling is enabled. */
export function buildIBuiltinHelpers(): Record<
  string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  any
> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const helpers: Record<string, any> = {};
  // Profiling hooks — no-ops by default, replaced by Runtime when profiling is enabled
  helpers._profileEnter = Function.prototype;
  helpers._profileLeave = Function.prototype;
  for (const [name, ib] of registry) {
    helpers[`ib_${name}`] = (...args: unknown[]) => {
      helpers._profileEnter("builtin:jit:" + name);
      const rtArgs = args as RuntimeValue[];
      const argTypes = rtArgs.map(inferJitType);
      const res = ib.resolve(argTypes, 1);
      if (!res) {
        helpers._profileLeave();
        throw new Error(`JIT ib_${name}: resolve failed`);
      }
      const result = coerceBooleans(res.apply(rtArgs, 1));
      helpers._profileLeave();
      return result;
    };
  }
  // Generic multi-output caller: ibcall(name, nargout, ...args)
  helpers["ibcall"] = (name: unknown, nargout: unknown, ...args: unknown[]) => {
    helpers._profileEnter("builtin:jit:" + (name as string));
    const ib = registry.get(name as string);
    if (!ib) {
      helpers._profileLeave();
      throw new Error(`JIT ibcall: unknown builtin ${name}`);
    }
    const rtArgs = args as RuntimeValue[];
    const argTypes = rtArgs.map(inferJitType);
    const res = ib.resolve(argTypes, nargout as number);
    if (!res) {
      helpers._profileLeave();
      throw new Error(`JIT ibcall: resolve failed for ${name}`);
    }
    const result = res.apply(rtArgs, nargout as number);
    helpers._profileLeave();
    if (Array.isArray(result)) return result.map(coerceBooleans);
    return [coerceBooleans(result)];
  };
  return helpers;
}

// ── Shared helpers ──────────────────────────────────────────────────────

export function mkc(re: number, im: number): number | RuntimeComplexNumber {
  if (im === 0) return re;
  return { kind: "complex_number", re, im };
}

export function makeTensor(
  data: InstanceType<typeof FloatXArray>,
  imag: InstanceType<typeof FloatXArray> | undefined,
  shape: number[]
): RuntimeTensor {
  // Strip trailing singleton dimensions (always keep minimum 2D)
  const s = [...shape];
  while (s.length > 2 && s[s.length - 1] === 1) s.pop();
  const t: RuntimeTensor = { kind: "tensor", data, shape: s, _rc: 1 };
  if (imag) t.imag = imag;
  return t;
}

// ── Type rule helpers ───────────────────────────────────────────────────

/** Type rule requiring two scalar numbers */
export function binaryNumberOnly(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 2) return null;
  const k0 = argTypes[0].kind,
    k1 = argTypes[1].kind;
  if (
    (k0 !== "number" && k0 !== "boolean") ||
    (k1 !== "number" && k1 !== "boolean")
  )
    return null;
  return [{ kind: "number" }];
}

// ── Apply helpers ───────────────────────────────────────────────────────

/** Apply a unary element-wise function with complex support */
/** Map Math.* functions to native unaryElemwise op codes. */
const nativeUnaryOpCode = new Map<(x: number) => number, number>([
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

export function applyUnaryElemwise(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    return applyUnaryElemwise(sparseToDense(v), realFn, complexFn, name);
  if (typeof v === "boolean") return realFn(v ? 1 : 0);
  if (typeof v === "number") return realFn(v);

  if (isRuntimeComplexNumber(v)) {
    const r = complexFn(v.re, v.im);
    return mkc(r.re, r.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    if (!v.imag) {
      // Tensor-ops fast path (native if available, TS fallback otherwise).
      const opCode = nativeUnaryOpCode.get(realFn);
      if (opCode !== undefined && v.data instanceof Float64Array) {
        const out = uninitFloat64(n);
        tensorOps.realUnaryElemwise(opCode, n, v.data, out);
        return RTV.tensorRaw(out, v.shape.slice());
      }
      const out = uninitFloatX(n);
      for (let i = 0; i < n; i++) out[i] = realFn(v.data[i]);
      return makeTensor(out, undefined, v.shape.slice());
    }
    // Complex tensor path.
    const opCode = nativeUnaryOpCode.get(realFn);
    // tensorOps.complexUnaryElemwise supports all ops except ABS (op 5).
    if (
      opCode !== undefined &&
      opCode !== 5 &&
      v.data instanceof Float64Array &&
      v.imag instanceof Float64Array
    ) {
      const outRe = uninitFloat64(n);
      const outIm = uninitFloat64(n);
      tensorOps.complexUnaryElemwise(opCode, n, v.data, v.imag, outRe, outIm);
      return makeTensor(outRe, outIm, v.shape.slice());
    }
    const outR = uninitFloatX(n);
    const outI = uninitFloatX(n);
    for (let i = 0; i < n; i++) {
      const r = complexFn(v.data[i], v.imag[i]);
      outR[i] = r.re;
      outI[i] = r.im;
    }
    return makeTensor(outR, outI, v.shape.slice());
  }

  throw new Error(`${name}: unsupported argument type`);
}

/** Apply a unary element-wise function that may produce complex for out-of-domain real inputs.
 * When realFn returns NaN, complexFn is used instead (e.g., acos(2), asin(2), log(-1)). */
function applyUnaryElemwiseMaybeComplex(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string,
  nativeOpCode?: number
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    return applyUnaryElemwiseMaybeComplex(
      sparseToDense(v),
      realFn,
      complexFn,
      name,
      nativeOpCode
    );
  if (typeof v === "boolean") v = (v ? 1 : 0) as unknown as RuntimeValue;
  if (typeof v === "number") {
    const r = realFn(v);
    if (!Number.isNaN(r)) return r;
    const c = complexFn(v, 0);
    return mkc(c.re, c.im);
  }

  if (isRuntimeComplexNumber(v)) {
    const r = complexFn(v.re, v.im);
    return mkc(r.re, r.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    if (!v.imag) {
      // Tensor-ops fast path: compute the whole buffer via the ops layer,
      // then scan for NaN (indicates out-of-domain, patch with complex fallback).
      if (nativeOpCode !== undefined && v.data instanceof Float64Array) {
        const nativeOut = uninitFloat64(n);
        tensorOps.realUnaryElemwise(nativeOpCode, n, v.data, nativeOut);
        let firstNaN = -1;
        for (let i = 0; i < n; i++) {
          if (Number.isNaN(nativeOut[i])) {
            firstNaN = i;
            break;
          }
        }
        if (firstNaN === -1) {
          return RTV.tensorRaw(nativeOut, v.shape.slice());
        }
        // Patch NaN positions with the complex fallback.
        const outR = uninitFloatX(n);
        const outI = new FloatXArray(n); // must zero-init: only NaN slots get im written
        let hasImag = false;
        for (let i = 0; i < n; i++) {
          if (!Number.isNaN(nativeOut[i])) {
            outR[i] = nativeOut[i];
          } else {
            const c = complexFn(v.data[i], 0);
            outR[i] = c.re;
            outI[i] = c.im;
            if (c.im !== 0) hasImag = true;
          }
        }
        return makeTensor(outR, hasImag ? outI : undefined, v.shape.slice());
      }
      const outR = uninitFloatX(n);
      const outI = uninitFloatX(n);
      let hasImag = false;
      for (let i = 0; i < n; i++) {
        const r = realFn(v.data[i]);
        if (!Number.isNaN(r)) {
          outR[i] = r;
          outI[i] = 0;
        } else {
          const c = complexFn(v.data[i], 0);
          outR[i] = c.re;
          outI[i] = c.im;
          if (c.im !== 0) hasImag = true;
        }
      }
      return makeTensor(outR, hasImag ? outI : undefined, v.shape.slice());
    }
    const outR = uninitFloatX(n);
    const outI = uninitFloatX(n);
    for (let i = 0; i < n; i++) {
      const r = complexFn(v.data[i], v.imag[i]);
      outR[i] = r.re;
      outI[i] = r.im;
    }
    return makeTensor(outR, outI, v.shape.slice());
  }

  throw new Error(`${name}: unsupported argument type`);
}

/** Apply a unary function that always returns real (e.g., abs) */
function applyUnaryRealResult(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => number,
  name: string
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    return applyUnaryRealResult(sparseToDense(v), realFn, complexFn, name);
  if (typeof v === "boolean") return realFn(v ? 1 : 0);
  if (typeof v === "number") return realFn(v);

  if (isRuntimeComplexNumber(v)) {
    return complexFn(v.re, v.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    // Fast path for abs (realFn = Math.abs) on Float64 data.
    if (realFn === Math.abs && v.data instanceof Float64Array) {
      if (!v.imag) {
        const out = uninitFloat64(n);
        tensorOps.realUnaryElemwise(5, n, v.data, out); // OpUnary.ABS
        return RTV.tensorRaw(out, v.shape.slice());
      }
      if (v.imag instanceof Float64Array) {
        const out = uninitFloat64(n);
        tensorOps.complexAbs(n, v.data, v.imag, out);
        return RTV.tensorRaw(out, v.shape.slice());
      }
    }
    const out = uninitFloatX(n);
    if (!v.imag) {
      for (let i = 0; i < n; i++) out[i] = realFn(v.data[i]);
    } else {
      for (let i = 0; i < n; i++) out[i] = complexFn(v.data[i], v.imag[i]);
    }
    return makeTensor(out, undefined, v.shape.slice());
  }

  throw new Error(`${name}: unsupported argument type`);
}

/** Apply a binary scalar function */
export function applyBinaryScalar(
  args: RuntimeValue[],
  fn: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  const a = typeof args[0] === "boolean" ? (args[0] ? 1 : 0) : args[0];
  const b = typeof args[1] === "boolean" ? (args[1] ? 1 : 0) : args[1];
  if (typeof a === "number" && typeof b === "number") return fn(a, b);
  throw new Error(`${name}: expected two scalar numbers`);
}

/** Apply a binary element-wise function supporting scalars, tensors, and broadcast. */
export function applyBinaryElemwise(
  args: RuntimeValue[],
  fn: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  const a0 = typeof args[0] === "boolean" ? (args[0] ? 1 : 0) : args[0];
  const a1 = typeof args[1] === "boolean" ? (args[1] ? 1 : 0) : args[1];
  const aIsNum = typeof a0 === "number";
  const bIsNum = typeof a1 === "number";
  const aIsTensor = isRuntimeTensor(a0);
  const bIsTensor = isRuntimeTensor(a1);
  if (aIsNum && bIsNum) return fn(a0 as number, a1 as number);
  if (aIsTensor && bIsNum) {
    const t = a0 as RuntimeTensor;
    const s = a1 as number;
    const out = uninitFloatX(t.data.length);
    for (let i = 0; i < t.data.length; i++) out[i] = fn(t.data[i], s);
    return makeTensor(out, undefined, t.shape);
  }
  if (aIsNum && bIsTensor) {
    const s = a0 as number;
    const t = a1 as RuntimeTensor;
    const out = uninitFloatX(t.data.length);
    for (let i = 0; i < t.data.length; i++) out[i] = fn(s, t.data[i]);
    return makeTensor(out, undefined, t.shape);
  }
  if (aIsTensor && bIsTensor) {
    const tA = a0 as RuntimeTensor;
    const tB = a1 as RuntimeTensor;
    if (tA.data.length !== tB.data.length) {
      throw new Error(`${name}: array dimensions must agree`);
    }
    const out = uninitFloatX(tA.data.length);
    for (let i = 0; i < tA.data.length; i++)
      out[i] = fn(tA.data[i], tB.data[i]);
    return makeTensor(out, undefined, tA.shape);
  }
  throw new Error(`${name}: unsupported argument types`);
}

/** Match a binary function accepting number/boolean/tensor args (no complex). */
export function binaryElemwiseMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 2) return null;
  const k0 = argTypes[0].kind,
    k1 = argTypes[1].kind;
  const ok = ["number", "boolean", "tensor"];
  if (!ok.includes(k0) || !ok.includes(k1)) return null;
  if (k0 === "tensor" || k1 === "tensor") {
    return [{ kind: "tensor", isComplex: false }];
  }
  return [{ kind: "number" }];
}

// ── Combined resolve helpers ────────────────────────────────────────────

// ── Case-based builtin definition ────────────────────────────────────

/** A single case in a builtin's dispatch table. Bundles type-level matching
 *  with runtime implementation so the two paths cannot diverge. */
export interface BuiltinCase {
  /** Return output types if this case handles the given arg types, or null. */
  match: (argTypes: JitType[], nargout: number) => JitType[] | null;
  /** Runtime implementation — only called when match succeeded. */
  apply: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[];
}

/** Register an IBuiltin from a list of cases. The first matching case wins. */
export function defineBuiltin(opts: {
  name: string;
  help?: BuiltinHelp;
  cases: BuiltinCase[];
  jitEmit?: (argCode: string[], argTypes: JitType[]) => string | null;
}): void {
  registerIBuiltin({
    name: opts.name,
    help: opts.help,
    resolve: (argTypes, nargout) => {
      for (const c of opts.cases) {
        const outputTypes = c.match(argTypes, nargout);
        if (outputTypes) return { outputTypes, apply: c.apply };
      }
      return null;
    },
    jitEmit: opts.jitEmit,
  });
}

// ── Unary element-wise case builder ──────────────────────────────────

type NumberJitType = Extract<JitType, { kind: "number" }>;
type BooleanJitType = Extract<JitType, { kind: "boolean" }>;
type ComplexJitType = Extract<JitType, { kind: "complex_or_number" }>;
type TensorJitType = Extract<JitType, { kind: "tensor" }>;

interface UnaryElemwiseSpec {
  numberType?: (a: NumberJitType) => JitType | null;
  booleanType?: (a: BooleanJitType) => JitType | null;
  complexType?: (a: ComplexJitType) => JitType | null;
  tensorType?: (a: TensorJitType) => JitType | null;
  realFn: (x: number) => number;
  complexFn: (re: number, im: number) => { re: number; im: number };
  /** Use applyUnaryElemwiseMaybeComplex instead of applyUnaryElemwise */
  maybeComplex?: boolean;
  /** Opcode for the native unaryElemwise dispatch (see unary_elemwise.cpp).
   * Used by the maybeComplex path so that e.g. sqrt/log can run in native C
   * on the common all-in-domain case, then fall back for NaN entries only. */
  nativeOpCode?: number;
}

/** Generate cases for a unary element-wise builtin. Each accepted kind gets
 *  its own case that bundles the type rule with the corresponding apply. */
export function unaryElemwiseCases(
  spec: UnaryElemwiseSpec,
  name: string
): BuiltinCase[] {
  const applyFn = spec.maybeComplex
    ? (args: RuntimeValue[]) =>
        applyUnaryElemwiseMaybeComplex(
          args[0],
          spec.realFn,
          spec.complexFn,
          name,
          spec.nativeOpCode
        )
    : (args: RuntimeValue[]) =>
        applyUnaryElemwise(args[0], spec.realFn, spec.complexFn, name);

  const cases: BuiltinCase[] = [];

  // number/boolean case
  const numRule = spec.numberType ?? (() => ({ kind: "number" }) as JitType);
  const boolRule = spec.booleanType ?? (() => ({ kind: "number" }) as JitType);
  cases.push({
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      const a = argTypes[0];
      if (a.kind === "number") {
        const out = numRule(a);
        return out ? [out] : null;
      }
      if (a.kind === "boolean") {
        const out = boolRule(a);
        return out ? [out] : null;
      }
      return null;
    },
    apply: applyFn,
  });

  // complex_or_number case
  const cplxRule =
    spec.complexType ?? (() => ({ kind: "complex_or_number" }) as JitType);
  cases.push({
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      if (argTypes[0].kind !== "complex_or_number") return null;
      const out = cplxRule(argTypes[0] as ComplexJitType);
      return out ? [out] : null;
    },
    apply: applyFn,
  });

  // tensor case
  const tensorRule =
    spec.tensorType ??
    ((a: TensorJitType) =>
      ({
        kind: "tensor",
        isComplex: a.isComplex,
        shape: a.shape,
        ndim: a.ndim,
      }) as JitType);
  cases.push({
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      if (argTypes[0].kind !== "tensor") return null;
      const out = tensorRule(argTypes[0] as TensorJitType);
      return out ? [out] : null;
    },
    apply: applyFn,
  });

  // sparse_matrix case — convert to dense tensor, result is dense tensor
  cases.push({
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      if (argTypes[0].kind !== "sparse_matrix") return null;
      const sp = argTypes[0];
      const shape =
        sp.m !== undefined && sp.n !== undefined ? [sp.m, sp.n] : undefined;
      const out = tensorRule({
        kind: "tensor",
        isComplex: sp.isComplex,
        shape,
      } as TensorJitType);
      return out ? [out] : null;
    },
    apply: applyFn,
  });

  return cases;
}

/** Shorthand: build cases where all kinds return always-real output. */
export function unaryRealResultCases(
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => number,
  name: string
): BuiltinCase[] {
  const applyFn = (args: RuntimeValue[]) =>
    applyUnaryRealResult(args[0], realFn, complexFn, name);
  return [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const k = argTypes[0].kind;
        if (k === "number" || k === "boolean" || k === "complex_or_number")
          return [{ kind: "number", sign: "nonneg" as const }];
        return null;
      },
      apply: applyFn,
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (a.kind !== "tensor") return null;
        return [
          {
            kind: "tensor" as const,
            isComplex: false,
            shape: a.shape,
            ndim: a.ndim,
            nonneg: true,
          },
        ];
      },
      apply: applyFn,
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "sparse_matrix") return null;
        const sp = argTypes[0];
        return [
          {
            kind: "tensor" as const,
            isComplex: false,
            shape:
              sp.m !== undefined && sp.n !== undefined
                ? [sp.m, sp.n]
                : undefined,
            nonneg: true,
          },
        ];
      },
      apply: applyFn,
    },
  ];
}

/** Build cases for numeric predicates (isnan, isinf, isfinite) that return logical. */
export function predicateCases(
  scalarTest: (x: number) => boolean,
  complexTest: (re: number, im: number) => boolean,
  tensorTest: (x: number) => boolean,
  tensorComplexTest: (re: number, im: number) => boolean,
  name: string
): BuiltinCase[] {
  return [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const k = argTypes[0].kind;
        if (k === "number" || k === "boolean" || k === "complex_or_number")
          return [{ kind: "boolean" }];
        return null;
      },
      apply: args => {
        const v = args[0];
        if (typeof v === "boolean") return scalarTest(v ? 1 : 0);
        if (typeof v === "number") return scalarTest(v);
        if (isRuntimeComplexNumber(v)) return complexTest(v.re, v.im);
        throw new Error(`${name}: unsupported argument type`);
      },
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (a.kind !== "tensor") return null;
        return [
          {
            kind: "tensor" as const,
            isComplex: false,
            shape: a.shape,
            ndim: a.ndim,
            nonneg: true,
            isLogical: true,
          },
        ];
      },
      apply: args => {
        const v = args[0];
        if (!isRuntimeTensor(v))
          throw new Error(`${name}: unsupported argument type`);
        const n = v.data.length;
        const out = uninitFloatX(n);
        if (!v.imag) {
          for (let i = 0; i < n; i++) out[i] = tensorTest(v.data[i]) ? 1 : 0;
        } else {
          for (let i = 0; i < n; i++)
            out[i] = tensorComplexTest(v.data[i], v.imag[i]) ? 1 : 0;
        }
        const t = makeTensor(out, undefined, v.shape.slice());
        t._isLogical = true;
        return t;
      },
    },
  ];
}

// ── JIT emit helpers ───────────────────────────────────────────────────

/** Fast-path emitter for unary Math.* functions.
 *  Emits Math.fn(x) for scalar numbers, $h.tHelper(x) for real tensors. */
export function unaryMathJitEmit(
  mathFn: string,
  tensorHelper: string,
  requireNonneg?: boolean
): (argCode: string[], argTypes: JitType[]) => string | null {
  return (argCode, argTypes) => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (a.kind === "number" || a.kind === "boolean") {
      if (requireNonneg && !isNonneg(a)) return null;
      return `${mathFn}(${argCode[0]})`;
    }
    if (a.kind === "tensor" && a.isComplex !== true) {
      if (requireNonneg && !isNonneg(a)) return null;
      return `$h.${tensorHelper}(${argCode[0]})`;
    }
    return null;
  };
}

/** Fast-path emitter for binary Math.* functions on two scalar numbers. */
export function binaryMathJitEmit(
  mathFn: string
): (argCode: string[], argTypes: JitType[]) => string | null {
  return (argCode, argTypes) => {
    if (argTypes.length !== 2) return null;
    const k0 = argTypes[0].kind,
      k1 = argTypes[1].kind;
    if (
      (k0 !== "number" && k0 !== "boolean") ||
      (k1 !== "number" && k1 !== "boolean")
    )
      return null;
    return `${mathFn}(${argCode[0]}, ${argCode[1]})`;
  };
}

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
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { type JitType, signFromNumber, isNonneg } from "../jit/jitTypes.js";

// ── IBuiltin interface ──────────────────────────────────────────────────

export interface IBuiltinResolution {
  outputTypes: JitType[];
  apply: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[];
}

export interface IBuiltin {
  name: string;
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
  registry.set(b.name, b);
}

// ── Infer JitType from a runtime value ──────────────────────────────────

export function inferJitType(value: unknown): JitType {
  if (typeof value === "boolean") return { kind: "boolean", value };
  if (typeof value === "number") {
    const sign = signFromNumber(value);
    return { kind: "number", exact: value, ...(sign ? { sign } : {}) };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "complex_number"
  ) {
    return { kind: "complex" };
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
  return { kind: "unknown" };
}

/** Coerce JS booleans to 0/1 so JIT code never sees boolean values. */
function coerceBooleans(v: unknown): unknown {
  if (typeof v === "boolean") return v ? 1 : 0;
  return v;
}

/** Build the ib_* entries for the jitHelpers object */
export function buildIBuiltinHelpers(): Record<
  string,
  (...args: unknown[]) => unknown
> {
  const helpers: Record<string, (...args: unknown[]) => unknown> = {};
  for (const [name, ib] of registry) {
    helpers[`ib_${name}`] = (...args: unknown[]) => {
      const rtArgs = args as RuntimeValue[];
      const argTypes = rtArgs.map(inferJitType);
      const res = ib.resolve(argTypes, 1);
      if (!res) throw new Error(`JIT ib_${name}: resolve failed`);
      return coerceBooleans(res.apply(rtArgs, 1));
    };
  }
  // Generic multi-output caller: ibcall(name, nargout, ...args)
  helpers["ibcall"] = (name: unknown, nargout: unknown, ...args: unknown[]) => {
    const ib = registry.get(name as string);
    if (!ib) throw new Error(`JIT ibcall: unknown builtin ${name}`);
    const rtArgs = args as RuntimeValue[];
    const argTypes = rtArgs.map(inferJitType);
    const res = ib.resolve(argTypes, nargout as number);
    if (!res) throw new Error(`JIT ibcall: resolve failed for ${name}`);
    const result = res.apply(rtArgs, nargout as number);
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
  const t: RuntimeTensor = { kind: "tensor", data, shape, _rc: 1 };
  if (imag) t.imag = imag;
  return t;
}

// ── Type rule helpers ───────────────────────────────────────────────────

/** Type rule for unary functions that preserve the type category */
export function unaryPreserveType(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 1) return null;
  const a = argTypes[0];
  switch (a.kind) {
    case "number":
    case "boolean":
      return [{ kind: "number" }];
    case "complex":
      return [{ kind: "complex" }];
    case "tensor":
      return [
        {
          kind: "tensor",
          isComplex: a.isComplex,
          shape: a.shape,
          ndim: a.ndim,
        },
      ];
    default:
      return null;
  }
}

/** Type rule for unary functions that always return real (e.g., abs) */
export function unaryAlwaysReal(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 1) return null;
  const a = argTypes[0];
  switch (a.kind) {
    case "number":
    case "boolean":
      return [{ kind: "number", sign: "nonneg" }];
    case "complex":
      return [{ kind: "number", sign: "nonneg" }];
    case "tensor":
      return [
        {
          kind: "tensor",
          isComplex: false,
          shape: a.shape,
          ndim: a.ndim,
          nonneg: true,
        },
      ];
    default:
      return null;
  }
}

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
export function applyUnaryElemwise(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string
): RuntimeValue {
  if (typeof v === "number") return realFn(v);

  if (isRuntimeComplexNumber(v)) {
    const r = complexFn(v.re, v.im);
    return mkc(r.re, r.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    if (!v.imag) {
      const out = new FloatXArray(n);
      for (let i = 0; i < n; i++) out[i] = realFn(v.data[i]);
      return makeTensor(out, undefined, v.shape.slice());
    }
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
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
export function applyUnaryElemwiseMaybeComplex(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string
): RuntimeValue {
  if (isRuntimeSparseMatrix(v))
    throw new Error(`${name}: sparse matrices not yet supported`);
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
      const outR = new FloatXArray(n);
      const outI = new FloatXArray(n);
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
    const outR = new FloatXArray(n);
    const outI = new FloatXArray(n);
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
export function applyUnaryRealResult(
  v: RuntimeValue,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => number,
  name: string
): RuntimeValue {
  if (typeof v === "number") return realFn(v);

  if (isRuntimeComplexNumber(v)) {
    return complexFn(v.re, v.im);
  }

  if (isRuntimeTensor(v)) {
    const n = v.data.length;
    const out = new FloatXArray(n);
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
  const a = args[0],
    b = args[1];
  if (typeof a === "number" && typeof b === "number") return fn(a, b);
  throw new Error(`${name}: expected two scalar numbers`);
}

// ── Combined resolve helpers ────────────────────────────────────────────

/** Resolve helper for unary element-wise functions with complex support. */
export function resolveUnaryElemwise(
  typeRule: (argTypes: JitType[]) => JitType[] | null,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string
): (argTypes: JitType[], nargout: number) => IBuiltinResolution | null {
  return argTypes => {
    const outputTypes = typeRule(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => applyUnaryElemwise(args[0], realFn, complexFn, name),
    };
  };
}

/** Resolve helper for unary element-wise functions that may produce complex. */
export function resolveUnaryElemwiseMaybeComplex(
  typeRule: (argTypes: JitType[]) => JitType[] | null,
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => { re: number; im: number },
  name: string
): (argTypes: JitType[], nargout: number) => IBuiltinResolution | null {
  return argTypes => {
    const outputTypes = typeRule(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args =>
        applyUnaryElemwiseMaybeComplex(args[0], realFn, complexFn, name),
    };
  };
}

/** Resolve helper for unary functions that always return real. */
export function resolveUnaryRealResult(
  realFn: (x: number) => number,
  complexFn: (re: number, im: number) => number,
  name: string
): (argTypes: JitType[], nargout: number) => IBuiltinResolution | null {
  return argTypes => {
    const outputTypes = unaryAlwaysReal(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => applyUnaryRealResult(args[0], realFn, complexFn, name),
    };
  };
}

/** Resolve helper for binary scalar-only functions. */
export function resolveBinaryScalar(
  fn: (a: number, b: number) => number,
  name: string
): (argTypes: JitType[], nargout: number) => IBuiltinResolution | null {
  return argTypes => {
    const outputTypes = binaryNumberOnly(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => applyBinaryScalar(args, fn, name),
    };
  };
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

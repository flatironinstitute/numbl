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
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { JitType } from "../jit/jitTypes.js";

// ── IBuiltin interface ──────────────────────────────────────────────────

export interface IBuiltin {
  name: string;
  /** Given input JIT types + nargout, return output JIT types or null if can't handle */
  typeRule: (argTypes: JitType[], nargout: number) => JitType[] | null;
  /** Apply the function */
  apply: (args: RuntimeValue[], nargout: number) => RuntimeValue;
}

// ── Registry ────────────────────────────────────────────────────────────

const registry = new Map<string, IBuiltin>();

export function getIBuiltin(name: string): IBuiltin | undefined {
  return registry.get(name);
}

export function registerIBuiltin(b: IBuiltin): void {
  registry.set(b.name, b);
}

/** Build the ib_* entries for the jitHelpers object */
export function buildIBuiltinHelpers(): Record<
  string,
  (...args: unknown[]) => unknown
> {
  const helpers: Record<string, (...args: unknown[]) => unknown> = {};
  for (const [name, ib] of registry) {
    helpers[`ib_${name}`] = (...args: unknown[]) =>
      ib.apply(args as RuntimeValue[], 1);
  }
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
      return [{ kind: "number" }];
    case "complex":
      return [{ kind: "complex" }];
    case "realTensor":
      return [{ kind: "realTensor" }];
    case "complexTensor":
      return [{ kind: "complexTensor" }];
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
      return [{ kind: "number", nonneg: true }];
    case "complex":
      return [{ kind: "number", nonneg: true }];
    case "realTensor":
      return [{ kind: "realTensor", nonneg: true }];
    case "complexTensor":
      return [{ kind: "realTensor", nonneg: true }];
    default:
      return null;
  }
}

/** Type rule requiring two scalar numbers */
export function binaryNumberOnly(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 2) return null;
  if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
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

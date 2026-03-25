/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * Binary scalar builtins: atan2, min, max, mod, rem, power.
 */

import {
  FloatXArray,
  isRuntimeTensor,
  type RuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { minMaxImpl } from "../../builtins/reduction/min-max.js";
import {
  type IBuiltinResolution,
  registerIBuiltin,
  makeTensor,
  binaryMathJitEmit,
} from "./types.js";
import {
  type JitType,
  shapeAfterReduction,
  unifySign,
} from "../jit/jitTypes.js";

// ── Type rule helpers ─────────────────────────────────────────────────

/** Type rule for binary real functions that accept number/logical or real tensor args. */
function binaryRealElemwise(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 2) return null;
  const a = argTypes[0];
  const b = argTypes[1];
  if (
    a.kind !== "number" &&
    a.kind !== "boolean" &&
    !(a.kind === "tensor" && a.isComplex === false)
  )
    return null;
  if (
    b.kind !== "number" &&
    b.kind !== "boolean" &&
    !(b.kind === "tensor" && b.isComplex === false)
  )
    return null;
  if (a.kind === "tensor" || b.kind === "tensor") {
    const t =
      a.kind === "tensor" ? a : (b as Extract<JitType, { kind: "tensor" }>);
    return [{ kind: "tensor", isComplex: false, shape: t.shape, ndim: t.ndim }];
  }
  return [{ kind: "number" }];
}

// ── Tensor-capable binary helper ─────────────────────────────────────────

/** Apply a binary function element-wise over two tensors/scalars/mixed. */
function applyBinaryElemwise(
  args: RuntimeValue[],
  fn: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  const a = typeof args[0] === "boolean" ? (args[0] ? 1 : 0) : args[0];
  const b = typeof args[1] === "boolean" ? (args[1] ? 1 : 0) : args[1];
  if (typeof a === "number" && typeof b === "number") return fn(a, b);

  // scalar + tensor or tensor + scalar or tensor + tensor
  const aIsNum = typeof a === "number";
  const bIsNum = typeof b === "number";
  const aTensor = aIsNum ? null : (a as RuntimeTensor);
  const bTensor = bIsNum ? null : (b as RuntimeTensor);

  if (!aIsNum && !isRuntimeTensor(a))
    throw new Error(`${name}: unsupported argument type`);
  if (!bIsNum && !isRuntimeTensor(b))
    throw new Error(`${name}: unsupported argument type`);

  if (aIsNum && bTensor) {
    const n = bTensor.data.length;
    const out = new FloatXArray(n);
    const av = a as number;
    for (let i = 0; i < n; i++) out[i] = fn(av, bTensor.data[i]);
    return makeTensor(out, undefined, bTensor.shape.slice());
  }
  if (aTensor && bIsNum) {
    const n = aTensor.data.length;
    const out = new FloatXArray(n);
    const bv = b as number;
    for (let i = 0; i < n; i++) out[i] = fn(aTensor.data[i], bv);
    return makeTensor(out, undefined, aTensor.shape.slice());
  }
  if (aTensor && bTensor) {
    // Same shape required (broadcasting not attempted here)
    const n = aTensor.data.length;
    if (n !== bTensor.data.length) throw new Error(`${name}: size mismatch`);
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = fn(aTensor.data[i], bTensor.data[i]);
    return makeTensor(out, undefined, aTensor.shape.slice());
  }
  throw new Error(`${name}: unsupported argument types`);
}

/** Resolve helper for binary real element-wise functions. */
function resolveBinaryRealElemwise(
  fn: (a: number, b: number) => number,
  name: string
): (argTypes: JitType[], nargout: number) => IBuiltinResolution | null {
  return argTypes => {
    const outputTypes = binaryRealElemwise(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => applyBinaryElemwise(args, fn, name),
    };
  };
}

// ── atan2 ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "atan2",
  resolve: resolveBinaryRealElemwise(Math.atan2, "atan2"),
  jitEmit: binaryMathJitEmit("Math.atan2"),
});

// ── min ──────────────────────────────────────────────────────────────────

function minMaxTypeRule(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length === 2) {
    const a = argTypes[0];
    const b = argTypes[1];
    if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
      return null;
    if (b.kind !== "number" && b.kind !== "boolean" && b.kind !== "tensor")
      return null;
    if (a.kind === "tensor" || b.kind === "tensor") {
      const t =
        a.kind === "tensor" ? a : (b as Extract<JitType, { kind: "tensor" }>);
      return [
        {
          kind: "tensor",
          isComplex: t.isComplex,
          shape: t.shape,
          ndim: t.ndim,
        },
      ];
    }
    {
      const aSign =
        a.kind === "number"
          ? a.sign
          : a.kind === "boolean"
            ? ("nonneg" as const)
            : undefined;
      const bSign =
        b.kind === "number"
          ? b.sign
          : b.kind === "boolean"
            ? ("nonneg" as const)
            : undefined;
      const sign = unifySign(aSign, bSign);
      return [{ kind: "number", ...(sign ? { sign } : {}) }];
    }
  }
  if (argTypes.length === 1) {
    const a = argTypes[0];
    if (a.kind === "number" || a.kind === "boolean" || a.kind === "complex")
      return [a];
    if (a.kind === "tensor") {
      if (!a.shape)
        return [
          a.isComplex === true ? { kind: "complex" } : { kind: "number" },
        ];
      const result = shapeAfterReduction(a.shape);
      if (result.scalar)
        return [
          a.isComplex === true ? { kind: "complex" } : { kind: "number" },
        ];
      return [{ kind: "tensor", isComplex: a.isComplex, shape: result.shape }];
    }
  }
  // 3-arg: min(X, [], dim) — second arg is always empty, third is dim
  if (argTypes.length === 3) {
    const a = argTypes[0];
    if (a.kind === "tensor") {
      // Try to compute reduced shape if dim is known
      const dimType = argTypes[2];
      const dim =
        dimType.kind === "number" && dimType.exact !== undefined
          ? dimType.exact
          : undefined;
      if (a.shape && dim !== undefined && dim >= 1 && dim <= a.shape.length) {
        const result = shapeAfterReduction(a.shape, dim);
        if (result.scalar)
          return [
            a.isComplex === true ? { kind: "complex" } : { kind: "number" },
          ];
        return [
          { kind: "tensor", isComplex: a.isComplex, shape: result.shape },
        ];
      }
      // Unknown dim or shape: return tensor with unknown shape
      return [
        {
          kind: "tensor",
          isComplex: a.isComplex,
          ndim: a.ndim,
        },
      ];
    }
    if (a.kind === "number" || a.kind === "boolean")
      return [{ kind: "number" }];
  }
  return null;
}

registerIBuiltin({
  name: "min",
  resolve: (argTypes, _nargout) => {
    const outputTypes = minMaxTypeRule(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: (args, nargout) =>
        minMaxImpl("min", args, nargout, Infinity, (a, b) => a < b, Math.min),
    };
  },
});

// ── max ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "max",
  resolve: (argTypes, _nargout) => {
    const outputTypes = minMaxTypeRule(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: (args, nargout) =>
        minMaxImpl("max", args, nargout, -Infinity, (a, b) => a > b, Math.max),
    };
  },
});

// ── mod ──────────────────────────────────────────────────────────────────

function modFn(a: number, b: number): number {
  return ((a % b) + b) % b;
}

registerIBuiltin({
  name: "mod",
  resolve: resolveBinaryRealElemwise(modFn, "mod"),
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 2) return null;
    const k0 = argTypes[0].kind,
      k1 = argTypes[1].kind;
    if (
      (k0 !== "number" && k0 !== "boolean") ||
      (k1 !== "number" && k1 !== "boolean")
    )
      return null;
    return `$h.mod(${argCode[0]}, ${argCode[1]})`;
  },
});

// ── rem ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "rem",
  resolve: resolveBinaryRealElemwise((a, b) => a % b, "rem"),
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 2) return null;
    const k0 = argTypes[0].kind,
      k1 = argTypes[1].kind;
    if (
      (k0 !== "number" && k0 !== "boolean") ||
      (k1 !== "number" && k1 !== "boolean")
    )
      return null;
    return `(${argCode[0]} % ${argCode[1]})`;
  },
});

// ── power ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "power",
  resolve: resolveBinaryRealElemwise(Math.pow, "power"),
  jitEmit: binaryMathJitEmit("Math.pow"),
});

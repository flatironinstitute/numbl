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
  registerIBuiltin,
  binaryNumberOnly,
  applyBinaryScalar,
  makeTensor,
} from "./types.js";

// ── Tensor-capable binary helper ─────────────────────────────────────────

/** Apply a binary function element-wise over two tensors/scalars/mixed. */
function applyBinaryElemwise(
  args: RuntimeValue[],
  fn: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  const [a, b] = args;
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

// ── atan2 ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "atan2",
  typeRule: argTypes => binaryNumberOnly(argTypes),
  apply: args => applyBinaryScalar(args, Math.atan2, "atan2"),
});

// ── min ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "min",
  typeRule: (argTypes, _nargout) => {
    if (argTypes.length === 2) {
      if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
        return null;
      return [
        {
          kind: "number",
          nonneg: !!argTypes[0].nonneg && !!argTypes[1].nonneg,
        },
      ];
    }
    if (argTypes.length === 1) {
      const a = argTypes[0];
      if (a.kind === "number" || a.kind === "complex") return [a];
      if (a.kind === "realTensor") return [{ kind: "number" }];
      if (a.kind === "complexTensor") return [{ kind: "complex" }];
    }
    return null;
  },
  apply: (args, nargout) =>
    minMaxImpl("min", args, nargout, Infinity, (a, b) => a < b, Math.min),
});

// ── max ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "max",
  typeRule: (argTypes, _nargout) => {
    if (argTypes.length === 2) {
      if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
        return null;
      return [
        {
          kind: "number",
          nonneg: !!argTypes[0].nonneg && !!argTypes[1].nonneg,
        },
      ];
    }
    if (argTypes.length === 1) {
      const a = argTypes[0];
      if (a.kind === "number" || a.kind === "complex") return [a];
      if (a.kind === "realTensor") return [{ kind: "number" }];
      if (a.kind === "complexTensor") return [{ kind: "complex" }];
    }
    return null;
  },
  apply: (args, nargout) =>
    minMaxImpl("max", args, nargout, -Infinity, (a, b) => a > b, Math.max),
});

// ── mod ──────────────────────────────────────────────────────────────────

function modFn(a: number, b: number): number {
  return ((a % b) + b) % b;
}

registerIBuiltin({
  name: "mod",
  typeRule: argTypes => {
    if (argTypes.length !== 2) return null;
    if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
      return null;
    return [{ kind: "number", nonneg: !!argTypes[1].nonneg }];
  },
  apply: args => applyBinaryElemwise(args, modFn, "mod"),
});

// ── rem ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "rem",
  typeRule: argTypes => binaryNumberOnly(argTypes),
  apply: args => applyBinaryElemwise(args, (a, b) => a % b, "rem"),
});

// ── power ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "power",
  typeRule: argTypes => binaryNumberOnly(argTypes),
  apply: args => applyBinaryScalar(args, Math.pow, "power"),
});

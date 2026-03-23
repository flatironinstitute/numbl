/**
 * Runtime helpers for JIT-compiled tensor operations.
 * Passed as `$h` to generated functions.
 */

import {
  FloatXArray,
  type FloatXArrayType,
  type RuntimeTensor,
} from "../../runtime/types.js";

function makeTensor(data: FloatXArrayType, shape: number[]): RuntimeTensor {
  return { kind: "tensor", data, shape, _rc: 1 };
}

function isTensor(v: unknown): v is RuntimeTensor {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).kind === "tensor"
  );
}

// ── Element-wise binary operations ──────────────────────────────────────

function tensorBinary(
  a: unknown,
  b: unknown,
  op: (x: number, y: number) => number
): RuntimeTensor {
  if (isTensor(a) && isTensor(b)) {
    const n = a.data.length;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = op(a.data[i], b.data[i]);
    return makeTensor(out, a.shape.slice());
  }
  if (isTensor(a) && typeof b === "number") {
    const n = a.data.length;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = op(a.data[i], b);
    return makeTensor(out, a.shape.slice());
  }
  if (typeof a === "number" && isTensor(b)) {
    const n = b.data.length;
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = op(a, b.data[i]);
    return makeTensor(out, b.shape.slice());
  }
  throw new Error("JIT tensor binary: unexpected argument types");
}

// ── Element-wise unary math ─────────────────────────────────────────────

function tensorUnary(
  a: RuntimeTensor,
  fn: (x: number) => number
): RuntimeTensor {
  const n = a.data.length;
  const out = new FloatXArray(n);
  for (let i = 0; i < n; i++) out[i] = fn(a.data[i]);
  return makeTensor(out, a.shape.slice());
}

// ── Exported helpers object ─────────────────────────────────────────────

export const jitHelpers = {
  // Binary ops
  tAdd: (a: unknown, b: unknown) => tensorBinary(a, b, (x, y) => x + y),
  tSub: (a: unknown, b: unknown) => tensorBinary(a, b, (x, y) => x - y),
  tMul: (a: unknown, b: unknown) => tensorBinary(a, b, (x, y) => x * y),
  tDiv: (a: unknown, b: unknown) => tensorBinary(a, b, (x, y) => x / y),

  // Unary
  tNeg: (a: RuntimeTensor) => tensorUnary(a, x => -x),

  // Math
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
  tFix: (a: RuntimeTensor) => tensorUnary(a, x => x | 0),
  tExp: (a: RuntimeTensor) => tensorUnary(a, Math.exp),
  tLog: (a: RuntimeTensor) => tensorUnary(a, Math.log),
  tLog2: (a: RuntimeTensor) => tensorUnary(a, Math.log2),
  tLog10: (a: RuntimeTensor) => tensorUnary(a, Math.log10),
  tSign: (a: RuntimeTensor) => tensorUnary(a, Math.sign),
};

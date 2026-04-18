/**
 * Binary scalar builtins: atan2, min, max, mod, rem, power.
 */

import {
  FloatXArray,
  isRuntimeTensor,
  type RuntimeTensor,
  type RuntimeValue,
} from "../../runtime/types.js";
import { mElemPow } from "../../helpers/arithmetic.js";
import { minMaxImpl } from "../../helpers/reduction/min-max.js";
import {
  type BuiltinCase,
  defineBuiltin,
  makeTensor,
  binaryMathJitEmit,
  binaryMathJitEmitC,
} from "./types.js";
import {
  type JitType,
  shapeAfterReduction,
  unifySign,
} from "../jit/jitTypes.js";

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
    const n = aTensor.data.length;
    if (n !== bTensor.data.length) throw new Error(`${name}: size mismatch`);
    const out = new FloatXArray(n);
    for (let i = 0; i < n; i++) out[i] = fn(aTensor.data[i], bTensor.data[i]);
    return makeTensor(out, undefined, aTensor.shape.slice());
  }
  throw new Error(`${name}: unsupported argument types`);
}

// ── Binary real element-wise cases ──────────────────────────────────────

/** Build cases for a binary real element-wise function. */
function binaryRealElemwiseCases(
  fn: (a: number, b: number) => number,
  name: string
): BuiltinCase[] {
  return [
    // Two scalars (number/boolean)
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0],
          b = argTypes[1];
        if (
          (a.kind !== "number" && a.kind !== "boolean") ||
          (b.kind !== "number" && b.kind !== "boolean")
        )
          return null;
        return [{ kind: "number" }];
      },
      apply: args => applyBinaryElemwise(args, fn, name),
    },
    // At least one tensor (both must be real)
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0],
          b = argTypes[1];
        const aOk =
          a.kind === "number" ||
          a.kind === "boolean" ||
          (a.kind === "tensor" && a.isComplex === false);
        const bOk =
          b.kind === "number" ||
          b.kind === "boolean" ||
          (b.kind === "tensor" && b.isComplex === false);
        if (!aOk || !bOk) return null;
        if (a.kind !== "tensor" && b.kind !== "tensor") return null;
        const t =
          a.kind === "tensor" ? a : (b as Extract<JitType, { kind: "tensor" }>);
        return [
          { kind: "tensor", isComplex: false, shape: t.shape, ndim: t.ndim },
        ];
      },
      apply: args => applyBinaryElemwise(args, fn, name),
    },
  ];
}

// ── atan2 ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "atan2",
  cases: binaryRealElemwiseCases(Math.atan2, "atan2"),
  jitEmit: binaryMathJitEmit("Math.atan2"),
  jitEmitC: binaryMathJitEmitC("atan2"),
});

// ── min / max ───────────────────────────────────────────────────────────

function minMaxCases(mode: "min" | "max"): BuiltinCase[] {
  const initVal = mode === "min" ? Infinity : -Infinity;
  const cmpFn =
    mode === "min"
      ? (a: number, b: number) => a < b
      : (a: number, b: number) => a > b;
  const mathFn = mode === "min" ? Math.min : Math.max;

  return [
    // 2-arg binary element-wise
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0],
          b = argTypes[1];
        const allowed = (k: string) =>
          k === "number" ||
          k === "boolean" ||
          k === "tensor" ||
          k === "sparse_matrix" ||
          k === "complex_or_number";
        if (!allowed(a.kind) || !allowed(b.kind)) return null;
        // Both-logical operand case: stay logical. MATLAB: class(min(true,
        // false)) == 'logical'. Mixed logical+double still promotes.
        const aIsLogical =
          a.kind === "boolean" || (a.kind === "tensor" && a.isLogical === true);
        const bIsLogical =
          b.kind === "boolean" || (b.kind === "tensor" && b.isLogical === true);
        if (a.kind === "tensor" || b.kind === "tensor") {
          const t =
            a.kind === "tensor"
              ? a
              : (b as Extract<JitType, { kind: "tensor" }>);
          const otherIsComplex =
            a.kind === "complex_or_number" || b.kind === "complex_or_number";
          return [
            {
              kind: "tensor",
              isComplex: t.isComplex || otherIsComplex,
              shape: t.shape,
              ndim: t.ndim,
              ...(aIsLogical && bIsLogical ? { isLogical: true } : {}),
            },
          ];
        }
        if (a.kind === "complex_or_number" || b.kind === "complex_or_number")
          return [{ kind: "complex_or_number" }];
        if (aIsLogical && bIsLogical) return [{ kind: "boolean" }];
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
      },
      apply: (args, nargout) =>
        minMaxImpl(mode, args, nargout, initVal, cmpFn, mathFn),
    },
    // 1-arg reduction
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1) return null;
        let a = argTypes[0];
        let valueType: JitType;
        if (
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "complex_or_number"
        ) {
          valueType = a;
        } else {
          if (a.kind === "sparse_matrix") {
            const shape =
              a.m !== undefined && a.n !== undefined ? [a.m, a.n] : undefined;
            a = { kind: "tensor", isComplex: a.isComplex, shape };
          }
          if (a.kind !== "tensor") return null;
          // Logical tensors reduce to a logical (scalar or smaller tensor).
          const isLogicalIn = a.isLogical === true;
          if (!a.shape) {
            valueType =
              a.isComplex === true
                ? { kind: "complex_or_number" }
                : isLogicalIn
                  ? { kind: "boolean" }
                  : { kind: "number" };
          } else {
            const result = shapeAfterReduction(a.shape);
            if (result.scalar) {
              valueType =
                a.isComplex === true
                  ? { kind: "complex_or_number" }
                  : isLogicalIn
                    ? { kind: "boolean" }
                    : { kind: "number" };
            } else {
              valueType = {
                kind: "tensor",
                isComplex: a.isComplex,
                shape: result.shape,
                ...(isLogicalIn ? { isLogical: true } : {}),
              };
            }
          }
        }
        // nargout >= 2: return [value, index] where index matches value shape
        if (nargout !== undefined && nargout >= 2) {
          const idxType: JitType =
            valueType.kind === "tensor"
              ? {
                  kind: "tensor",
                  isComplex: false,
                  shape: (valueType as { shape?: number[] }).shape,
                }
              : { kind: "number", sign: "positive" as const };
          return [valueType, idxType];
        }
        return [valueType];
      },
      apply: (args, nargout) =>
        minMaxImpl(mode, args, nargout, initVal, cmpFn, mathFn),
    },
    // 3-arg: min(X, [], dim)
    {
      match: argTypes => {
        if (argTypes.length !== 3) return null;
        let a = argTypes[0];
        if (a.kind === "sparse_matrix") {
          const shape =
            a.m !== undefined && a.n !== undefined ? [a.m, a.n] : undefined;
          a = { kind: "tensor", isComplex: a.isComplex, shape };
        }
        if (a.kind === "tensor") {
          const dimType = argTypes[2];
          const dim =
            dimType.kind === "number" && dimType.exact !== undefined
              ? dimType.exact
              : undefined;
          if (
            a.shape &&
            dim !== undefined &&
            dim >= 1 &&
            dim <= a.shape.length
          ) {
            const result = shapeAfterReduction(a.shape, dim);
            if (result.scalar)
              return [
                a.isComplex === true
                  ? { kind: "complex_or_number" }
                  : { kind: "number" },
              ];
            return [
              { kind: "tensor", isComplex: a.isComplex, shape: result.shape },
            ];
          }
          return [{ kind: "tensor", isComplex: a.isComplex, ndim: a.ndim }];
        }
        if (a.kind === "number" || a.kind === "boolean")
          return [{ kind: "number" }];
        return null;
      },
      apply: (args, nargout) =>
        minMaxImpl(mode, args, nargout, initVal, cmpFn, mathFn),
    },
  ];
}

// Only the 2-arg scalar form routes through jitEmitC. Reductions and
// tensor-binary forms are handled by dedicated tensor-op dispatch in
// jitCodegenC.ts / cFusedCodegen.ts, which skip ib.jitEmitC.
defineBuiltin({
  name: "min",
  cases: minMaxCases("min"),
  jitEmitC: binaryMathJitEmitC("fmin"),
});
defineBuiltin({
  name: "max",
  cases: minMaxCases("max"),
  jitEmitC: binaryMathJitEmitC("fmax"),
});

// ── mod ──────────────────────────────────────────────────────────────────

function modFn(a: number, b: number): number {
  if (b === 0) return a;
  let r = a % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return r;
}

defineBuiltin({
  name: "mod",
  cases: binaryRealElemwiseCases(modFn, "mod"),
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
  // numbl_mod lives in jit_runtime.a; matches JS's $h.mod semantics
  // (MATLAB-style floored modulo).
  jitEmitC: binaryMathJitEmitC("numbl_mod"),
});

// ── rem ──────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "rem",
  cases: binaryRealElemwiseCases((a, b) => a % b, "rem"),
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
  // C fmod matches JS's `%` truncated-toward-zero semantics for doubles.
  jitEmitC: binaryMathJitEmitC("fmod"),
});

// ── power ────────────────────────────────────────────────────────────────
// power(a,b) is the functional form of a.^b and must produce complex
// results for negative bases with fractional exponents, so delegate to
// mElemPow which handles that correctly.

defineBuiltin({
  name: "power",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0],
          b = argTypes[1];
        // Accept scalars and real tensors (mElemPow handles all combos)
        const aOk =
          a.kind === "number" ||
          a.kind === "boolean" ||
          (a.kind === "tensor" && a.isComplex === false);
        const bOk =
          b.kind === "number" ||
          b.kind === "boolean" ||
          (b.kind === "tensor" && b.isComplex === false);
        if (!aOk || !bOk) return null;
        // Output may be complex (negative base ^ fractional exp), so
        // return a generic type; mElemPow decides at runtime.
        if (a.kind === "tensor" || b.kind === "tensor") {
          const t =
            a.kind === "tensor"
              ? a
              : (b as Extract<JitType, { kind: "tensor" }>);
          return [
            {
              kind: "tensor",
              isComplex: undefined as unknown as boolean,
              shape: t.shape,
              ndim: t.ndim,
            },
          ];
        }
        return [{ kind: "number" }];
      },
      apply: args => mElemPow(args[0] as RuntimeValue, args[1] as RuntimeValue),
    },
  ],
  jitEmit: binaryMathJitEmit("Math.pow"),
  jitEmitC: binaryMathJitEmitC("pow"),
});

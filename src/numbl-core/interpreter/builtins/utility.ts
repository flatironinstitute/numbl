/**
 * Utility builtins for JIT: isequal, assert, abs.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeSparseMatrix,
  FloatXArray,
} from "../../runtime/types.js";
import type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeSparseMatrix,
} from "../../runtime/types.js";
import { defineBuiltin } from "./types.js";

// ── isequal ──────────────────────────────────────────────────────────────

function sparseToDense(S: RuntimeSparseMatrix): RuntimeTensor {
  const data = new FloatXArray(S.m * S.n);
  const imag = S.pi ? new FloatXArray(S.m * S.n) : undefined;
  for (let col = 0; col < S.n; col++) {
    for (let k = S.jc[col]; k < S.jc[col + 1]; k++) {
      const idx = col * S.m + S.ir[k];
      data[idx] = S.pr[k];
      if (imag && S.pi) imag[idx] = S.pi[k];
    }
  }
  return { kind: "tensor", data, imag, shape: [S.m, S.n], _rc: 1 };
}

function valuesEqualSimple(a: RuntimeValue, b: RuntimeValue): boolean {
  {
    const aText = textValue(a);
    const bText = textValue(b);
    if (aText !== null && bText !== null) return aText === bText;
  }
  if (isRuntimeSparseMatrix(a)) return valuesEqualSimple(sparseToDense(a), b);
  if (isRuntimeSparseMatrix(b)) return valuesEqualSimple(a, sparseToDense(b));
  if (typeof a === "number" && typeof b === "number") return a === b;
  if (
    (isRuntimeNumber(a) || isRuntimeLogical(a)) &&
    (isRuntimeNumber(b) || isRuntimeLogical(b))
  ) {
    const av = isRuntimeLogical(a) ? (a ? 1 : 0) : (a as number);
    const bv = isRuntimeLogical(b) ? (b ? 1 : 0) : (b as number);
    return av === bv;
  }
  if (isRuntimeComplexNumber(a) && isRuntimeComplexNumber(b))
    return a.re === b.re && a.im === b.im;
  if (isRuntimeComplexNumber(a) && typeof b === "number")
    return a.re === b && a.im === 0;
  if (typeof a === "number" && isRuntimeComplexNumber(b))
    return b.re === a && b.im === 0;
  if (isRuntimeTensor(a) && isRuntimeTensor(b)) return tensorsEqual(a, b);
  if (isRuntimeTensor(a) && typeof b === "number")
    return a.data.length === 1 && !a.imag && a.data[0] === b;
  if (typeof a === "number" && isRuntimeTensor(b))
    return b.data.length === 1 && !b.imag && b.data[0] === a;
  if (isRuntimeTensor(a) && isRuntimeComplexNumber(b))
    return (
      a.data.length === 1 &&
      a.data[0] === b.re &&
      (a.imag ? a.imag[0] === b.im : b.im === 0)
    );
  if (isRuntimeComplexNumber(a) && isRuntimeTensor(b))
    return (
      b.data.length === 1 &&
      b.data[0] === a.re &&
      (b.imag ? b.imag[0] === a.im : a.im === 0)
    );
  if (isRuntimeCell(a) && isRuntimeCell(b)) {
    if (a.shape.length !== b.shape.length) return false;
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) return false;
    }
    if (a.data.length !== b.data.length) return false;
    for (let i = 0; i < a.data.length; i++) {
      if (!valuesEqualSimple(a.data[i], b.data[i])) return false;
    }
    return true;
  }
  if (isRuntimeStruct(a) && isRuntimeStruct(b)) {
    if (a.fields.size !== b.fields.size) return false;
    for (const [key, val] of a.fields) {
      if (!b.fields.has(key)) return false;
      if (!valuesEqualSimple(val, b.fields.get(key)!)) return false;
    }
    return true;
  }
  return false;
}

function textValue(v: RuntimeValue): string | null {
  if (isRuntimeChar(v)) return v.value;
  if (isRuntimeString(v)) return v;
  return null;
}

function tensorsEqual(a: RuntimeTensor, b: RuntimeTensor): boolean {
  if (a.data.length !== b.data.length) return false;
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) return false;
  }
  if (a.imag || b.imag) {
    if (!a.imag || !b.imag) return false;
    for (let i = 0; i < a.imag.length; i++) {
      if (a.imag[i] !== b.imag[i]) return false;
    }
  }
  return true;
}

defineBuiltin({
  name: "isequal",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2) return null;
        return [{ kind: "boolean" }];
      },
      apply: args => {
        for (let i = 1; i < args.length; i++) {
          if (!valuesEqualSimple(args[0], args[i])) return false;
        }
        return true;
      },
    },
  ],
});

// ── assert ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "assert",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        let pass = false;
        if (typeof v === "boolean") pass = v;
        else if (typeof v === "number") pass = v !== 0;
        else if (isRuntimeLogical(v)) pass = v;
        else if (isRuntimeTensor(v)) {
          pass = v.data.length > 0;
          for (let i = 0; i < v.data.length; i++) {
            if (v.data[i] === 0) {
              pass = false;
              break;
            }
          }
        }
        if (!pass) {
          const msg =
            args.length > 1
              ? (textValue(args[1]) ?? String(args[1]))
              : "Assertion failed";
          throw new Error(msg);
        }
        return 0;
      },
    },
  ],
});

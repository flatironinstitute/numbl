/**
 * Utility builtins for JIT: isequal, assert, error, abs.
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
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  isRuntimeStringArray,
  isRuntimeClassInstance,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import { RuntimeError } from "../../runtime/error.js";
import { isEnumInstance, enumScalarEquals } from "../../runtime/runtimeEnum.js";
import { sprintfFormat } from "../../helpers/string.js";
import {
  isMException,
  makeMException,
  raiseException,
  exceptionText,
} from "../../runtime/mexception.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";

// ── isequal ──────────────────────────────────────────────────────────────

function valuesEqualSimple(a: RuntimeValue, b: RuntimeValue): boolean {
  // Enumeration members compare like `==` (value vs member/number, name vs
  // char) — so `isequal(patchtype.tri, 1)` and `isequal(x, 'tri')` hold.
  if (isEnumInstance(a)) return enumScalarEquals(a, b);
  if (isEnumInstance(b)) return enumScalarEquals(b, a);
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
  if (isRuntimeStructArray(a) || isRuntimeStructArray(b)) {
    const as = structElements(a);
    const bs = structElements(b);
    if (as === null || bs === null) return false;
    if (as.shape.length !== bs.shape.length) return false;
    for (let i = 0; i < as.shape.length; i++)
      if (as.shape[i] !== bs.shape[i]) return false;
    if (as.names.length !== bs.names.length) return false;
    if (as.names.some(f => !bs.names.includes(f))) return false;
    for (let i = 0; i < as.elems.length; i++)
      if (!valuesEqualSimple(as.elems[i], bs.elems[i])) return false;
    return true;
  }
  // Objects of the same class compare by their property values (a handle
  // object only equals itself).
  if (isRuntimeClassInstance(a) && isRuntimeClassInstance(b)) {
    if (a === b) return true;
    if (a.className !== b.className || a.isHandleClass) return false;
    if (a.fields.size !== b.fields.size) return false;
    for (const [key, val] of a.fields) {
      const other = b.fields.get(key);
      if (other === undefined || !valuesEqualSimple(val, other)) return false;
    }
    return true;
  }
  if (isRuntimeStringArray(a) && isRuntimeStringArray(b)) {
    if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1]) return false;
    for (let i = 0; i < a.data.length; i++) {
      if (a.data[i] !== b.data[i]) return false;
    }
    return true;
  }
  return false;
}

/** A struct or struct array as field names, elements and shape. */
function structElements(v: RuntimeValue): {
  names: string[];
  elems: RuntimeValue[];
  shape: number[];
} | null {
  if (isRuntimeStruct(v))
    return { names: [...v.fields.keys()], elems: [v], shape: [1, 1] };
  if (isRuntimeStructArray(v))
    return { names: v.fieldNames, elems: v.elements, shape: v.shape };
  return null;
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

builtin({
  name: "isequal",
  nargin: [2, Infinity],
  apply: args => {
    for (let i = 1; i < args.length; i++) {
      if (!valuesEqualSimple(args[0], args[i])) return false;
    }
    return true;
  },
});

// ── assert ───────────────────────────────────────────────────────────────

/** A MATLAB message identifier: component:mnemonic, word chars and colons
 *  (e.g. 'MATLAB:badInput'). */
export const MATLAB_ID_RE = /^[A-Za-z]\w*(:[A-Za-z]\w*)+$/;

builtin({
  name: "assert",
  nargin: [1, Infinity],
  nargout: 0,
  apply: args => {
    const v = args[0];
    let pass = false;
    if (typeof v === "boolean") pass = v;
    else if (typeof v === "number") pass = v !== 0 && !isNaN(v);
    else if (isRuntimeLogical(v)) pass = v;
    else if (isRuntimeTensor(v)) {
      pass = v.data.length > 0;
      for (let i = 0; i < v.data.length; i++) {
        if (v.data[i] === 0 || isNaN(v.data[i])) {
          pass = false;
          break;
        }
      }
    }
    if (pass) return;
    // MATLAB's forms: assert(c) -> "Assertion failed."; assert(c, msg) takes
    // msg literally; assert(c, [id,] fmt, A1, ...) formats.
    if (args.length === 1) {
      throw new RuntimeError("Assertion failed.", null, {
        identifier: "MATLAB:assertion:failed",
      });
    }
    const first = textValue(args[1]) ?? String(args[1]);
    if (args.length === 2) throw new RuntimeError(first);
    if (MATLAB_ID_RE.test(first)) {
      const fmt = textValue(args[2]) ?? String(args[2]);
      throw new RuntimeError(sprintfFormat(fmt, args.slice(3)), null, {
        identifier: first,
      });
    }
    throw new RuntimeError(sprintfFormat(first, args.slice(2)));
  },
});

// ── error ────────────────────────────────────────────────────────────────

builtin({
  name: "error",
  nargin: [1, Infinity],
  nargout: 0,
  apply: (args, _n, ctx) => {
    let s = args[0];
    if (isRuntimeStructArray(s)) {
      // error(errStruct) with an empty struct is a no-op.
      if (s.elements.length === 0) return;
      if (s.elements.length === 1) s = s.elements[0];
    }
    if (isRuntimeStruct(s) && args.length === 1) {
      // error(errStruct): its message, identifier and (if given) stack.
      const { identifier, message } = exceptionText(s);
      const stack = s.fields.get("stack");
      throw raiseException(
        makeMException(identifier, message, stack),
        stack !== undefined
      );
    }
    if (isMException(s, c => ctx.rt.classAncestors(c)) || isRuntimeStruct(s))
      throw builtinError(
        "MATLAB:error:invalidMessageType",
        "The message must be specified as either a character vector, string scalar, or a message structure."
      );
    const first = textValue(args[0]) ?? String(args[0]);
    // error('') (an empty message alone) is a no-op.
    if (args.length === 1 && first === "") return;
    // error(id, fmt, ...): the first arg is a MATLAB identifier. As in
    // MATLAB, the id form always formats its message (escapes too), while
    // a lone error(msg) is taken literally.
    if (args.length >= 2 && MATLAB_ID_RE.test(first)) {
      const msg = sprintfFormat(
        textValue(args[1]) ?? String(args[1]),
        args.slice(2)
      );
      const err = new RuntimeError(msg);
      err.identifier = first;
      throw err;
    }
    const msg = args.length === 1 ? first : sprintfFormat(first, args.slice(1));
    throw new RuntimeError(msg);
  },
});

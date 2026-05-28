/**
 * Type adapter for the mtoc2 C-JIT path.
 *
 *   - `mtoc2TypeToCDecl(ty)` returns the C type string that goes in the
 *     koffi prototype for a value of that type, or `null` when the
 *     type isn't supported across the FFI today.
 *   - `registerMtoc2TensorStruct(koffi)` registers `mtoc2_tensor_t`
 *     with koffi (idempotent across calls — koffi's `struct` returns
 *     the existing definition on re-registration of the same name).
 *   - `buildCDeclaration(signature)` assembles the full koffi prototype
 *     string from a `SpecCSignature`.
 *
 * Supported today (Phase 1):
 *   - Scalar real numeric → `double`
 *   - Scalar complex → ⛔ deferred (koffi's `_Complex` support varies;
 *     pass-by-(re, im) pair would simplify but mtoc2's emit signature
 *     is `double _Complex` and we don't rewrite the wrapper yet).
 *   - Scalar logical → `bool`
 *   - Tensor (real or complex) → `mtoc2_tensor_t` (koffi struct)
 *   - Void return → `void`
 *
 * Everything else (Struct / Class / Handle / String / Char / Cell)
 * returns null so the C-JIT executor declines and JS-JIT picks up.
 */

import type { SpecCSignature, Type, NumericType } from "../../mtoc2/index.js";
import { isMultiElement } from "../../mtoc2/index.js";

/** koffi's runtime-shaped api object. Loose typing because koffi
 *  doesn't ship TS types for its declaration-string API. */
interface Koffi {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  struct(name: string, fields: Record<string, unknown>): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  array(elemType: string, count: number, encoding?: string): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  resolve(name: string): any | undefined;
}

let tensorStructRegistered = false;

/** Register the `mtoc2_tensor_t` struct with koffi. Idempotent — safe
 *  to call from each compile() invocation; the second call short-
 *  circuits.
 *
 *  Layout MUST match `mtoc2/src/builtins/runtime/tensor/tensor.h`:
 *  ```
 *  typedef struct {
 *    double *real;
 *    double *imag;
 *    int  ndim;
 *    long dims[MTOC2_MAX_NDIM];   // MTOC2_MAX_NDIM == 8
 *  } mtoc2_tensor_t;
 *  ```
 *  Field order and padding match the C struct's natural alignment;
 *  koffi handles the platform-specific alignment automatically when
 *  asked for the same field types.
 */
export const MTOC2_MAX_NDIM = 8;
export function registerMtoc2TensorStruct(koffi: Koffi): void {
  if (tensorStructRegistered) return;
  // koffi.array("long", 8) gives a fixed-size inline array (NOT a
  // pointer-to-array); the "Array" encoding makes koffi return a JS
  // Array when reading the struct (faster than a TypedArray view for
  // a small N).
  koffi.struct("mtoc2_tensor_t", {
    real: "double *",
    imag: "double *",
    ndim: "int",
    dims: koffi.array("long", MTOC2_MAX_NDIM, "Array"),
  });
  tensorStructRegistered = true;
}

/** Map an mtoc2 `Type` to the C-type token used in the koffi
 *  declaration string. Returns `null` for unsupported shapes. */
export function mtoc2TypeToCDecl(ty: Type): string | null {
  if (ty.kind === "Void") return "void";
  if (ty.kind === "Numeric") return numericToCDecl(ty);
  // Struct / Class / Handle / String / Char / Cell / Unknown — defer.
  return null;
}

function numericToCDecl(ty: NumericType): string | null {
  if (isMultiElement(ty)) {
    // Tensor: real or complex, both ride the same struct.
    return "mtoc2_tensor_t";
  }
  // Scalar shape (1×1).
  if (ty.isComplex) return null; // deferred (see header).
  // Logical scalars share the `double` ABI with real numerics —
  // mtoc2's `cTypeFor` returns "double" uniformly for any
  // single-element Numeric. The koffi declaration must match the
  // actual C signature byte-for-byte; if we declared `bool` here,
  // koffi would pack one byte and the stack offsets would skew,
  // corrupting subsequent args / outputs.
  if (ty.elem === "logical" || ty.elem === "double") return "double";
  return null;
}

/** Assemble the koffi prototype string for a `SpecCSignature`.
 *
 *  Convention (mirrors mtoc2's `fnRetType` + `fnParamList`):
 *   - 0 outputs ⇒ `void <cName>(<params>)`
 *   - 1 output  ⇒ `<retType> <cName>(<params>)`
 *   - ≥2 outputs ⇒ `void <cName>(<params>, <outTy> *_mtoc2_o<i>...)`
 *
 *  Returns `null` if any param or output type fails the C-FFI
 *  feasibility check — the executor must decline. */
export function buildCDeclaration(signature: SpecCSignature): string | null {
  const paramParts: string[] = [];
  for (let i = 0; i < signature.params.length; i++) {
    const p = signature.params[i];
    const t = mtoc2TypeToCDecl(p.ty);
    if (t === null) return null;
    paramParts.push(`${t} ${p.cName}`);
  }
  const nOut = signature.outputs.length;
  let retType: string;
  if (nOut === 0) {
    retType = "void";
  } else if (nOut === 1) {
    const t = mtoc2TypeToCDecl(signature.outputs[0].ty);
    if (t === null) return null;
    retType = t;
  } else {
    retType = "void";
    for (let i = 0; i < nOut; i++) {
      const t = mtoc2TypeToCDecl(signature.outputs[i].ty);
      if (t === null) return null;
      paramParts.push(`${t} *_mtoc2_o${i}`);
    }
  }
  const params = paramParts.length === 0 ? "void" : paramParts.join(", ");
  return `${retType} ${signature.cName}(${params})`;
}

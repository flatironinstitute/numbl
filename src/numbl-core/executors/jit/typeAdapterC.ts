/**
 * Type adapter for the mtoc2 C-JIT path.
 *
 *   - `compilerTypeToCDecl(ty)` returns the C type string that goes in the
 *     koffi prototype for a value of that type, or `null` when the
 *     type isn't supported across the FFI today.
 *   - `registerTensorStruct(koffi)` registers `mtoc2_tensor_t`
 *     with koffi (idempotent across calls — koffi's `struct` returns
 *     the existing definition on re-registration of the same name).
 *   - `buildCDeclaration(signature)` assembles the full koffi prototype
 *     string from a `SpecCSignature`.
 *
 * Supported today:
 *   - Scalar real numeric → `double`
 *   - Scalar complex → `mtoc2_cscalar_t` (a `{double re; double im;}`
 *     koffi struct). The emitted C signature uses `double _Complex` by
 *     value, but on the SysV-x86-64 / AAPCS64 ABIs a 2-double struct
 *     and `double _Complex` share the same by-value calling convention
 *     (two SSE / SIMD registers, returned in xmm0:xmm1), so koffi
 *     marshals the struct against the real `_Complex` function — koffi's
 *     own prototype parser doesn't accept the `double _Complex` token.
 *   - Scalar logical → `double` (shares the `double` ABI)
 *   - Tensor (real or complex) → `mtoc2_tensor_t` (koffi struct)
 *   - Void return → `void`
 *
 * Everything else (Struct / Class / Handle / String / Char / Cell)
 * returns null so the C-JIT executor declines and JS-JIT picks up.
 */

import type { SpecCSignature, Type, NumericType } from "../../jit/index.js";
import { isMultiElement } from "../../jit/index.js";

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
let cscalarStructRegistered = false;

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
 *    long dims[JIT_MAX_NDIM];   // JIT_MAX_NDIM == 8
 *  } mtoc2_tensor_t;
 *  ```
 *  Field order and padding match the C struct's natural alignment;
 *  koffi handles the platform-specific alignment automatically when
 *  asked for the same field types.
 */
export const JIT_MAX_NDIM = 8;
export function registerTensorStruct(koffi: Koffi): void {
  // Register the complex-scalar struct too (self-gated) — kept before
  // the tensor short-circuit so it's registered even on calls where the
  // tensor struct already is.
  registerCScalarStruct(koffi);
  if (tensorStructRegistered) return;
  // koffi.array("long", 8) gives a fixed-size inline array (NOT a
  // pointer-to-array); the "Array" encoding makes koffi return a JS
  // Array when reading the struct (faster than a TypedArray view for
  // a small N).
  koffi.struct("mtoc2_tensor_t", {
    real: "double *",
    imag: "double *",
    ndim: "int",
    dims: koffi.array("long", JIT_MAX_NDIM, "Array"),
  });
  tensorStructRegistered = true;
}

/** Register the `mtoc2_cscalar_t` struct — `{double re; double im;}` —
 *  used as the koffi stand-in for `double _Complex` scalar params and
 *  returns (ABI-compatible by-value; see header). Idempotent. */
export function registerCScalarStruct(koffi: Koffi): void {
  if (cscalarStructRegistered) return;
  koffi.struct("mtoc2_cscalar_t", { re: "double", im: "double" });
  cscalarStructRegistered = true;
}

/** Map an mtoc2 `Type` to the C-type token used in the koffi
 *  declaration string. Returns `null` for unsupported shapes. */
export function compilerTypeToCDecl(ty: Type): string | null {
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
  // Complex scalar: the emitted C signature is `double _Complex`, but
  // koffi marshals the ABI-identical `{re, im}` struct (see header).
  if (ty.isComplex) return "mtoc2_cscalar_t";
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
    const t = compilerTypeToCDecl(p.ty);
    if (t === null) return null;
    paramParts.push(`${t} ${p.cName}`);
  }
  const nOut = signature.outputs.length;
  let retType: string;
  if (nOut === 0) {
    retType = "void";
  } else if (nOut === 1) {
    const t = compilerTypeToCDecl(signature.outputs[0].ty);
    if (t === null) return null;
    retType = t;
  } else {
    retType = "void";
    for (let i = 0; i < nOut; i++) {
      const t = compilerTypeToCDecl(signature.outputs[i].ty);
      if (t === null) return null;
      paramParts.push(`${t} *_mtoc2_o${i}`);
    }
  }
  const params = paramParts.length === 0 ? "void" : paramParts.join(", ");
  return `${retType} ${signature.cName}(${params})`;
}

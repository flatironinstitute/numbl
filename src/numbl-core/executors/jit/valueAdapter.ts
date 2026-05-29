/**
 * Adapter: numbl `RuntimeValue` ↔ mtoc2 emit-JS value shape.
 *
 * Numbl wraps tensors / chars / complex / structs in Refcounted
 * classes (`RuntimeTensor`, `RuntimeChar`, `RuntimeComplexNumber`)
 * keyed by a `kind:` discriminator. mtoc2's emitted JS uses plain
 * objects keyed by `mtoc2Tag:` (no class wrapper, no refcount).
 *
 * **Pass-by-value at the boundary.** MATLAB function calls are
 * pass-by-value; a JIT'd function that mutates its argument must
 * not leak the mutation to the caller. mtoc2's whole-program
 * codegen handles this by wrapping every call-site argument in
 * `mtoc2_deep_clone(...)` — but the JIT executor is a *new* call
 * site that mtoc2 doesn't know about, so we have to clone the
 * tensor data buffer ourselves on the way in. On the way out, we
 * take ownership of the returned buffer directly (mtoc2's spec
 * always returns a freshly-owned tensor).
 *
 * The executor only invokes these adapters on values whose JitType
 * was accepted by `jitTypeToCompilerType`. Unsupported value kinds
 * arriving here are a programmer error — the type adapter should
 * have caused `propose()` to decline earlier.
 */

import {
  RuntimeTensor,
  RuntimeChar,
  RuntimeComplexNumber,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  type RuntimeValue,
} from "../../runtime/types.js";

/** numbl RuntimeValue → mtoc2 emit-JS value shape. Owned-typed
 *  values (tensors) get their data buffer cloned so mtoc2's spec
 *  body can mutate freely without leaking the change back through
 *  numbl's caller-side env. */
export function numblToJit(v: RuntimeValue): unknown {
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return v ? 1 : 0;
  if (isRuntimeString(v)) return v;
  if (isRuntimeTensor(v)) {
    // Clone the data buffer. mtoc2's emitted body mutates parameters
    // in place (it expects the caller to have cloned at call site).
    const out: {
      mtoc2Tag: "tensor";
      shape: number[];
      data: Float64Array;
      imag?: Float64Array;
    } = {
      mtoc2Tag: "tensor",
      shape: [...v.shape],
      data: new Float64Array(v.data),
    };
    if (v.imag !== undefined) out.imag = new Float64Array(v.imag);
    return out;
  }
  if (isRuntimeChar(v)) {
    return { mtoc2Tag: "char", value: v.value };
  }
  if (isRuntimeComplexNumber(v)) {
    return { re: v.re, im: v.im };
  }
  throw new Error(
    `numblToJit: unsupported RuntimeValue (executor should have declined)`
  );
}

/** mtoc2 emit-JS return value → numbl RuntimeValue. */
export function jitToNumbl(v: unknown): RuntimeValue {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return v;
  if (typeof v !== "object" || v === null) {
    throw new Error(`jitToNumbl: unexpected primitive of type ${typeof v}`);
  }
  const tagged = v as {
    mtoc2Tag?: string;
    re?: number;
    im?: number;
    shape?: number[];
    data?: Float64Array;
    imag?: Float64Array;
    value?: string;
  };
  if (tagged.mtoc2Tag === "tensor") {
    if (!tagged.shape || !tagged.data) {
      throw new Error(`jitToNumbl: tensor missing shape/data`);
    }
    return new RuntimeTensor(tagged.data, [...tagged.shape], tagged.imag);
  }
  if (tagged.mtoc2Tag === "char") {
    if (typeof tagged.value !== "string") {
      throw new Error(`jitToNumbl: char missing value`);
    }
    return new RuntimeChar(tagged.value);
  }
  if (typeof tagged.re === "number" && typeof tagged.im === "number") {
    return new RuntimeComplexNumber(tagged.re, tagged.im);
  }
  throw new Error(`jitToNumbl: unrecognized return shape`);
}

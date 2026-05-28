/**
 * Adapter: numbl `RuntimeValue` ↔ mtoc2 emit-JS value shape.
 *
 * Numbl wraps tensors / chars / complex / structs in Refcounted
 * classes (`RuntimeTensor`, `RuntimeChar`, `RuntimeComplexNumber`)
 * keyed by a `kind:` discriminator. mtoc2's emitted JS uses plain
 * objects keyed by `mtoc2Tag:` (no class wrapper, no refcount). Field
 * shapes are otherwise identical (shape / data / imag, value, re/im),
 * so the adapter copies field references — no buffer copy.
 *
 * The executor only invokes these adapters on values whose JitType
 * was accepted by `jitTypeToMtoc2Type`. Unsupported value kinds
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

/** numbl RuntimeValue → mtoc2 emit-JS value shape. */
export function numblToMtoc2(v: RuntimeValue): unknown {
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return v ? 1 : 0;
  if (isRuntimeString(v)) return v;
  if (isRuntimeTensor(v)) {
    return v.imag !== undefined
      ? { mtoc2Tag: "tensor", shape: v.shape, data: v.data, imag: v.imag }
      : { mtoc2Tag: "tensor", shape: v.shape, data: v.data };
  }
  if (isRuntimeChar(v)) {
    return { mtoc2Tag: "char", value: v.value };
  }
  if (isRuntimeComplexNumber(v)) {
    return { re: v.re, im: v.im };
  }
  throw new Error(
    `numblToMtoc2: unsupported RuntimeValue (executor should have declined)`
  );
}

/** mtoc2 emit-JS return value → numbl RuntimeValue. */
export function mtoc2ToNumbl(v: unknown): RuntimeValue {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return v;
  if (typeof v !== "object" || v === null) {
    throw new Error(`mtoc2ToNumbl: unexpected primitive of type ${typeof v}`);
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
      throw new Error(`mtoc2ToNumbl: tensor missing shape/data`);
    }
    return new RuntimeTensor(tagged.data, [...tagged.shape], tagged.imag);
  }
  if (tagged.mtoc2Tag === "char") {
    if (typeof tagged.value !== "string") {
      throw new Error(`mtoc2ToNumbl: char missing value`);
    }
    return new RuntimeChar(tagged.value);
  }
  if (typeof tagged.re === "number" && typeof tagged.im === "number") {
    return new RuntimeComplexNumber(tagged.re, tagged.im);
  }
  throw new Error(`mtoc2ToNumbl: unrecognized return shape`);
}

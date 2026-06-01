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
import type { JitType } from "../../jitTypes.js";

/** numbl RuntimeValue → mtoc2 emit-JS value shape. Owned-typed
 *  values (tensors) get their data buffer cloned so mtoc2's spec
 *  body can mutate freely without leaking the change back through
 *  numbl's caller-side env.
 *
 *  `targetType` is the JitType the spec was COMPILED for at this
 *  parameter. Type-widening can reuse a complex specialization for a
 *  later real/boolean scalar call (both keys collapse to
 *  `complex_or_number`); the complex-typed body reads `.re`/`.im`, so a
 *  bare JS number arriving there yields `undefined` → NaN. Box such a
 *  scalar into `{re, im:0}` to mirror the C adapter (valueAdapterC),
 *  which already boxes a real as a complex with `im=0`. */
export function numblToJit(v: RuntimeValue, targetType?: JitType): unknown {
  if (targetType?.kind === "complex_or_number") {
    if (isRuntimeNumber(v)) return { re: v as number, im: 0 };
    if (isRuntimeLogical(v)) return { re: v ? 1 : 0, im: 0 };
    if (isRuntimeComplexNumber(v)) return { re: v.re, im: v.im };
  }
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

/** True if `e` is the grow-bail sentinel thrown by the emitted JS
 *  store helpers (`mtoc2_idx_*_grow_js` in `scalar_index.js`) — an
 *  indexed store whose index exceeds the runtime extent, which would
 *  grow the array (unsupported in the JIT). The JS executors recognize
 *  it to surface a one-time warning; they bail on it (and on any other
 *  runtime error) so the interpreter re-runs the scope with full
 *  MATLAB grow semantics. */
export function isGrowBail(e: unknown): boolean {
  return (
    typeof e === "object" &&
    e !== null &&
    (e as { mtoc2GrowBail?: boolean }).mtoc2GrowBail === true
  );
}

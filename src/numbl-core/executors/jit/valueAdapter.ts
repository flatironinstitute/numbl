/**
 * Adapter: numbl `RuntimeValue` ↔ JIT emit-JS value shape.
 *
 * Numbl wraps tensors / chars / complex / structs in Refcounted
 * classes (`RuntimeTensor`, `RuntimeChar`, `RuntimeComplexNumber`)
 * keyed by a `kind:` discriminator. The JIT's emitted JS uses plain
 * objects keyed by `jitTag:` (no class wrapper, no refcount).
 *
 * **Pass-by-value at the boundary.** MATLAB function calls are
 * pass-by-value; a JIT'd function that mutates its argument must
 * not leak the mutation to the caller. The JIT's whole-program
 * codegen handles this by wrapping every call-site argument in
 * `jit_deep_clone(...)` — but the JIT executor is a *new* call
 * site that the JIT doesn't know about, so we have to clone the
 * tensor data buffer ourselves on the way in. On the way out, we
 * take ownership of the returned buffer directly (the JIT's spec
 * always returns a freshly-owned tensor).
 *
 * The executor only invokes these adapters on values whose type
 * (`typeOfValue`) the backend accepted. Unsupported value kinds
 * arriving here are a programmer error: `propose()` should have
 * declined earlier.
 */

import {
  RuntimeTensor,
  RuntimeChar,
  RuntimeStruct,
  RuntimeComplexNumber,
  isRuntimeStruct,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  type RuntimeValue,
} from "../../runtime/types.js";
import type { Type, NumericType } from "../../jit/index.js";
import { isMultiElement } from "../../jit/index.js";

/** numbl RuntimeValue → JIT emit-JS value shape. Owned-typed
 *  values (tensors) get their data buffer cloned so the JIT's spec
 *  body can mutate freely without leaking the change back through
 *  numbl's caller-side env.
 *
 *  `paramType` is the compiler `Type` the spec was COMPILED for at this
 *  parameter. It reconciles two value/type mismatches at the boundary,
 *  both mirroring the C adapter (valueAdapterC):
 *
 *   1. **1×1 tensor → scalar param.** The compiler collapses a [1,1]
 *      tensor type to a scalar `double`/`complex` (`!isMultiElement`),
 *      so scalar codegen reads a bare number / `{re, im}`. A 1×1
 *      `RuntimeTensor` value must therefore be unwrapped to its element;
 *      otherwise the tensor OBJECT leaks into scalar arithmetic and
 *      `a.data[i] + b` becomes `number + object = NaN`.
 *   2. **real → complex-scalar param.** Type-widening can reuse a complex
 *      specialization for a later real/boolean scalar call; the
 *      complex-typed body reads `.re`/`.im`, so a bare number must be
 *      boxed into `{re, im:0}`.
 *   3. **real → complex-tensor param.** Likewise for a real tensor: it
 *      gets a zero `imag` lane.
 *
 *  `borrow` passes a tensor's buffers uncloned, for a spec that only
 *  reads its parameters and returns none of them
 *  (`CompileRequest.borrowsInputs`). */
export function numblToJit(
  v: RuntimeValue,
  paramType?: Type,
  borrow = false
): unknown {
  if (paramType?.kind === "Numeric" && !isMultiElement(paramType)) {
    const nty = paramType as NumericType;
    // (1) Unwrap a 1×1 tensor flowing into a scalar param to its element.
    if (isRuntimeTensor(v) && v.data.length === 1) {
      if (nty.isComplex || v.imag !== undefined) {
        return { re: v.data[0], im: v.imag?.[0] ?? 0 };
      }
      return v.data[0];
    }
    // (2) Box a real/logical/complex scalar for a complex-scalar param.
    if (nty.isComplex) {
      if (isRuntimeNumber(v)) return { re: v as number, im: 0 };
      if (isRuntimeLogical(v)) return { re: v ? 1 : 0, im: 0 };
      if (isRuntimeComplexNumber(v)) return { re: v.re, im: v.im };
    }
  }
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return v ? 1 : 0;
  if (isRuntimeString(v)) return v;
  if (isRuntimeTensor(v)) {
    // Clone the data buffer. The JIT's emitted body mutates parameters
    // in place (it expects the caller to have cloned at call site).
    const out: {
      jitTag: "tensor";
      shape: number[];
      data: Float64Array;
      imag?: Float64Array;
    } = {
      jitTag: "tensor",
      shape: [...v.shape],
      data: borrow ? v.data : new Float64Array(v.data),
    };
    if (v.imag !== undefined) {
      out.imag = borrow ? v.imag : new Float64Array(v.imag);
    } else if (paramType?.kind === "Numeric" && paramType.isComplex) {
      // (3) Widening can reuse a complex specialization for a later real
      // tensor; the complex-typed body reads `.imag`, so give it zeros.
      out.imag = new Float64Array(v.data.length);
    }
    return out;
  }
  if (isRuntimeChar(v)) {
    return { jitTag: "char", value: v.value };
  }
  if (isRuntimeComplexNumber(v)) {
    return { re: v.re, im: v.im };
  }
  if (isRuntimeStruct(v)) {
    // The JIT represents a struct as a plain JS object keyed by field name
    // (see StructLit / MemberLoad in emitJs). Each field is converted with
    // the field type the spec was compiled for, so a 1x1 tensor field
    // reaching a scalar-typed field is unwrapped exactly as an argument
    // would be. Fields the spec does not know about are dropped: the spec
    // was compiled against this struct's field set.
    const sty = paramType?.kind === "Struct" ? paramType : undefined;
    const out: Record<string, unknown> = {};
    for (const [name, fv] of v.fields) {
      const fty = sty?.fields.find(f => f.name === name)?.ty;
      out[name] = numblToJit(fv, fty);
    }
    return out;
  }
  throw new Error(
    `numblToJit: unsupported RuntimeValue (executor should have declined)`
  );
}

/** JIT emit-JS return value → numbl RuntimeValue.
 *
 *  `retType` is the compiler type the spec declares for this output. It is
 *  only consulted for plain untagged objects, which is how the emitted code
 *  represents both a struct and a class instance — the value alone cannot
 *  tell them apart, so a struct is only rebuilt when the type says Struct
 *  and anything else untagged is rejected (the executor then bails and the
 *  interpreter, which has full class semantics, re-runs the call). */
export function jitToNumbl(v: unknown, retType?: Type): RuntimeValue {
  // Logical values ride as plain numbers (and untagged-logical tensors) in
  // emitted JS, e.g. a logical parameter returned unchanged; the declared
  // output type restores the class.
  const logical =
    retType?.kind === "Numeric" && (retType as NumericType).elem === "logical";
  if (typeof v === "number") return logical ? v !== 0 : v;
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return v;
  if (typeof v !== "object" || v === null) {
    throw new Error(`jitToNumbl: unexpected primitive of type ${typeof v}`);
  }
  const tagged = v as {
    jitTag?: string;
    re?: number;
    im?: number;
    shape?: number[];
    data?: Float64Array;
    imag?: Float64Array;
    value?: string;
  };
  if (tagged.jitTag === "tensor") {
    if (!tagged.shape || !tagged.data) {
      throw new Error(`jitToNumbl: tensor missing shape/data`);
    }
    return new RuntimeTensor(
      tagged.data,
      [...tagged.shape],
      tagged.imag,
      logical ? "logical" : "double"
    );
  }
  if (tagged.jitTag === "char") {
    if (typeof tagged.value !== "string") {
      throw new Error(`jitToNumbl: char missing value`);
    }
    return new RuntimeChar(tagged.value);
  }
  if (typeof tagged.re === "number" && typeof tagged.im === "number") {
    return new RuntimeComplexNumber(tagged.re, tagged.im);
  }
  if (retType?.kind === "Struct") {
    // A struct: the JIT emits it as a plain, untagged JS object of field
    // values, exactly as it emits a class instance — hence the type check.
    const fields = new Map<string, RuntimeValue>();
    for (const f of retType.fields) {
      const fv = (v as Record<string, unknown>)[f.name];
      if (fv !== undefined) fields.set(f.name, jitToNumbl(fv, f.ty));
    }
    return new RuntimeStruct(fields);
  }
  throw new Error(`jitToNumbl: unrecognized return shape`);
}

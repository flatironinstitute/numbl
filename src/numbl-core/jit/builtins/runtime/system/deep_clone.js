/* Deep-clone an owned-typed runtime value so a subsequent member /
 * indexed write through one alias doesn't leak to others. Mirrors
 * the c-aot per-typedef `_copy` protocol: every Var-RHS Assign /
 * MemberLoad-RHS Assign / function-call arg of an owned type is
 * wrapped through this helper at emit time, so JS-side aliasing
 * stops at the assignment boundary.
 *
 * Recognises the runtime shapes the rest of mtoc2 emits:
 *   - RuntimeTensor (`{mtoc2Tag:"tensor", shape, data, imag?, isLogical?}`)
 *   - RuntimeChar   (`{mtoc2Tag:"char", value}`)
 *   - struct / class / handle / handle-capture objects — plain JS
 *     objects, optionally with a non-enumerable `mtoc2Class` tag
 *     (class instances) that must survive the clone for method
 *     dispatch
 *
 * Primitives (numbers, booleans, strings, null/undefined) flow
 * through unchanged — they're already value-typed in JS.
 */

export function mtoc2_deep_clone(v) {
  if (v === null || typeof v !== "object") return v;
  if (v.mtoc2Tag === "tensor") {
    const out = {
      mtoc2Tag: "tensor",
      shape: v.shape.slice(),
      data: v.data.slice(),
    };
    if (v.imag !== undefined) out.imag = v.imag.slice();
    if (v.isLogical) out.isLogical = true;
    return out;
  }
  if (v.mtoc2Tag === "char") {
    return { mtoc2Tag: "char", value: v.value };
  }
  if (v.mtoc2Tag === "cell") {
    return {
      mtoc2Tag: "cell",
      shape: v.shape.slice(),
      data: v.data.map(mtoc2_deep_clone),
    };
  }
  if (v.mtoc2Handle === true) {
    // Function handle — captures are deep-cloned so a later write
    // through one alias doesn't bleed into another handle's frozen
    // capture snapshot.
    if (v.kind === "named") {
      return { mtoc2Handle: true, kind: "named", name: v.name };
    }
    const captures = {};
    for (const k of Object.keys(v.captures)) {
      captures[k] = mtoc2_deep_clone(v.captures[k]);
    }
    return {
      mtoc2Handle: true,
      kind: "anon",
      params: v.params,
      body: v.body,
      captures,
    };
  }
  // Plain struct / class instance: walk enumerable fields, then
  // restore the non-enumerable `mtoc2Class` tag if it was present.
  const out = {};
  for (const k of Object.keys(v)) {
    out[k] = mtoc2_deep_clone(v[k]);
  }
  const tag = Object.getOwnPropertyDescriptor(v, "mtoc2Class");
  if (tag !== undefined) {
    Object.defineProperty(out, "mtoc2Class", {
      value: tag.value,
      enumerable: false,
      writable: false,
    });
  }
  return out;
}

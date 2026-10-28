/* Deep-clone an owned-typed runtime value so a subsequent member /
 * indexed write through one alias doesn't leak to others. Mirrors
 * the C-JIT per-typedef `_copy` protocol: every Var-RHS Assign /
 * MemberLoad-RHS Assign / function-call arg of an owned type is
 * wrapped through this helper at emit time, so JS-side aliasing
 * stops at the assignment boundary.
 *
 * Recognises the runtime shapes the rest of the JIT emits:
 *   - RuntimeTensor (`{jitTag:"tensor", shape, data, imag?, isLogical?}`)
 *   - RuntimeChar   (`{jitTag:"char", value}`)
 *   - struct / class / handle / handle-capture objects — plain JS
 *     objects, optionally with a non-enumerable `jitClass` tag
 *     (class instances) that must survive the clone for method
 *     dispatch
 *
 * Primitives (numbers, booleans, strings, null/undefined) flow
 * through unchanged — they're already value-typed in JS.
 */

export function jit_deep_clone(v) {
  if (v === null || typeof v !== "object") return v;
  if (v.jitTag === "tensor") {
    const out = {
      jitTag: "tensor",
      shape: v.shape.slice(),
      data: v.data.slice(),
    };
    if (v.imag !== undefined) out.imag = v.imag.slice();
    if (v.isLogical) out.isLogical = true;
    return out;
  }
  if (v.jitTag === "char") {
    return { jitTag: "char", value: v.value };
  }
  if (v.jitTag === "cell") {
    return {
      jitTag: "cell",
      shape: v.shape.slice(),
      data: v.data.map(jit_deep_clone),
    };
  }
  if (v.jitHandle === true) {
    // Function handle — captures are deep-cloned so a later write
    // through one alias doesn't bleed into another handle's frozen
    // capture snapshot.
    if (v.kind === "named") {
      return { jitHandle: true, kind: "named", name: v.name };
    }
    const captures = {};
    for (const k of Object.keys(v.captures)) {
      captures[k] = jit_deep_clone(v.captures[k]);
    }
    return {
      jitHandle: true,
      kind: "anon",
      params: v.params,
      body: v.body,
      captures,
    };
  }
  // Plain struct / class instance: walk enumerable fields, then
  // restore the non-enumerable `jitClass` tag if it was present.
  const out = {};
  for (const k of Object.keys(v)) {
    out[k] = jit_deep_clone(v[k]);
  }
  const tag = Object.getOwnPropertyDescriptor(v, "jitClass");
  if (tag !== undefined) {
    Object.defineProperty(out, "jitClass", {
      value: tag.value,
      enumerable: false,
      writable: false,
    });
  }
  return out;
}

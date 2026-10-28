// JS sibling of `tensor.h`. The C side defines a struct
// (`jit_tensor_t`); JS uses a plain object with the same logical
// fields:
//
//   { jitTag: "tensor", shape: number[], data: Float64Array,
//     imag?: Float64Array }
//
// `jitTag` distinguishes tensors from plain JS objects. `shape` is
// column-major to match numbl / LAPACK. `imag` is undefined for
// statically-real tensors.
//
// JS has GC, so the C-side ownership protocol's `_free` / `_assign`
// helpers have no JS counterpart — lifetime is automatic. Value
// semantics aren't: an alias-producing assignment (`b = a`, function
// arg pass) still needs a deep copy or the next member / indexed
// write through one alias would leak. See `system/deep_clone.js`
// and the `emitOwnedRhsJs` wrapper in `src/codegen/emitJs.ts`.

export function jit_tensor_make(shape, data) {
  let total = 1;
  for (const s of shape) total *= s;
  if (data.length !== total) {
    throw new Error(
      `jit_tensor_make: shape [${shape.join(",")}] expects ${total} elements, got ${data.length}`
    );
  }
  return { jitTag: "tensor", shape: shape.slice(), data };
}

export function jit_tensor_numel(t) {
  let n = 1;
  for (const s of t.shape) n *= s;
  return n;
}

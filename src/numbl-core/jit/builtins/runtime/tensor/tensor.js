// JS sibling of `tensor.h`. The C side defines a struct
// (`mtoc2_tensor_t`); JS uses a plain object with the same logical
// fields:
//
//   { mtoc2Tag: "tensor", shape: number[], data: Float64Array,
//     imag?: Float64Array }
//
// `mtoc2Tag` lets the interpreter distinguish tensors from plain JS
// objects (matches the `RuntimeTensor` discriminator in
// `runtime/value.ts`). `shape` is column-major to match numbl /
// LAPACK. `imag` is undefined for statically-real tensors.
//
// JS has GC, so the C-side ownership protocol's `_free` / `_assign`
// helpers have no JS counterpart — lifetime is automatic. Value
// semantics aren't: an alias-producing assignment (`b = a`, function
// arg pass) still needs a deep copy or the next member / indexed
// write through one alias would leak. See `system/deep_clone.js`
// and the `emitOwnedRhsJs` wrapper in `src/codegen/emitJs.ts`.

export function mtoc2_tensor_make(shape, data) {
  let total = 1;
  for (const s of shape) total *= s;
  if (data.length !== total) {
    throw new Error(
      `mtoc2_tensor_make: shape [${shape.join(",")}] expects ${total} elements, got ${data.length}`
    );
  }
  return { mtoc2Tag: "tensor", shape: shape.slice(), data };
}

export function mtoc2_tensor_numel(t) {
  let n = 1;
  for (const s of t.shape) n *= s;
  return n;
}

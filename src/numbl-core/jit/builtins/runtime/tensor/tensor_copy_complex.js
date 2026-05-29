// JS sibling of `tensor_copy_complex.h`. Deep-copy a complex tensor
// (both lanes). The C path needs this because every owned-value
// produces a freshly-owned result; in JS the GC handles ownership,
// but we still want a structural deep copy so the source can be
// mutated independently.

export function mtoc2_tensor_copy_complex(t) {
  return {
    mtoc2Tag: "tensor",
    shape: t.shape.slice(),
    data: new Float64Array(t.data),
    imag:
      t.imag !== undefined
        ? new Float64Array(t.imag)
        : new Float64Array(t.data.length),
  };
}

// JS sibling of `tensor_alloc_nd_complex.h`. Allocate an N-D complex
// tensor; both `data` and `imag` lanes are zero-initialised.

export function mtoc2_tensor_alloc_nd_complex(ndim, dims) {
  const shape = [];
  for (let i = 0; i < ndim; i++) {
    shape.push(dims[i] < 0 ? 0 : dims[i]);
  }
  let total = 1;
  for (const s of shape) total *= s;
  return {
    mtoc2Tag: "tensor",
    shape,
    data: new Float64Array(total),
    imag: new Float64Array(total),
  };
}

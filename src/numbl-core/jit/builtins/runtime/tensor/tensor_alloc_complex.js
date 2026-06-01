// JS sibling of `tensor_alloc_complex.h`. Allocate a 2-D complex
// tensor; both `data` (real) and `imag` lanes are zero-initialised.

export function mtoc2_tensor_alloc_complex(rows, cols) {
  if (rows < 0) rows = 0;
  if (cols < 0) cols = 0;
  const n = rows * cols;
  return {
    mtoc2Tag: "tensor",
    shape: [rows, cols],
    data: new Float64Array(n),
    imag: new Float64Array(n),
  };
}

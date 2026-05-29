// JS sibling of `tensor_linspace.h`. Build a 1×n row tensor of n
// linearly-spaced values from `a` to `b`.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";

export function mtoc2_tensor_linspace(a, b, n) {
  if (n < 0) n = 0;
  const out = mtoc2_tensor_alloc(1, n);
  if (n === 0) return out;
  if (n === 1) {
    out.data[0] = b;
    return out;
  }
  out.data[0] = a;
  out.data[n - 1] = b;
  for (let i = 1; i < n - 1; i++) {
    out.data[i] = a + ((b - a) * i) / (n - 1);
  }
  if ((n & 1) === 1 && !Number.isFinite(a) && !Number.isFinite(b)) {
    const sa = Math.sign(a);
    const sb = Math.sign(b);
    if (sa !== sb) out.data[(n - 1) / 2] = 0;
  }
  return out;
}

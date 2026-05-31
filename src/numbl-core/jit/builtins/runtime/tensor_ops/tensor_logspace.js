// JS sibling of `tensor_logspace.h`. Build a 1×n row tensor of n
// logarithmically-spaced values from 10^a to 10^b. Byte-for-byte with
// numbl's interpreter `logspace`, including the MATLAB special case
// where an upper limit of exactly `pi` makes the last point `pi`.

import { mtoc2_tensor_alloc } from "../tensor/tensor_alloc.js";

export function mtoc2_tensor_logspace(a, b, n) {
  if (n <= 0) return mtoc2_tensor_alloc(1, 0);
  const isPi = b === Math.PI;
  const endVal = isPi ? Math.PI : Math.pow(10, b);
  const out = mtoc2_tensor_alloc(1, n);
  if (n === 1) {
    out.data[0] = endVal;
    return out;
  }
  if (isPi) {
    const logStart = Math.log10(Math.pow(10, a));
    const logEnd = Math.log10(Math.PI);
    for (let i = 0; i < n; i++) {
      const t = logStart + ((logEnd - logStart) * i) / (n - 1);
      out.data[i] = Math.pow(10, t);
    }
  } else {
    for (let i = 0; i < n; i++) {
      const t = a + ((b - a) * i) / (n - 1);
      out.data[i] = Math.pow(10, t);
    }
  }
  return out;
}

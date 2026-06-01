// JS sibling of `tensor_from_row.h`. Build a 1×N tensor from a JS
// array (or typed array) of values, copied into a fresh
// Float64Array. The codegen path passes a plain JS array literal:
//   mtoc2_tensor_from_row([1, 2, 3], 3)
//
// The C side signature is `mtoc2_tensor_from_row(double[], long
// cols)`, which the C compiler can read inline from a compound
// literal. The JS analogue is the same shape.

import { mtoc2_tensor_alloc } from "./tensor_alloc.js";

export function mtoc2_tensor_from_row(values, cols) {
  const t = mtoc2_tensor_alloc(1, cols);
  for (let i = 0; i < cols; i++) t.data[i] = values[i];
  return t;
}

import { defineShapeConstructor } from "./_construct.js";

export const zeros = defineShapeConstructor(
  "zeros",
  0,
  "jit_tensor_zeros_nd",
  "jit_tensor_zeros_square"
);

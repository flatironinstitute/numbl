import { defineShapeConstructor } from "./_construct.js";

export const ones = defineShapeConstructor(
  "ones",
  1,
  "jit_tensor_ones_nd",
  "jit_tensor_ones_square"
);

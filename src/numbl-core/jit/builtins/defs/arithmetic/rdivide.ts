import type { Builtin } from "../../registry.js";
import { mtoc2_cdiv_mixed } from "../../runtime/snippets.gen.js";
import { defineElemwiseRealBinary, signQuot } from "./_elemwise.js";

export const rdivide: Builtin = defineElemwiseRealBinary({
  name: "rdivide",
  cOp: "/",
  helperBase: "rdivide",
  commutative: false,
  fold: (a, b) => a / b,
  signRule: signQuot,
  // Complex scalar `/` routes through the runtime helper (Smith's
  // algorithm, numbl's signed-Inf edges, MATLAB's mixed real/complex
  // rule). The real-typed scalar path keeps the bare `/`.
  complexFold: mtoc2_cdiv_mixed,
  complexScalarExpr: (a, b) => `mtoc2_cdiv_mixed(${a}, ${b})`,
  complexRuntimeDeps: ["mtoc2_cdiv"],
});

/**
 * Logical builtins: or, and, not.
 *
 * Functional forms of `|`, `&`, and `~`. Mirror the corresponding operator
 * implementations in runtimeOperators.ts so `or(a, b)` behaves the same as
 * `a | b`, etc. (`xor` is registered separately in reductions.ts and uses a
 * broadcast-aware implementation.) Like MATLAB, they accept numeric,
 * logical, and char input and reject complex input.
 */

import {
  isRuntimeTensor,
  isRuntimeSparseMatrix,
  type RuntimeValue,
} from "../../runtime/types.js";
import { RTV } from "../../runtime/constructors.js";
import { toBool } from "../../runtime/index.js";
import { not as notOp } from "../../runtime/runtimeOperators.js";
import { elementWiseLogicalOp } from "../../runtime/runtimeHelpers.js";
import { builtin } from "./builtin.js";
import { builtinError, realNumericArg } from "./args.js";

function applyBinaryLogical(
  args: RuntimeValue[],
  op: (x: number, y: number) => number,
  scalarFn: (a: boolean, b: boolean) => boolean
): RuntimeValue {
  const a = realNumericArg(args[0]);
  const b = realNumericArg(args[1]);
  if (
    isRuntimeTensor(a) ||
    isRuntimeSparseMatrix(a) ||
    isRuntimeTensor(b) ||
    isRuntimeSparseMatrix(b)
  ) {
    return elementWiseLogicalOp(a, b, op);
  }
  return RTV.logical(scalarFn(toBool(a), toBool(b)));
}

// ── or ──────────────────────────────────────────────────────────────────

builtin({
  name: "or",
  help: {
    signatures: ["TF = or(A, B)"],
    description:
      "Logical OR. Functional form of A | B; returns a logical scalar or tensor.",
  },
  nargin: 2,
  apply: args =>
    applyBinaryLogical(
      args,
      (x, y) => (x !== 0 || y !== 0 ? 1 : 0),
      (a, b) => a || b
    ),
});

// ── and ─────────────────────────────────────────────────────────────────

builtin({
  name: "and",
  help: {
    signatures: ["TF = and(A, B)"],
    description:
      "Logical AND. Functional form of A & B; returns a logical scalar or tensor.",
  },
  nargin: 2,
  apply: args =>
    applyBinaryLogical(
      args,
      (x, y) => (x !== 0 && y !== 0 ? 1 : 0),
      (a, b) => a && b
    ),
});

// ── not ─────────────────────────────────────────────────────────────────

builtin({
  name: "not",
  help: {
    signatures: ["TF = not(A)"],
    description:
      "Logical negation. Functional form of ~A; returns a logical scalar or tensor.",
  },
  nargin: 1,
  apply: args =>
    notOp(
      realNumericArg(args[0], () =>
        builtinError("MATLAB:Not:operandsNotReal", "Operands must be real.")
      )
    ),
});

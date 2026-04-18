/**
 * Logical builtins: or, and, not.
 *
 * Functional forms of `|`, `&`, and `~`. Mirror the corresponding operator
 * implementations in runtimeOperators.ts so `or(a, b)` behaves the same as
 * `a | b`, etc. (`xor` is registered separately in reductions.ts and uses a
 * broadcast-aware implementation.)
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
import { type JitType } from "../../jit/jitTypes.js";
import { defineBuiltin, type BuiltinCase } from "./types.js";

// ── Type-level matcher ──────────────────────────────────────────────────

const LOGICAL_KINDS: ReadonlySet<JitType["kind"]> = new Set([
  "number",
  "boolean",
  "tensor",
  "sparse_matrix",
]);

function binaryLogicalMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 2) return null;
  if (!LOGICAL_KINDS.has(argTypes[0].kind)) return null;
  if (!LOGICAL_KINDS.has(argTypes[1].kind)) return null;
  if (argTypes[0].kind === "tensor" || argTypes[1].kind === "tensor") {
    return [{ kind: "tensor", isComplex: false, isLogical: true }];
  }
  if (
    argTypes[0].kind === "sparse_matrix" ||
    argTypes[1].kind === "sparse_matrix"
  ) {
    return [{ kind: "tensor", isComplex: false, isLogical: true }];
  }
  return [{ kind: "boolean" }];
}

function isTensorLike(v: unknown): boolean {
  return (
    isRuntimeTensor(v as RuntimeValue) ||
    isRuntimeSparseMatrix(v as RuntimeValue)
  );
}

function applyBinaryLogical(
  args: RuntimeValue[],
  op: (x: number, y: number) => number,
  scalarFn: (a: boolean, b: boolean) => boolean
): RuntimeValue {
  const a = args[0];
  const b = args[1];
  if (isTensorLike(a) || isTensorLike(b)) {
    return elementWiseLogicalOp(a, b, op);
  }
  return RTV.logical(scalarFn(toBool(a), toBool(b)));
}

// ── or ──────────────────────────────────────────────────────────────────

const orCase: BuiltinCase = {
  match: binaryLogicalMatch,
  apply: args =>
    applyBinaryLogical(
      args,
      (x, y) => (x !== 0 || y !== 0 ? 1 : 0),
      (a, b) => a || b
    ),
};

defineBuiltin({
  name: "or",
  help: {
    signatures: ["TF = or(A, B)"],
    description:
      "Logical OR. Functional form of A | B; returns a logical scalar or tensor.",
  },
  cases: [orCase],
});

// ── and ─────────────────────────────────────────────────────────────────

const andCase: BuiltinCase = {
  match: binaryLogicalMatch,
  apply: args =>
    applyBinaryLogical(
      args,
      (x, y) => (x !== 0 && y !== 0 ? 1 : 0),
      (a, b) => a && b
    ),
};

defineBuiltin({
  name: "and",
  help: {
    signatures: ["TF = and(A, B)"],
    description:
      "Logical AND. Functional form of A & B; returns a logical scalar or tensor.",
  },
  cases: [andCase],
});

// ── not ─────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "not",
  help: {
    signatures: ["TF = not(A)"],
    description:
      "Logical negation. Functional form of ~A; returns a logical scalar or tensor.",
  },
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (!LOGICAL_KINDS.has(argTypes[0].kind)) return null;
        if (
          argTypes[0].kind === "tensor" ||
          argTypes[0].kind === "sparse_matrix"
        ) {
          return [{ kind: "tensor", isComplex: false, isLogical: true }];
        }
        return [{ kind: "boolean" }];
      },
      apply: args => notOp(args[0]),
    },
  ],
});

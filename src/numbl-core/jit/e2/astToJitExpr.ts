/**
 * e2 — minimal AST `Expr` → `JitExpr` lowerer.
 *
 * Only handles the whitelist that `classify.ts` accepts: Number, Ident,
 * whitelisted Binary/Unary, whitelisted FuncCall. Types are read from
 * the live runtime environment (the caller passes in a per-name
 * `JitType` lookup), so there's no cross-branch unification.
 *
 * The classifier already replaced opaque subtrees with synthetic Ident
 * nodes whose `name` is also in `envTypes`, so this lowerer doesn't need
 * to know about opacity — every Ident it sees has a known runtime type.
 */

import type { Expr } from "../../parser/types.js";
import { BinaryOperation } from "../../parser/types.js";
import type { JitExpr, JitType } from "../jitTypes.js";
import { E2_BUILTIN_WHITELIST } from "./classify.js";

export class E2LowerError extends Error {}

/** Pick the result type of a Binary op given operand types. Tensor wins
 *  over scalar; complex propagates. Booleans coerce to number. */
function unifyBinaryType(l: JitType, r: JitType): JitType {
  // Any tensor input yields a tensor output (elementwise). Complex
  // propagates from either side — a complex scalar on one side widens
  // the tensor result to complex (e.g. `y * 1i` where y is real).
  if (l.kind === "tensor" || r.kind === "tensor") {
    const lt = l.kind === "tensor" ? l : null;
    const rt = r.kind === "tensor" ? r : null;
    const isComplex =
      !!lt?.isComplex ||
      !!rt?.isComplex ||
      l.kind === "complex_or_number" ||
      r.kind === "complex_or_number";
    return { kind: "tensor", isComplex };
  }
  if (l.kind === "complex_or_number" || r.kind === "complex_or_number") {
    return { kind: "complex_or_number" };
  }
  return { kind: "number" };
}

/** True when both operand types are tensor-shaped. Used to reject
 *  matrix-multiplication operators (`Mul`/`Div`/`Pow` without dot)
 *  on tensor pairs — those aren't elementwise and have no per-element
 *  meaning in a fused loop. */
function bothTensor(l: JitType, r: JitType): boolean {
  return l.kind === "tensor" && r.kind === "tensor";
}

const MATRIX_OPS: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Mul,
  BinaryOperation.Div,
  BinaryOperation.LeftDiv,
  BinaryOperation.Pow,
]);

/** Binary ops the complex emitter supports element-wise. Everything
 *  outside this set on a complex result bails early at the lowerer
 *  level so the driver short-circuits before running phases 3+ on
 *  every hot-loop iteration. Mirrors the emitter's own capability
 *  (`Add`, `Sub`, `Mul`, `ElemMul` only). */
const COMPLEX_BINARY_OPS: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Add,
  BinaryOperation.Sub,
  BinaryOperation.Mul,
  BinaryOperation.ElemMul,
]);

/** Builtins that always produce a real-valued result, even on complex
 *  input (complex → real type transition). In a paired-buffer kernel
 *  these still emit a per-element pair with im = 0.0. */
const REAL_OUTPUT_BUILTINS: ReadonlySet<string> = new Set(["real", "imag"]);

/** Builtins that preserve complex-ness: `conj(real)` is real,
 *  `conj(complex)` is complex. Handled structurally in the complex
 *  emitter (flips the sign of im). */
const COMPLEX_PASSTHROUGH_BUILTINS: ReadonlySet<string> = new Set(["conj"]);

export function lowerAstToJitExpr(
  expr: Expr,
  envTypes: ReadonlyMap<string, JitType>
): JitExpr {
  switch (expr.type) {
    case "Number": {
      const value = parseFloat(expr.value);
      return {
        tag: "NumberLiteral",
        value,
        jitType: { kind: "number", exact: value },
      };
    }
    case "ImagUnit": {
      return {
        tag: "ImagLiteral",
        jitType: { kind: "complex_or_number", pureImaginary: true },
      };
    }
    case "Ident": {
      const t = envTypes.get(expr.name);
      if (!t) {
        throw new E2LowerError(
          `e2: identifier '${expr.name}' has no known type`
        );
      }
      return { tag: "Var", name: expr.name, jitType: t };
    }
    case "Binary": {
      const left = lowerAstToJitExpr(expr.left, envTypes);
      const right = lowerAstToJitExpr(expr.right, envTypes);
      if (MATRIX_OPS.has(expr.op) && bothTensor(left.jitType, right.jitType)) {
        throw new E2LowerError(
          `e2: matrix op ${expr.op} on two tensors is not elementwise`
        );
      }
      const jitType = unifyBinaryType(left.jitType, right.jitType);
      const resultIsComplex =
        (jitType.kind === "tensor" && jitType.isComplex) ||
        jitType.kind === "complex_or_number";
      if (resultIsComplex && !COMPLEX_BINARY_OPS.has(expr.op)) {
        throw new E2LowerError(
          `e2: binary op ${expr.op} not supported on complex operands`
        );
      }
      return {
        tag: "Binary",
        op: expr.op,
        left,
        right,
        jitType,
      };
    }
    case "Unary": {
      const operand = lowerAstToJitExpr(expr.operand, envTypes);
      return {
        tag: "Unary",
        op: expr.op,
        operand,
        jitType: operand.jitType,
      };
    }
    case "FuncCall": {
      if (!E2_BUILTIN_WHITELIST.has(expr.name)) {
        throw new E2LowerError(`e2: builtin '${expr.name}' not whitelisted`);
      }
      const args = expr.args.map(a => lowerAstToJitExpr(a, envTypes));
      // Result is tensor if any arg is tensor; complex if any arg is
      // complex; otherwise scalar number.
      let isTensor = false;
      let isComplex = false;
      for (const a of args) {
        if (a.jitType.kind === "tensor") {
          isTensor = true;
          if (a.jitType.isComplex) isComplex = true;
        } else if (a.jitType.kind === "complex_or_number") {
          isComplex = true;
        }
      }
      // `real` / `imag` always return real. `conj` preserves complex-ness
      // and is handled by the complex emitter. Any other builtin with a
      // complex argument bails — the emitter has no complex-aware C
      // lowering for the rest (`exp`, `sin`, ...).
      if (isComplex) {
        if (REAL_OUTPUT_BUILTINS.has(expr.name)) {
          isComplex = false;
        } else if (!COMPLEX_PASSTHROUGH_BUILTINS.has(expr.name)) {
          throw new E2LowerError(
            `e2: builtin '${expr.name}' not supported on complex input`
          );
        }
      }
      const jitType: JitType = isTensor
        ? { kind: "tensor", isComplex }
        : isComplex
          ? { kind: "complex_or_number" }
          : { kind: "number" };
      return { tag: "Call", name: expr.name, args, jitType };
    }
    default:
      throw new E2LowerError(
        `e2: AST node '${(expr as { type: string }).type}' not handled`
      );
  }
}

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
import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr, JitType } from "../../jit/jitTypes.js";
import { E2_BUILTIN_WHITELIST } from "./classify.js";

export class E2LowerError extends Error {}

const COMPARISON_OPS: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Equal,
  BinaryOperation.NotEqual,
  BinaryOperation.Less,
  BinaryOperation.LessEqual,
  BinaryOperation.Greater,
  BinaryOperation.GreaterEqual,
]);

/** Pick the result type of a Binary op given operand types. Tensor wins
 *  over scalar; complex propagates. Comparison ops produce logical
 *  tensors (so that `_isLogical` rides through to the output
 *  `RuntimeTensor` — without it, `pts(:, mask)` treats the result as
 *  a double index and fails). Booleans coerce to number. */
function unifyBinaryType(op: BinaryOperation, l: JitType, r: JitType): JitType {
  const isComparison = COMPARISON_OPS.has(op);
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
    // Comparisons always produce a real logical tensor.
    if (isComparison) {
      return { kind: "tensor", isComplex: false, isLogical: true };
    }
    return { kind: "tensor", isComplex };
  }
  if (isComparison) {
    return { kind: "boolean" };
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

export interface LowerOptions {
  /** When a `FuncCall{name, args}` has `name` in `envTypes` as a tensor,
   *  treat it as tensor indexing and lower to an `Index` node instead
   *  of looking up a builtin. Used by the e2 whole-loop kernel — the
   *  chain emitters don't set this (their classifier has already
   *  marked tensor-access FuncCalls as opaque). */
  resolveFuncCallAsTensorIndex?: boolean;
}

/** Build an `Index` JitExpr for a 1-D scalar tensor read from a
 *  validated tensor `base`. Both the `FuncCall`-as-index and `Index`
 *  AST paths share this: they differ only in how they obtain `base`. */
function buildScalarIndexRead(
  base: JitExpr,
  indexArgs: readonly Expr[],
  lower: (e: Expr) => JitExpr
): JitExpr {
  if (indexArgs.length !== 1) {
    throw new E2LowerError(`e2: multi-index tensor access not supported`);
  }
  const idx = lower(indexArgs[0]);
  if (idx.jitType.kind !== "number" && idx.jitType.kind !== "boolean") {
    throw new E2LowerError(`e2: tensor index must be a scalar number`);
  }
  return {
    tag: "Index",
    base,
    indices: [idx],
    jitType: { kind: "number" },
  };
}

export function lowerAstToJitExpr(
  expr: Expr,
  envTypes: ReadonlyMap<string, JitType>,
  options: LowerOptions = {}
): JitExpr {
  function rec(e: Expr): JitExpr {
    return lowerAstToJitExpr(e, envTypes, options);
  }
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
      const left = rec(expr.left);
      const right = rec(expr.right);
      if (MATRIX_OPS.has(expr.op) && bothTensor(left.jitType, right.jitType)) {
        throw new E2LowerError(
          `e2: matrix op ${expr.op} on two tensors is not elementwise`
        );
      }
      const jitType = unifyBinaryType(expr.op, left.jitType, right.jitType);
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
      const operand = rec(expr.operand);
      // `!x` on a tensor produces a logical tensor; on a scalar, a
      // boolean. Plus/Minus pass the operand type through.
      let jitType = operand.jitType;
      if (expr.op === UnaryOperation.Not) {
        if (jitType.kind === "tensor") {
          jitType = { kind: "tensor", isComplex: false, isLogical: true };
        } else if (
          jitType.kind === "number" ||
          jitType.kind === "boolean" ||
          jitType.kind === "complex_or_number"
        ) {
          jitType = { kind: "boolean" };
        }
      }
      return {
        tag: "Unary",
        op: expr.op,
        operand,
        jitType,
      };
    }
    case "FuncCall": {
      // If the caller opts in and `name` is a tensor in envTypes,
      // `name(idx)` is tensor indexing (MATLAB syntax overload), not a
      // builtin call. Rewrite to an Index node.
      if (options.resolveFuncCallAsTensorIndex) {
        const existingType = envTypes.get(expr.name);
        if (existingType?.kind === "tensor" && !existingType.isComplex) {
          const base: JitExpr = {
            tag: "Var",
            name: expr.name,
            jitType: existingType,
          };
          return buildScalarIndexRead(base, expr.args, rec);
        }
      }
      if (!E2_BUILTIN_WHITELIST.has(expr.name)) {
        throw new E2LowerError(`e2: builtin '${expr.name}' not whitelisted`);
      }
      const args = expr.args.map(a => rec(a));
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
    case "Index": {
      // Only the simplest form: `x(<scalar_idx>)` where `x` has a known
      // tensor type and the result is a scalar read. Used by the e2
      // whole-loop kernel. Multi-index, range-index, and other forms
      // fall through to the caller's bail path.
      const base = rec(expr.base);
      if (base.tag !== "Var") {
        throw new E2LowerError(`e2: Index base must be an Ident`);
      }
      if (base.jitType.kind !== "tensor" || base.jitType.isComplex) {
        throw new E2LowerError(`e2: Index requires a real tensor base`);
      }
      return buildScalarIndexRead(base, expr.indices, rec);
    }
    default:
      throw new E2LowerError(
        `e2: AST node '${(expr as { type: string }).type}' not handled`
      );
  }
}

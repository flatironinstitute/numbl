/**
 * e2 (experimental) — per-assign expression classifier.
 *
 * Walks an AST `Expr` and decides whether it can be compiled into a
 * single per-element C kernel. The classifier never evaluates anything
 * — it only inspects the AST shape and the names referenced.
 *
 * Whitelist:
 *   - Number, Ident
 *   - Binary with arithmetic / comparison ops
 *   - Unary Plus, Minus, Not
 *   - FuncCall to a whitelisted scalar math builtin
 *
 * Anything outside the whitelist is recorded as an "opaque root": the
 * driver is expected to evaluate that subtree via the interpreter and
 * bind the result to a fresh synthetic name, then re-classify with that
 * name in scope.
 *
 * The classifier returns a list of opaque-root subtrees and the
 * "rewritten" expression that uses synthetic names where the opaque
 * roots used to be. The driver is responsible for runtime type checks
 * and for actually evaluating the opaque subtrees.
 */

import type { Expr } from "../../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
export { BinaryOperation } from "../../parser/types.js";

// ── Whitelists ─────────────────────────────────────────────────────────

/** Scalar math builtins that map cleanly to C99. Mirrors the JS-JIT
 *  Math.* table plus pow / hypot / atan2 / etc. */
export const E2_BUILTIN_WHITELIST: ReadonlySet<string> = new Set([
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "acosh",
  "atanh",
  "exp",
  "expm1",
  "log",
  "log1p",
  "log2",
  "log10",
  "sqrt",
  "abs",
  "floor",
  "ceil",
  "fix",
  "round",
  "sign",
  "max",
  "min",
  "mod",
  "rem",
  "atan2",
  "hypot",
  "pow",
]);

/** Binary ops we'll let through without further checks. Matrix ops
 *  (Mul/Div/Pow without dot) are allowed structurally — the driver
 *  rejects them at runtime when both operands turn out to be tensors,
 *  since matrix multiplication is not per-element. */
const E2_BINARY_WHITELIST: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Add,
  BinaryOperation.Sub,
  BinaryOperation.Mul,
  BinaryOperation.ElemMul,
  BinaryOperation.Div,
  BinaryOperation.ElemDiv,
  BinaryOperation.Pow,
  BinaryOperation.ElemPow,
  BinaryOperation.Equal,
  BinaryOperation.NotEqual,
  BinaryOperation.Less,
  BinaryOperation.LessEqual,
  BinaryOperation.Greater,
  BinaryOperation.GreaterEqual,
]);

const E2_UNARY_WHITELIST: ReadonlySet<UnaryOperation> = new Set([
  UnaryOperation.Plus,
  UnaryOperation.Minus,
  UnaryOperation.Not,
]);

// ── Public types ───────────────────────────────────────────────────────

/** One opaque subtree the driver must evaluate before invoking the
 *  kernel. The classifier replaces it in `emittableExpr` with an Ident
 *  named `syntheticName`. */
export interface OpaqueRoot {
  syntheticName: string;
  expr: Expr;
}

export interface ClassifyResult {
  /** AST with opaque subtrees replaced by Ident(syntheticName) nodes. */
  emittableExpr: Expr;
  /** Subtrees the driver must evaluate via the interpreter. */
  opaqueRoots: OpaqueRoot[];
  /** Identifiers referenced in `emittableExpr` that originated from the
   *  user's environment (i.e. NOT synthetic opaque-root bindings). The
   *  driver looks these up in the env to determine input types. */
  envIdents: Set<string>;
}

// ── Implementation ─────────────────────────────────────────────────────

interface Ctx {
  opaque: OpaqueRoot[];
  envIdents: Set<string>;
  /** Counter for unique opaque-root names. */
  nextOpaqueId: number;
}

function makeOpaque(ctx: Ctx, expr: Expr): Expr {
  const name = `__e2_op_${ctx.nextOpaqueId++}`;
  ctx.opaque.push({ syntheticName: name, expr });
  // Fabricate a span — interpreter / lowerer don't read it for synthetic Idents.
  return { type: "Ident", name, span: expr.span };
}

function visit(expr: Expr, ctx: Ctx): Expr {
  switch (expr.type) {
    case "Number":
      return expr;

    case "Ident":
      ctx.envIdents.add(expr.name);
      return expr;

    case "Binary": {
      if (!E2_BINARY_WHITELIST.has(expr.op)) {
        return makeOpaque(ctx, expr);
      }
      const left = visit(expr.left, ctx);
      const right = visit(expr.right, ctx);
      return { ...expr, left, right };
    }

    case "Unary": {
      if (!E2_UNARY_WHITELIST.has(expr.op)) {
        return makeOpaque(ctx, expr);
      }
      const operand = visit(expr.operand, ctx);
      return { ...expr, operand };
    }

    case "FuncCall": {
      // We only treat as an emittable call if the name is in the
      // whitelist AND not shadowed by a local — but the driver checks
      // shadowing at runtime (it knows the env). The classifier is
      // syntactic only.
      if (!E2_BUILTIN_WHITELIST.has(expr.name)) {
        return makeOpaque(ctx, expr);
      }
      const args = expr.args.map(a => visit(a, ctx));
      return { ...expr, args };
    }

    // Everything else: opaque. Includes Index, Member, MethodCall,
    // Range, Tensor literal, AnonFunc, etc.
    default:
      return makeOpaque(ctx, expr);
  }
}

/** Classify an expression. Always succeeds — the worst case is the
 *  whole expression becomes a single opaque root, which the driver
 *  will reject. */
export function classifyExpr(expr: Expr): ClassifyResult {
  const ctx: Ctx = { opaque: [], envIdents: new Set(), nextOpaqueId: 0 };
  const emittableExpr = visit(expr, ctx);
  return {
    emittableExpr,
    opaqueRoots: ctx.opaque,
    envIdents: ctx.envIdents,
  };
}

/** Heuristic gate: an expression is "worth" JIT'ing only when it does
 *  some work — a bare Ident or Number is not. The driver also gates
 *  on tensor size at runtime; this is just a structural pre-filter to
 *  skip the cost of lowering trivial expressions. */
export function isWorthCompiling(emittableExpr: Expr): boolean {
  switch (emittableExpr.type) {
    case "Number":
    case "Ident":
      return false;
    default:
      return true;
  }
}

// ── Same-LHS assign chain detection ────────────────────────────────────

/** True when `expr` references the identifier `name` anywhere. */
function exprReferencesName(expr: Expr, name: string): boolean {
  switch (expr.type) {
    case "Ident":
      return expr.name === name;
    case "Binary":
      return (
        exprReferencesName(expr.left, name) ||
        exprReferencesName(expr.right, name)
      );
    case "Unary":
      return exprReferencesName(expr.operand, name);
    case "FuncCall":
      return expr.args.some(a => exprReferencesName(a, name));
    default:
      return false;
  }
}

/** A single classification entry for one assign in a chain. The chain
 *  emitter consumes one of these per assign in order. */
export interface ChainAssignClassification {
  /** The original AST stmt (kept so the cache can key on it). */
  stmt: import("../../parser/types.js").Stmt & { type: "Assign" };
  /** The classifier's rewritten RHS — opaque subtrees replaced by
   *  Ident(syntheticName). */
  emittableExpr: Expr;
  /** Opaque subtrees this assign contributed (driver evaluates these
   *  before the kernel call). */
  opaqueRoots: OpaqueRoot[];
  /** Identifiers referenced by this assign's emittableExpr. Includes
   *  synthetic opaque-root names. */
  envIdents: Set<string>;
  /** True if the assign reads its own LHS (e.g. `r = r + x`). For the
   *  first stmt of a chain this means the kernel needs `in_<lhs>` as
   *  an input pointer; for later stmts it just means a chain-local
   *  read. */
  selfReadsLhs: boolean;
}

export interface ChainClassification {
  /** Chain assigns, in source order. May have multiple distinct LHSs. */
  assigns: ChainAssignClassification[];
}

/** Detect a chain of consecutive suppressed classifiable Assigns
 *  starting at `stmts[startIdx]`. Each LHS may be a different name;
 *  the driver decides — using full-scope liveness — whether each LHS
 *  becomes a chain-local or a materialized output buffer.
 *
 *  The chain ends at the first non-Assign, the first unsuppressed
 *  Assign, or the first Assign whose RHS classification is not worth
 *  compiling.
 *
 *  Returns null if the very first stmt isn't a chainable Assign. */
export function classifyAssignChain(
  stmts: import("../../parser/types.js").Stmt[],
  startIdx: number
): ChainClassification | null {
  const assigns: ChainAssignClassification[] = [];
  for (let i = startIdx; i < stmts.length; i++) {
    const s = stmts[i];
    if (s.type !== "Assign") break;
    if (!s.suppressed) break;
    const cls = classifyExpr(s.expr);
    if (!isWorthCompiling(cls.emittableExpr)) break;
    assigns.push({
      stmt: s,
      emittableExpr: cls.emittableExpr,
      opaqueRoots: cls.opaqueRoots,
      envIdents: cls.envIdents,
      selfReadsLhs: exprReferencesName(s.expr, s.name),
    });
  }
  if (assigns.length === 0) return null;
  return { assigns };
}

// ── Trailing reduction detection ───────────────────────────────────────

/** Reduction op names whose semantics the e2 reduction emitter knows. */
export const E2_REDUCTION_OPS: ReadonlySet<string> = new Set([
  "sum",
  "prod",
  "max",
  "min",
  "mean",
  "any",
  "all",
]);

/** Binary ops the trailing-reduction matcher recognizes for the
 *  `acc OP reduce(...)` accumulate form. */
const REDUCE_ACC_OPS: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Add,
  BinaryOperation.Sub,
  BinaryOperation.Mul,
  BinaryOperation.ElemMul,
]);

/** Commutative subset of `REDUCE_ACC_OPS` — we only accept the
 *  `reduce(...) OP acc` order for these (Sub is not commutative; the
 *  matcher would otherwise misorder the operands). */
const REDUCE_ACC_OPS_COMMUTATIVE: ReadonlySet<BinaryOperation> = new Set([
  BinaryOperation.Add,
  BinaryOperation.Mul,
  BinaryOperation.ElemMul,
]);

export interface TrailingReductionMatch {
  /** The original Assign stmt — pinned for cache identity. */
  stmt: import("../../parser/types.js").Stmt & { type: "Assign" };
  /** LHS = accumulator name. */
  accName: string;
  /** Reduction op name. */
  reduceName: string;
  /** AST expression argument to the reduction call. The driver decides
   *  whether to treat it as a Var-targeting-chain-local (for trailing-
   *  after-chain) or as a standalone elemwise expression. */
  targetExpr: Expr;
  /** True for `acc = acc OP reduce(...)`; false for `acc = reduce(...)`. */
  hasAccumulate: boolean;
  /** The accumulate op (only meaningful when `hasAccumulate` is true). */
  accOp?: BinaryOperation;
}

/** Match an Assign of the form:
 *    acc = reduce(arg)
 *    acc = acc OP reduce(arg)
 *    acc = reduce(arg) OP acc        (commutative ops only)
 *  where `reduce` is a single-argument call to a known reduction op.
 *  Returns the matched details or null. */
export function matchTrailingReduction(
  stmt: import("../../parser/types.js").Stmt
): TrailingReductionMatch | null {
  if (stmt.type !== "Assign") return null;
  const accName = stmt.name;
  const rhs = stmt.expr;

  // Pattern A: rhs = reduce(arg)
  if (
    rhs.type === "FuncCall" &&
    E2_REDUCTION_OPS.has(rhs.name) &&
    rhs.args.length === 1
  ) {
    return {
      stmt,
      accName,
      reduceName: rhs.name,
      targetExpr: rhs.args[0],
      hasAccumulate: false,
    };
  }

  // Patterns B/C: rhs = acc OP reduce(arg) or reduce(arg) OP acc
  if (rhs.type === "Binary" && REDUCE_ACC_OPS.has(rhs.op)) {
    const isAccIdent = (e: Expr): boolean =>
      e.type === "Ident" && e.name === accName;
    const isReduceCall = (e: Expr): null | { name: string; arg: Expr } => {
      if (
        e.type === "FuncCall" &&
        E2_REDUCTION_OPS.has(e.name) &&
        e.args.length === 1
      ) {
        return { name: e.name, arg: e.args[0] };
      }
      return null;
    };
    // B: acc OP reduce(arg)
    if (isAccIdent(rhs.left)) {
      const r = isReduceCall(rhs.right);
      if (r) {
        return {
          stmt,
          accName,
          reduceName: r.name,
          targetExpr: r.arg,
          hasAccumulate: true,
          accOp: rhs.op,
        };
      }
    }
    // C: reduce(arg) OP acc — commutative only
    if (isAccIdent(rhs.right) && REDUCE_ACC_OPS_COMMUTATIVE.has(rhs.op)) {
      const r = isReduceCall(rhs.left);
      if (r) {
        return {
          stmt,
          accName,
          reduceName: r.name,
          targetExpr: r.arg,
          hasAccumulate: true,
          accOp: rhs.op,
        };
      }
    }
  }
  return null;
}

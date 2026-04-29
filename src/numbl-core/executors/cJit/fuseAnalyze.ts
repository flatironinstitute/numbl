/**
 * c-jit-fuse — structural feasibility analysis on a single AST `Assign`.
 *
 * Walks the AST directly (not the JS-JIT IR — that pipeline is sized
 * for whole-function lowering). Determines whether the RHS expression
 * tree can be compiled to a single-pass element-wise C kernel, and
 * extracts the tensor / scalar input identifiers in a stable order.
 *
 * Returns `null` when the stmt isn't a fusable shape — caller (the
 * lowering pipeline) then leaves the stmt for the interpreter.
 *
 * Today's scope: real-only, plain-Ident LHS, same-shape tensor
 * leaves. Broadcasting, complex tensors, and in-place mutation are
 * not handled yet.
 */
import { BinaryOperation, type Expr, type Stmt } from "../../parser/types.js";
import type { Environment } from "../../interpreter/types.js";
import type { RuntimeValue } from "../../runtime/types.js";

/** Per-leaf classification: tensor, scalar (number), or literal. */
export type LeafKind = "tensor" | "scalar";

export interface FuseClassification {
  /** Output variable name (LHS of the Assign). */
  readonly outputName: string;
  /** Tensor inputs in the order they should appear as C parameters. */
  readonly tensorInputs: readonly string[];
  /** Scalar inputs in the order they should appear as C parameters. */
  readonly scalarInputs: readonly string[];
  /** The RHS expression, kept as AST for codegen to walk. */
  readonly rhs: Expr;
  /** Stable cache key projected from the AST shape and per-leaf
   *  classification. Drops sizes — those are runtime-only and live
   *  in `propose()`. */
  readonly cacheKey: string;
  /** Per-name kind, used by codegen to map identifiers to either
   *  `tN[i]` (tensor) or `sN` (scalar). */
  readonly kindOf: ReadonlyMap<string, LeafKind>;
}

/** Whitelisted unary-elementwise builtins. The C codegen emits a
 *  bare `<name>(arg)` for these. `abs` → `fabs`. */
const FUSE_UNARY_BUILTINS: ReadonlySet<string> = new Set([
  "exp",
  "log",
  "log2",
  "log10",
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "sqrt",
  "abs",
  "floor",
  "ceil",
  "round",
]);

/** Determine whether `stmt` is a fusable Assign and, if so, return
 *  its classification. Returns null on any structural mismatch. */
export function analyzeFuse(
  stmt: Stmt,
  env: Environment
): FuseClassification | null {
  if (stmt.type !== "Assign") return null;

  const outputName = stmt.name;
  const rhs = stmt.expr;

  const tensorInputs: string[] = [];
  const scalarInputs: string[] = [];
  const seen = new Set<string>();
  const kindOf = new Map<string, LeafKind>();

  // Walk the RHS, collecting leaves and verifying all inner nodes
  // are element-wise feasible.
  const ok = walkExpr(rhs, env, {
    seen,
    kindOf,
    tensorInputs,
    scalarInputs,
    output: outputName,
  });
  if (!ok) return null;

  // At least one tensor input — otherwise this is a pure-scalar expr
  // and another path (interpreter / future c-jit-loop scope) handles it.
  if (tensorInputs.length === 0) return null;

  const cacheKey = buildCacheKey(rhs, kindOf);

  return {
    outputName,
    tensorInputs,
    scalarInputs,
    rhs,
    cacheKey,
    kindOf,
  };
}

interface WalkCtx {
  readonly seen: Set<string>;
  readonly kindOf: Map<string, LeafKind>;
  readonly tensorInputs: string[];
  readonly scalarInputs: string[];
  readonly output: string;
}

/** Walk a sub-expr. Returns true iff the entire subtree is fusable. */
function walkExpr(e: Expr, env: Environment, ctx: WalkCtx): boolean {
  switch (e.type) {
    case "Number":
      // Literal — no leaf binding needed; the codegen emits the
      // value inline.
      return true;
    case "Ident": {
      // The output var may also appear as a leaf (e.g. `r = r .* y +
      // 3.0`). That's fine — we treat it as just another tensor /
      // scalar input bound from env.
      const name = e.name;
      if (ctx.seen.has(name)) return true;
      const v = env.get(name);
      const kind = classifyEnvValue(v);
      if (kind === null) return false;
      ctx.seen.add(name);
      ctx.kindOf.set(name, kind);
      if (kind === "tensor") ctx.tensorInputs.push(name);
      else ctx.scalarInputs.push(name);
      return true;
    }
    case "Binary":
      if (!isFusableBinary(e, env, ctx)) return false;
      return walkExpr(e.left, env, ctx) && walkExpr(e.right, env, ctx);
    case "Unary":
      // Plus / Minus are element-wise. Not (~) is allowed but
      // produces a logical — we restrict to numeric tensors so skip
      // it for v0.
      if (e.op !== "Plus" && e.op !== "Minus") return false;
      return walkExpr(e.operand, env, ctx);
    case "FuncCall":
      if (!FUSE_UNARY_BUILTINS.has(e.name)) return false;
      if (e.args.length !== 1) return false;
      return walkExpr(e.args[0], env, ctx);
    default:
      return false;
  }
}

/** Element-wise feasibility for a Binary node. The expression's
 *  leaves are walked separately by `walkExpr`. */
function isFusableBinary(
  e: Expr & { type: "Binary" },
  env: Environment,
  ctx: WalkCtx
): boolean {
  switch (e.op) {
    case BinaryOperation.Add:
    case BinaryOperation.Sub:
    case BinaryOperation.ElemMul:
    case BinaryOperation.ElemDiv:
      return true;
    case BinaryOperation.Mul:
    case BinaryOperation.Div:
      // Plain `*` and `/` are matrix ops between two tensors but
      // act element-wise (broadcasting) when at least one operand
      // is a scalar. Walk both operands to determine their kinds.
      // This is a peek — we re-walk in walkExpr; the cost is tiny.
      return (
        atLeastOneScalar(e.left, env, ctx) ||
        atLeastOneScalar(e.right, env, ctx)
      );
    default:
      return false;
  }
}

/** True if the expression is statically a scalar (literal or scalar
 *  Ident). Conservative — returns false for anything we can't prove
 *  scalar (so a Binary subtree containing a tensor is not scalar). */
function atLeastOneScalar(e: Expr, env: Environment, _ctx: WalkCtx): boolean {
  if (e.type === "Number") return true;
  if (e.type === "Unary") return atLeastOneScalar(e.operand, env, _ctx);
  if (e.type === "Ident") {
    const v = env.get(e.name);
    return classifyEnvValue(v) === "scalar";
  }
  // For Binary and FuncCall subtrees, we can't decide cheaply
  // without recursing — return false. Worst case: we miss a fusion
  // opportunity (the interpreter handles it).
  return false;
}

function classifyEnvValue(v: RuntimeValue | undefined): LeafKind | null {
  if (v === undefined) return null;
  if (typeof v === "number") return "scalar";
  if (typeof v === "boolean") return "scalar";
  if (typeof v === "object" && v !== null && "kind" in v) {
    const k = (v as { kind?: string }).kind;
    if (k === "tensor") {
      // Only real tensors with Float64 data are supported in v0.
      const t = v as {
        kind: "tensor";
        data: ArrayBufferView;
        imag?: ArrayBufferView;
      };
      if (t.imag !== undefined) return null; // complex tensor
      if (!(t.data instanceof Float64Array)) return null;
      return "tensor";
    }
  }
  return null;
}

/** Build a cache key from the AST shape + leaf kinds. The result
 *  must be the same for any two dispatches that produce the same C
 *  source. Sizes / values aren't included — those affect runtime
 *  cost, not codegen. */
function buildCacheKey(
  rhs: Expr,
  kindOf: ReadonlyMap<string, LeafKind>
): string {
  const parts: string[] = [];
  emitKey(rhs, kindOf, parts);
  return parts.join("|");
}

function emitKey(
  e: Expr,
  kindOf: ReadonlyMap<string, LeafKind>,
  out: string[]
): void {
  switch (e.type) {
    case "Number":
      out.push(`N:${e.value}`);
      return;
    case "Ident":
      out.push(`${kindOf.get(e.name) === "tensor" ? "T" : "S"}:${e.name}`);
      return;
    case "Binary":
      out.push(`B:${e.op}`);
      emitKey(e.left, kindOf, out);
      emitKey(e.right, kindOf, out);
      return;
    case "Unary":
      out.push(`U:${e.op}`);
      emitKey(e.operand, kindOf, out);
      return;
    case "FuncCall":
      out.push(`C:${e.name}`);
      for (const a of e.args) emitKey(a, kindOf, out);
      return;
    default:
      out.push(`?:${e.type}`);
  }
}

/**
 * Cheap gate for the `stmt` shape (`jit/stmtExecutor.ts`): which
 * interpreted statements are worth offering to the JIT on their own.
 *
 * A candidate is a suppressed `y = <expr>` whose right-hand side is an
 * element-wise tree (literals, variables, element-wise operators and
 * calls, optionally under a full `sum` / `prod` / `mean`) with at least
 * two operations the JS kernels fuse, since a single operation gains
 * nothing over the interpreter's own kernel. The verdict depends only on
 * the AST, so it is computed once per statement node and cached; a
 * dispatch then costs a `WeakMap` lookup. Lowering still decides what
 * the names mean (a "call" may turn out to index a variable), so this is
 * a heuristic, never a correctness gate.
 */

import {
  BinaryOperation,
  UnaryOperation,
  type Expr,
  type Stmt,
} from "../parser/types.js";

export interface StmtCandidate {
  /** Identifiers the RHS reads, call names excluded. */
  readonly reads: readonly string[];
  /** Names the RHS calls; a variable of one of these names would make
   *  the call an index, which the unit would not see. */
  readonly calls: readonly string[];
  /** Times the statement was dispatched (see `tryLowerStmt`). */
  runs: number;
  /** Dispatches left to skip after a decline, and the declines in a row
   *  (see `stmtDeclined`). */
  skip: number;
  misses: number;
}

/** Back off after a decline (inputs too small, or no executor took the
 *  unit): skip the next 1, 3, 7, ... up to 1023 dispatches, so a hot
 *  statement that keeps declining costs a counter decrement. A success
 *  resets it. */
export function stmtDeclined(c: StmtCandidate): void {
  c.misses = Math.min(c.misses + 1, 10);
  c.skip = (1 << c.misses) - 1;
}

/** Binary operators the JS kernels fuse (`*` and `/` only with a scalar
 *  side; lowering checks that). */
const FUSED_BINARY = new Set<BinaryOperation>([
  BinaryOperation.Add,
  BinaryOperation.Sub,
  BinaryOperation.Mul,
  BinaryOperation.ElemMul,
  BinaryOperation.Div,
  BinaryOperation.ElemDiv,
]);

/** Element-wise calls the JS kernels fuse (see `JS_FUSION`). */
const FUSED_CALLS = new Set(["abs", "floor", "ceil", "fix", "sqrt"]);

const REDUCERS = new Set(["sum", "prod", "mean"]);

/** Fewest fused operations that make a statement worth compiling. */
const MIN_FUSED_OPS = 2;

const cache = new WeakMap<Stmt, StmtCandidate | null>();

/** The statement's candidate data, or null when it is not one. */
export function stmtCandidate(stmt: Stmt): StmtCandidate | null {
  let c = cache.get(stmt);
  if (c === undefined) {
    c = analyze(stmt);
    cache.set(stmt, c);
  }
  return c;
}

function analyze(stmt: Stmt): StmtCandidate | null {
  if (stmt.type !== "Assign" || !stmt.suppressed) return null;
  let e = stmt.expr;
  let ops = 0;
  const reads = new Set<string>();
  const calls = new Set<string>();
  const call = callOf(e);
  if (call !== null && REDUCERS.has(call.name) && call.args.length === 1) {
    calls.add(call.name);
    e = call.args[0];
    ops++;
  }
  const count = countOps(e, reads, calls);
  if (count === null || ops + count < MIN_FUSED_OPS) return null;
  return { reads: [...reads], calls: [...calls], runs: 0, skip: 0, misses: 0 };
}

/** `f(args)` as parsed either way (a call, or an index of a name). */
function callOf(e: Expr): { name: string; args: Expr[] } | null {
  if (e.type === "FuncCall") return { name: e.name, args: e.args };
  if (e.type === "Index" && e.base.type === "Ident") {
    return { name: e.base.name, args: e.indices };
  }
  return null;
}

/** Fused operations in an element-wise tree, collecting the names it
 *  reads; null when the tree has a node no kernel takes. */
function countOps(
  e: Expr,
  reads: Set<string>,
  calls: Set<string>
): number | null {
  switch (e.type) {
    case "Number":
      return 0;
    case "Ident":
      reads.add(e.name);
      return 0;
    case "Binary": {
      if (!FUSED_BINARY.has(e.op)) return null;
      const l = countOps(e.left, reads, calls);
      const r = countOps(e.right, reads, calls);
      return l === null || r === null ? null : l + r + 1;
    }
    case "Unary": {
      if (e.op !== UnaryOperation.Minus && e.op !== UnaryOperation.Plus) {
        return null;
      }
      const n = countOps(e.operand, reads, calls);
      return n === null ? null : n + (e.op === UnaryOperation.Minus ? 1 : 0);
    }
    case "FuncCall":
    case "Index": {
      const call = callOf(e);
      if (call === null || !FUSED_CALLS.has(call.name)) return null;
      if (call.args.length !== 1) return null;
      calls.add(call.name);
      const n = countOps(call.args[0], reads, calls);
      return n === null ? null : n + 1;
    }
    default:
      return null;
  }
}

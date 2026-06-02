/**
 * Function-handle inlining for the JIT.
 *
 * A function handle that crosses a compile boundary — a loop input
 * (`f = @...; for ...; f(x); end`) or a call argument
 * (`applyfn(@sq, x)`) — arrives at the executor as an opaque
 * `function_handle` value the JIT can't type. But a *capture-free*
 * handle is fully described by its source AST, so instead of marshaling
 * it across the boundary we inline its `@...` definition as an in-scope
 * handle constant inside the synthesized spec. That reduces the boundary
 * case to the already-supported in-scope case (`lowerAnonFunc` /
 * `lowerFuncHandle` + `dispatchHandleCall`), which lowers identically on
 * the JS and C backends — no value-adapter or ABI work.
 *
 * "Capture-free" is the key constraint: a handle that closes over an
 * outer variable can't be inlined, because the inlined copy would
 * re-resolve that name in the spec's scope (a different value, or
 * unresolved) rather than the by-value snapshot the handle took. We
 * detect captures against the handle's own recorded definition
 * environment, so the test reflects what the handle actually closed
 * over, not a syntactic guess.
 *
 * A second constraint applies at the relocation site. A body name that
 * is NOT a variable in the def env resolves at definition time to a
 * function (or is undefined). After relocation into the loop's synthetic
 * scope, that same name may be bound as a variable (a loop input or
 * assigned local). Re-lowering would then capture the loop variable
 * instead of resolving the function/undefined the handle saw — silently
 * turning `@(t) sq(t)` from a function call into an array index, or
 * masking an undefined-variable error. So the caller passes the set of
 * names that will be in scope at the relocation site, and we decline
 * when a free body name collides with it. A name that is purely a
 * function (never an env variable) is not in that set, so the common
 * `@(t) sq(t)` case still inlines; only the genuine shadowing case
 * declines (and falls back to the interpreter).
 */

import {
  isRuntimeFunction,
  type RuntimeFunction,
  type RuntimeValue,
} from "../runtime/types.js";
import type { Expr } from "../parser/index.js";

/** The defining AST a `RuntimeFunction` records (see RuntimeFunction
 *  `handleAst`): an `@(p) body` or `@name`. */
type HandleAst =
  | Extract<Expr, { type: "AnonFunc" }>
  | Extract<Expr, { type: "FuncHandle" }>;

/** Collect every identifier referenced in an expression — variable
 *  reads, function-call names, index/member bases. Over-approximates
 *  (a 0-arg function reference looks like a variable), which is safe
 *  here: we only use the result to test membership in the handle's
 *  capture environment, and function names aren't env variables. */
function collectReferenced(e: Expr, out: Set<string>): void {
  switch (e.type) {
    case "Ident":
      out.add(e.name);
      break;
    case "FuncCall":
      out.add(e.name);
      for (const a of e.args) collectReferenced(a, out);
      break;
    case "Binary":
      collectReferenced(e.left, out);
      collectReferenced(e.right, out);
      break;
    case "Unary":
      collectReferenced(e.operand, out);
      break;
    case "Index":
    case "IndexCell":
      collectReferenced(e.base, out);
      for (const idx of e.indices) collectReferenced(idx, out);
      break;
    case "Range":
      collectReferenced(e.start, out);
      if (e.step) collectReferenced(e.step, out);
      collectReferenced(e.end, out);
      break;
    case "Tensor":
    case "Cell":
      for (const row of e.rows)
        for (const el of row) collectReferenced(el, out);
      break;
    case "Member":
      collectReferenced(e.base, out);
      break;
    case "MemberDynamic":
      collectReferenced(e.base, out);
      collectReferenced((e as { nameExpr: Expr }).nameExpr, out);
      break;
    case "MethodCall":
      collectReferenced(e.base, out);
      for (const a of e.args) collectReferenced(a, out);
      break;
    case "AnonFunc":
      // Nested anon: its own params shadow ours, but for a conservative
      // capture test we just collect everything; shadowed names that
      // happen to be in our capture env would only make us decline.
      collectReferenced(e.body, out);
      break;
    default:
      break;
  }
}

/**
 * If `val` is a function handle the JIT can inline as a compile-time
 * constant in a spec compiled for `compileFile`, return its `@...`
 * defining Expr; otherwise return null.
 *
 * Inlinable iff:
 *  - it's a `RuntimeFunction` carrying a recorded `handleAst`, AND
 *  - it was defined in `compileFile` (so a `@name` target resolves to
 *    the same function it did at definition — file-local functions can
 *    shadow across files), AND
 *  - for an anonymous handle, it is capture-free: no name its body
 *    references (other than its own params) exists as a variable in the
 *    handle's recorded definition environment, AND no free body name
 *    collides with `relocScopeNames` (the names that will be in scope
 *    where the handle is inlined).
 *
 * Named (`@name`) handles are always capture-free. Anything the lowerer
 * ultimately can't compile (e.g. a builtin target) simply makes the
 * enclosing unit decline to the interpreter — still correct.
 */
export function inlinableHandleExpr(
  val: unknown,
  compileFile: string,
  relocScopeNames?: ReadonlySet<string>
): Expr | null {
  if (!isRuntimeFunction(val as RuntimeValue)) return null;
  const fn = val as RuntimeFunction;
  const ast = fn.handleAst as HandleAst | undefined;
  if (ast === undefined) return null;
  if (fn.handleDefFile !== compileFile) return null;

  if (ast.type === "FuncHandle") return ast;

  // Anonymous: require capture-free against the recorded env.
  const env = fn.capturedEnv;
  if (env === undefined) return null;
  const referenced = new Set<string>();
  collectReferenced(ast.body, referenced);
  const params = new Set(ast.params);
  for (const name of referenced) {
    if (params.has(name)) continue;
    if (env.vars.has(name)) return null; // a real capture — not inlinable
    // A non-captured free name resolved to a function/undefined at
    // definition time. If the relocation scope binds it as a variable,
    // inlining would silently re-resolve it to that variable — decline.
    if (relocScopeNames?.has(name)) return null;
  }
  return ast;
}

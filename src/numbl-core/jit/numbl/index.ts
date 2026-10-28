/**
 * Single intra-repo bridge module for everything the vendored JIT
 * (`src/numbl-core/jit/*`) consumes from the rest of numbl-core. Every
 * `numbl-core/*` path the JIT depends on appears here
 * exactly once, so:
 *
 * - a numbl rename / move surfaces as a single tsc error in this file
 *   rather than scattering across the vendored compiler's call sites;
 * - the surface area between the JIT and the rest of numbl is visible at
 *   a glance.
 *
 * `parser/index.ts` re-exports through this module.
 */

// ── Parser (AST shape, lexer dispatch, operator enums) ──────────────────

export { BinaryOperation, UnaryOperation } from "../../parser/index.js";

export type {
  Span,
  Expr,
  Stmt,
  LValue,
  AbstractSyntaxTree,
} from "../../parser/index.js";

// ── Resolver / code index (workspace function dispatch) ─────────────────

export type { CodeIndex } from "../../workspace/codeIndex.js";
export { resolveFunction } from "../../resolution/resolveFunction.js";

export type { CallSite } from "../../runtime/runtimeHelpers.js";
export type { ItemType } from "../../itemTypes.js";
export type { ClassInfo } from "../../workspace/classInfo.js";

// ── Plot builtin names ──────────────────────────────────────────────────

export { PLOT_ALL_NAMES } from "../../plotting/dispatch.js";

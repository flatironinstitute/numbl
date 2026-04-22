/**
 * Public entry point to the emit/ subpackage.
 *
 * Outside callers (the outer-function orchestrator in
 * [../assemble.ts](../assemble.ts)) see only:
 *   - `emitStmts` — walk a statement list and push C source lines.
 *   - `shapeExprsFor` — derive (d0, d1) C expressions for a dynamic
 *     tensor output (used inside this package too, but also by the
 *     outer orchestrator for param-output shape plumbing).
 *
 * Everything else is private to the emit/ subpackage.
 */
export { emitStmts } from "./stmt.js";
export { shapeExprsFor } from "./tensor.js";

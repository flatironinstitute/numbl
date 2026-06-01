/**
 * Parser re-exports for mtoc2 consumers. The actual cross-import to
 * the sibling numbl checkout lives in `src/numbl/index.ts` — this
 * module is the historic stable path used by ~25 call sites and
 * preserves their import shape unchanged.
 *
 * Structural AST drift is caught at `tsc` time; semantic drift is
 * caught by the cross-runner. The pinned numbl SHA lives in
 * `NUMBL_VERSION`, which CI checks out against.
 */

export {
  parseMFile,
  SyntaxError,
  BinaryOperation,
  UnaryOperation,
} from "../numbl/index.js";

export type {
  Span,
  Expr,
  Stmt,
  LValue,
  AbstractSyntaxTree,
} from "../numbl/index.js";

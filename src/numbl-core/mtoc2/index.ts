/**
 * Single bridge module for everything numbl imports from mtoc2 (the
 * sibling repo wired in as `"mtoc2": "file:../mtoc2"` in package.json).
 * Symmetric inverse of mtoc2's `src/numbl/index.ts` bridge: every
 * `mtoc2/*` path numbl depends on appears here exactly once, so an
 * mtoc2 rename / move surfaces as a single tsc error in this file
 * rather than scattering across consumer call sites.
 *
 * Imports go through the `mtoc2` package name (resolved via the
 * symlinked node_modules entry) so numbl's `skipLibCheck` setting
 * applies to mtoc2's internal `.js` runtime snippets — without it,
 * numbl's tsc would reject the .js-only files in mtoc2's
 * builtins/runtime/ tree.
 */

export {
  compileSpec,
  type CompileSpecArgs,
  type CompileSpecResult,
} from "mtoc2/src/jit/compileSpec.js";

export { Workspace } from "mtoc2/src/workspace/workspace.js";
export type { WorkspaceFile } from "mtoc2/src/workspace/workspace.js";

export { Lowerer } from "mtoc2/src/lowering/lower.js";

export {
  UnsupportedConstruct,
  TypeError as Mtoc2TypeError,
} from "mtoc2/src/lowering/errors.js";

export {
  scalarDouble,
  scalarComplex,
  scalarLogical,
  tensorDouble,
  tensorComplex,
  tensorDoubleFromDims,
  tensorComplexFromDims,
  withoutExact,
  UNKNOWN,
  VOID,
} from "mtoc2/src/lowering/types.js";

export type { Type } from "mtoc2/src/lowering/types.js";

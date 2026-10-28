/**
 * The scope a name is resolved from.
 */

import type { Environment } from "../interpreter/types.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import type { CodeIndex } from "../workspace/codeIndex.js";
import type { FunctionIndex, ImportEntry } from "../workspace/functionIndex.js";

const NO_IMPORTS: readonly ImportEntry[] = [];

/**
 * Where a name is looked up from: the running file, the class and method
 * whose code is running (if any), and the environment holding the variables
 * and nested functions in scope (a nested function is owned by the
 * environment of the function that defines it). A `Scope` is also a
 * `CallSite`, the form `resolveFunction` takes.
 */
export class Scope implements CallSite {
  constructor(
    readonly code: CodeIndex,
    readonly file: string,
    readonly className?: string,
    readonly methodName?: string,
    readonly env?: Environment
  ) {}

  get index(): FunctionIndex {
    return this.code.functionIndex;
  }

  /** The `import` statements in effect. numbl tracks imports per file. */
  get imports(): readonly ImportEntry[] {
    return this.index.fileImports.get(this.file) ?? NO_IMPORTS;
  }
}

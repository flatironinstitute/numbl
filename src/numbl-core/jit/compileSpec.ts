/**
 * JS-JIT entry point: the shared pipeline (`compile.ts`) with the JS
 * target.
 *
 * Produces an emitted JS module that returns a factory `($h) => specFn`.
 * The caller evaluates the source with `new Function(source)()`, invokes
 * the factory once to bind its host hooks, then calls `specFn(...args)`
 * per dispatch.
 */

import { emitJsProgram, type EmitJsResult } from "./codegen/emitJs.js";
import {
  compileWith,
  type CompileSpecArgs,
  type CompileTarget,
} from "./compile.js";
import type { Type } from "./lowering/types.js";
import { JS_FUSION } from "./lowering/elemwiseKernels.js";

export type { CompileSpecArgs } from "./compile.js";

export interface CompileSpecResult {
  /** Mangled spec name (the IRFunc cName). */
  cName: string;
  /** Full module source, ready for `new Function(source)()`. */
  source: string;
  /** Names of runtime snippets activated by emit. */
  activatedSnippets: ReadonlyArray<string>;
  /** The spec's output types, in order. The host needs them to convert
   *  returned values back: a struct and a class instance are both plain
   *  JS objects in the emitted code, so only the type says which is which. */
  outputTypes: ReadonlyArray<Type>;
  /** Files named by the statement locations a raise reports, by index
   *  (decode with `decodeLoc`). */
  locFiles: ReadonlyArray<string>;
  /** The entry spec can call itself (directly or through its callees). */
  recursive: boolean;
  /** Element-wise kernels in the entry spec (`ElemwiseKernel`). */
  kernels: number;
}

/** JS models an unassigned variable as `undefined` and checks reads at
 *  run time, and fuses only exact element-wise operations (`JS_FUSION`). */
export const jsTarget: CompileTarget<EmitJsResult> = {
  definiteAssignment: false,
  fusion: JS_FUSION,
  emit: (prog, exposeSpec) => emitJsProgram(prog, { exposeSpec }),
};

export function compileSpec(args: CompileSpecArgs): CompileSpecResult {
  const { cName, entry, recursive, kernels, emitted } = compileWith(
    jsTarget,
    args
  );
  return {
    cName,
    source: emitted.source,
    activatedSnippets: emitted.activatedSnippets,
    outputTypes: entry.outputTypes,
    locFiles: emitted.locFiles,
    recursive,
    kernels,
  };
}

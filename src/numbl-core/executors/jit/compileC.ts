/**
 * Types of the koffi C-JIT pipeline, kept apart from the Node-only
 * implementation (`compileC.node.ts`, which imports `node:fs`,
 * `node:child_process`, ...) so modules the browser bundle reaches can
 * import them. The C backend itself reaches a run only through its host
 * (`NumblHost.cjit`, installed by `nodeCJitBackend` in `cjit.node.ts`).
 */

import type { NativeBridge } from "../../workspace/types.js";

/** The koffi function reference returned by `lib.func(decl)`. The
 *  actual call signature is controlled by the caller's declaration —
 *  type as variadic and let the caller cast. */
export type CFn = (...args: unknown[]) => unknown;

export interface CompiledC {
  readonly fn: CFn;
  /** koffi `Lib` instance — useful for binding auxiliary symbols
   *  (`free` from libc, allocator helpers, etc.) without re-opening
   *  the .so. */
  readonly lib: KoffiLib;
  /** Path to the .so on disk — diagnostics only. */
  readonly libPath: string;
  /** Cache hit (true) vs. fresh cc invocation (false). */
  readonly cacheHit: boolean;
}

/** koffi's runtime-shaped lib object. Loosely typed because koffi
 *  doesn't ship TS types for its declaration-string API. */
export interface KoffiLib {
  func(declaration: string): CFn;
}

/** Signature of the real compile-and-load. The CLI registers an
 *  implementation that shells out to `cc` and `dlopen`s the result
 *  via koffi. */
export type CompileAndLoadCImpl = (
  source: string,
  declaration: string,
  bridge: NativeBridge
) => CompiledC;

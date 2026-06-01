/**
 * Browser-safe surface for the koffi C-JIT pipeline.
 *
 * The real implementation (`compileAndLoadCImpl`) lives in
 * `compileC.node.ts` because it imports Node-only modules (`node:fs`,
 * `node:child_process`, `node:os`, `node:crypto`) that vite/rollup
 * cannot bundle for the browser. The CLI's bootstrap registers the
 * Node implementation here at startup via `setCompileAndLoadCImpl()`;
 * any browser bundle leaves it unregistered, so the koffi-backed
 * executors decline (their proposals already gate on
 * `bridge.koffi !== undefined`).
 *
 * Types stay in this file so the koffi executors can import them
 * without dragging Node-only deps into the static dep graph.
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

let impl: CompileAndLoadCImpl | null = null;

/** Install the Node-side C-JIT implementation. Called by `cli.ts`
 *  during CLI bootstrap; browser bundles never call this, so `impl`
 *  stays null and the koffi executors decline. */
export function setCompileAndLoadCImpl(fn: CompileAndLoadCImpl): void {
  impl = fn;
}

/** Compile (or look up cached) and load. Delegates to whatever was
 *  registered via `setCompileAndLoadCImpl()`; throws if no
 *  implementation is available (= running in a browser host that
 *  somehow got past the koffi executor's `bridge.koffi` gate). */
export function compileAndLoadC(
  source: string,
  declaration: string,
  bridge: NativeBridge
): CompiledC {
  if (impl === null) {
    throw new Error(
      "compileAndLoadC: no Node-side implementation registered " +
        "(call setCompileAndLoadCImpl from cli.ts during bootstrap)"
    );
  }
  return impl(source, declaration, bridge);
}

/** Read a previously-compiled C source from disk. Diagnostic helper.
 *  Set alongside the main impl in the Node bootstrap. */
let readCachedCSourceImpl: ((hash: string) => string | null) | null = null;

export function setReadCachedCSourceImpl(
  fn: (hash: string) => string | null
): void {
  readCachedCSourceImpl = fn;
}

export function readCachedCSource(hash: string): string | null {
  return readCachedCSourceImpl?.(hash) ?? null;
}

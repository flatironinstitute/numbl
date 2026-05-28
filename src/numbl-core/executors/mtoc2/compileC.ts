/**
 * Compile + cache + load for the mtoc2-emitted C-JIT pipeline.
 *
 * Cache layout: `~/.cache/numbl/mtoc2-c-jit/<sha>.so` (one per unique
 * `(declaration, source, flags)` triple). On hit, `dlopen` and reuse —
 * no `cc` invocation. On miss, write `<sha>.c`, exec `cc`, then
 * `dlopen`.
 *
 * The koffi function declaration is owned by the caller — each
 * executor (call / loop / top-level) builds the prototype that matches
 * its spec's signature. This module is `source-string in → koffi-callable
 * + dlopen handle out`.
 *
 * Adapted from the retired in-tree C-JIT's `compile.ts`; the only
 * meaningful difference is the cache directory and a friendlier error
 * message that calls out the mtoc2 emit pipeline.
 */

import { createHash } from "crypto";
import { execSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type { NativeBridge } from "../../workspace/types.js";

const CACHE_DIR = join(homedir(), ".cache", "numbl", "mtoc2-c-jit");

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

/**
 * Compile (or look up cached) and load. The cache key is the SHA of
 * `(flags, declaration, source)` so two different signatures or
 * compile-flag sets against the same source get distinct artifacts.
 */
export function compileAndLoadC(
  source: string,
  declaration: string,
  bridge: NativeBridge
): CompiledC {
  if (!existsSync(CACHE_DIR)) {
    mkdirSync(CACHE_DIR, { recursive: true });
  }

  const cc = process.env.NUMBL_CC || "cc";
  // -O3 + -march=native for the JIT specialization payoff.
  // -fPIC + -shared for the .so build. -std=c11 because mtoc2's
  // emitted helpers use designated initializers and _Complex_I.
  // -fno-math-errno avoids errno-update overhead in transcendentals
  // (numbl's interpreter doesn't surface errno either; matches
  // semantics). -ffast-math is OFF by default — reduction ordering
  // and signed-zero / NaN handling are observable, and the cross-
  // boundary copy already costs more than libmvec vectorization
  // typically saves.
  const flags = [
    "-O3",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-march=native",
    "-fopenmp-simd",
    "-fno-math-errno",
    "-Wall",
  ];
  const libs = process.platform === "linux" ? ["-lm"] : ["-lm"];

  const flagKey = `${flags.join(" ")} ${libs.join(" ")}`;
  const hash = createHash("sha256")
    .update(`${flagKey}\n${declaration}\n${source}`)
    .digest("hex")
    .slice(0, 32);
  const libPath = join(CACHE_DIR, `${hash}.so`);
  const srcPath = join(CACHE_DIR, `${hash}.c`);

  let cacheHit = true;
  if (!existsSync(libPath)) {
    cacheHit = false;
    writeFileSync(srcPath, source);
    try {
      execSync(
        `${cc} ${flags.join(" ")} -o ${libPath} ${srcPath} ${libs.join(" ")}`,
        { stdio: ["ignore", "ignore", "pipe"] }
      );
    } catch (e) {
      const stderr =
        e && typeof e === "object" && "stderr" in e
          ? String((e as { stderr?: Buffer }).stderr ?? "")
          : "";
      throw new Error(`mtoc2-c-jit compile failed for ${srcPath}\n${stderr}`);
    }
  }

  const lib = bridge.load(libPath) as KoffiLib;
  const fn = lib.func(declaration);
  return { fn, lib, libPath, cacheHit };
}

/** Read a previously-compiled C source from disk. Diagnostic helper. */
export function readCachedCSource(hash: string): string | null {
  const srcPath = join(CACHE_DIR, `${hash}.c`);
  if (!existsSync(srcPath)) return null;
  return readFileSync(srcPath, "utf8");
}

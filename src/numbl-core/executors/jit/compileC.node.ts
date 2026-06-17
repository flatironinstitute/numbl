/**
 * Node-only implementation of the koffi C-JIT pipeline.
 *
 * Cache layout: `~/.cache/numbl/mtoc2-c-jit/<sha>.so` (one per unique
 * `(declaration, source, flags)` triple). On hit, `dlopen` and reuse —
 * no `cc` invocation. On miss, write `<sha>.c`, exec `cc`, then
 * `dlopen`.
 *
 * This file is split out of `compileC.ts` so the browser bundle stays
 * free of `node:fs` / `node:child_process` imports — vite/rollup
 * cannot resolve them for the browser target. Browser bundles import
 * only `compileC.ts` (types + a runtime-injected stub); the CLI
 * imports this file at bootstrap and calls `setCompileAndLoadCImpl()`.
 */

import { createHash } from "node:crypto";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { NativeBridge } from "../../workspace/types.js";
import {
  setCompileAndLoadCImpl,
  setReadCachedCSourceImpl,
  type CompiledC,
  type KoffiLib,
} from "./compileC.js";
import { setCSnippets } from "../../jit/codegen/runtime.js";
import { C_SNIPPETS } from "../../jit/builtins/runtime/snippets.c.gen.js";

const CACHE_DIR = join(homedir(), ".cache", "numbl", "mtoc2-c-jit");

/** Compile (or look up cached) and load. The cache key is the SHA of
 *  `(flags, declaration, source)` so two different signatures or
 *  compile-flag sets against the same source get distinct artifacts. */
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

/** Register both implementations with the browser-safe surface. The
 *  CLI imports this module at bootstrap to wire them in. */
export function registerNodeCompileC(): void {
  setCompileAndLoadCImpl(compileAndLoadC);
  setReadCachedCSourceImpl(readCachedCSource);
  // Feed the C runtime-snippet table to the codegen runtime. Kept here (not
  // a static import in codegen/runtime.ts) so the ~300 KB of C source stays
  // out of the browser worker bundle.
  setCSnippets(C_SNIPPETS);
}

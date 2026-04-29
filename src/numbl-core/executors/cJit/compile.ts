/**
 * c-jit compile + cache + load.
 *
 * Cache layout: `~/.cache/numbl/c-jit/<sha>.so` (one per unique
 * (declaration, source) pair). On hit, dlopen and reuse — no `cc`
 * invocation. On miss, write `<sha>.c`, exec `cc`, dlopen.
 *
 * The koffi function declaration is fully owned by the caller —
 * different executors (c-jit-loop, c-jit-fuse, ...) emit the
 * declaration that matches their ABI. This module only handles
 * source-string-in / koffi-callable-out.
 */

import { createHash } from "crypto";
import { execSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type { NativeBridge } from "../../workspace/types.js";

const CACHE_DIR = join(homedir(), ".cache", "numbl", "c-jit");

/**
 * The koffi function reference returned by `lib.func(decl)`. The
 * actual call signature is controlled by the caller's declaration —
 * type as variadic and let the caller cast as needed.
 */
export type CFn = (...args: unknown[]) => unknown;

export interface CompiledC {
  readonly fn: CFn;
  /** Path to the .so on disk — useful for diagnostics. */
  readonly libPath: string;
  /** Cache hit (true) vs. fresh cc invocation (false). */
  readonly cacheHit: boolean;
}

/** koffi's runtime-shaped lib object. */
interface KoffiLib {
  func(declaration: string): CFn;
}

/**
 * Compile (or look up cached) and load. The cache key is the SHA of
 * `(declaration, source)` so the same source compiled with different
 * koffi declarations gets distinct cache entries (defensive — usually
 * the declaration matches the source's signature).
 */
export function compileAndLoad(
  source: string,
  declaration: string,
  bridge: NativeBridge
): CompiledC {
  if (!existsSync(CACHE_DIR)) {
    mkdirSync(CACHE_DIR, { recursive: true });
  }

  const hash = createHash("sha256")
    .update(`${declaration}\n${source}`)
    .digest("hex")
    .slice(0, 32);
  const libPath = join(CACHE_DIR, `${hash}.so`);
  const srcPath = join(CACHE_DIR, `${hash}.c`);

  let cacheHit = true;
  if (!existsSync(libPath)) {
    cacheHit = false;
    writeFileSync(srcPath, source);

    const cc = process.env.NUMBL_CC || "cc";
    const flags = [
      "-O2",
      "-fPIC",
      "-shared",
      "-std=c11",
      "-march=native",
      "-fno-math-errno",
      "-Wall",
    ];
    try {
      execSync(`${cc} ${flags.join(" ")} -o ${libPath} ${srcPath}`, {
        stdio: ["ignore", "ignore", "pipe"],
      });
    } catch (e) {
      const stderr =
        e && typeof e === "object" && "stderr" in e
          ? String((e as { stderr?: Buffer }).stderr ?? "")
          : "";
      throw new Error(`c-jit compile failed for ${srcPath}\n${stderr}`);
    }
  }

  const lib = bridge.load(libPath) as KoffiLib;
  const fn = lib.func(declaration);
  return { fn, libPath, cacheHit };
}

/**
 * Read a previously-compiled C source from disk. Diagnostic helper
 * for verifying cache hits.
 */
export function readCachedSource(hash: string): string | null {
  const srcPath = join(CACHE_DIR, `${hash}.c`);
  if (!existsSync(srcPath)) return null;
  return readFileSync(srcPath, "utf8");
}

/**
 * Node-only implementation of the koffi C-JIT pipeline.
 *
 * Cache layout: `~/.cache/numbl/c-jit/<sha>.so` (or `$NUMBL_CJIT_CACHE`),
 * one per unique `(declaration, source without comments, flags)` triple.
 * On hit, `dlopen` and reuse, with no `cc` invocation. On miss, write
 * `<sha>.c`, exec `cc`, then `dlopen`.
 *
 * Node-only (`node:fs`, `node:child_process`): reached only through the
 * C backend (`cBackend.ts`), which a host gets from `cjit.node.ts`.
 */

import { createHash } from "node:crypto";
import { execSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { NativeBridge } from "../../workspace/types.js";
import { stripCComments } from "../../jit/codegen/cSource.js";
import type { CompiledC, KoffiLib } from "./compileC.js";

/** `NUMBL_CJIT_CACHE` overrides the cache directory (e.g. an empty one to
 *  time cold compiles). */
function cacheDir(): string {
  return (
    process.env.NUMBL_CJIT_CACHE || join(homedir(), ".cache", "numbl", "c-jit")
  );
}

/** Compile (or look up cached) and load. The cache key is the SHA of
 *  `(flags, declaration, source)` so two different signatures or
 *  compile-flag sets against the same source get distinct artifacts.
 *  Comments are left out of the key: the emitter's spec names do not
 *  depend on argument types (see `cSource.ts`), so a module that differs
 *  from a cached one only in its type comments reuses its object. */
export function compileAndLoadC(
  source: string,
  declaration: string,
  bridge: NativeBridge
): CompiledC {
  const dir = cacheDir();
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const cc = process.env.NUMBL_CC || "cc";
  // -O3 + -march=native for the JIT specialization payoff.
  // -fPIC + -shared for the .so build. -std=c11 because the\n  // emitted helpers use designated initializers and _Complex_I.
  // -fno-math-errno avoids errno-update overhead in transcendentals
  // (numbl's interpreter doesn't surface errno either; matches
  // semantics). -ffast-math is OFF by default — reduction ordering
  // and signed-zero / NaN handling are observable, and the cross-
  // boundary copy already costs more than libmvec vectorization
  // typically saves. -ffp-contract=off: gcc in ISO mode never fuses
  // `a*b + c` into an FMA, but clang does by default, which changes
  // last-bit results; every --opt level must round each op separately.
  const flags = [
    "-O3",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-march=native",
    "-fopenmp-simd",
    "-fno-math-errno",
    "-ffp-contract=off",
    "-Wall",
  ];
  const libs = process.platform === "linux" ? ["-lm"] : ["-lm"];

  const flagKey = `${flags.join(" ")} ${libs.join(" ")}`;
  const hash = createHash("sha256")
    .update(`${flagKey}\n${declaration}\n${stripCComments(source)}`)
    .digest("hex")
    .slice(0, 32);
  const libPath = join(dir, `${hash}.so`);
  const srcPath = join(dir, `${hash}.c`);

  let cacheHit = true;
  if (!existsSync(libPath)) {
    cacheHit = false;
    // Other numbl processes may compile the same unit concurrently and
    // load `libPath` as soon as it exists, so build under unique names and
    // rename into place (atomic within the cache directory).
    const tmp = `${hash}.${process.pid}.${Date.now()}`;
    const tmpSrc = join(dir, `${tmp}.c`);
    const tmpLib = join(dir, `${tmp}.so`);
    writeFileSync(tmpSrc, source);
    try {
      execSync(
        `${cc} ${flags.join(" ")} -o ${tmpLib} ${tmpSrc} ${libs.join(" ")}`,
        { stdio: ["ignore", "ignore", "pipe"] }
      );
    } catch (e) {
      const stderr =
        e && typeof e === "object" && "stderr" in e
          ? String((e as { stderr?: Buffer }).stderr ?? "")
          : "";
      rmSync(tmpLib, { force: true });
      renameSync(tmpSrc, srcPath);
      throw new Error(`C-JIT compile failed for ${srcPath}\n${stderr}`);
    }
    renameSync(tmpSrc, srcPath);
    renameSync(tmpLib, libPath);
  }

  const lib = bridge.load(libPath) as KoffiLib;
  const fn = lib.func(declaration);
  return { fn, lib, libPath, cacheHit };
}

/** Read a previously-compiled C source from disk. Diagnostic helper. */
export function readCachedCSource(hash: string): string | null {
  const srcPath = join(cacheDir(), `${hash}.c`);
  if (!existsSync(srcPath)) return null;
  return readFileSync(srcPath, "utf8");
}

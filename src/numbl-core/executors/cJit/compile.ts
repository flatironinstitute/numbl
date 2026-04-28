/**
 * c-jit-loop compile + cache + load.
 *
 * Cache layout: `~/.cache/numbl/c-jit/<sha>.so` (one per unique C
 * source). On hit, dlopen and reuse — no `cc` invocation. On miss,
 * write `<sha>.c`, exec `cc`, dlopen.
 *
 * Compilation is synchronous (Node-side `execSync`). The user's
 * `--opt e3` workload pays the compile cost once on first call; the
 * warm-up call in `scalar_bench` lands the artifact in cache before
 * the timed call.
 */

import { createHash } from "crypto";
import { execSync } from "child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type { NativeBridge } from "../../workspace/types.js";

const CACHE_DIR = join(homedir(), ".cache", "numbl", "c-jit");

/**
 * Description of a C-JIT artifact's runtime ABI. The koffi bindings
 * read this to declare the function pointer.
 */
export interface CFnSignature {
  /** Symbol name in the .so. Must match the function name in the C source. */
  readonly fnName: string;
  /** Number of input parameters (all are `double`). */
  readonly nInputs: number;
  /** Number of output slots written to via `*out`. */
  readonly nOutputs: number;
}

/**
 * The koffi function reference returned by `lib.func(decl)`. We type
 * it as a callable so the caller can invoke it directly:
 *   `cfn(out, a, b, ...)`.
 */
export type CFn = (out: Float64Array, ...inputs: number[]) => void;

export interface CompiledC {
  readonly fn: CFn;
  /** Path to the .so on disk — useful for diagnostics. */
  readonly libPath: string;
  /** Cache hit (true) vs. fresh cc invocation (false). */
  readonly cacheHit: boolean;
}

/** koffi's runtime-shaped lib object. We poke at it via `lib.func(...)`. */
interface KoffiLib {
  func(declaration: string): CFn;
}

/**
 * Compile (or look up cached) and load. Returns a callable C function
 * with the koffi-exposed `(out, ...inputs)` shape. Throws on cc
 * failure or koffi load failure.
 */
export function compileAndLoad(
  cSource: string,
  sig: CFnSignature,
  bridge: NativeBridge
): CompiledC {
  if (!existsSync(CACHE_DIR)) {
    mkdirSync(CACHE_DIR, { recursive: true });
  }

  const hash = createHash("sha256")
    .update(`${sig.fnName}|${sig.nInputs}|${sig.nOutputs}|${cSource}`)
    .digest("hex")
    .slice(0, 32);
  const libPath = join(CACHE_DIR, `${hash}.so`);
  const srcPath = join(CACHE_DIR, `${hash}.c`);

  let cacheHit = true;
  if (!existsSync(libPath)) {
    cacheHit = false;
    writeFileSync(srcPath, cSource);

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
      throw new Error(`c-jit-loop: cc failed for ${srcPath}\n${stderr}`);
    }
  }

  const lib = bridge.load(libPath) as KoffiLib;
  const params = Array.from({ length: sig.nInputs }, (_, i) => `double a${i}`);
  const decl = `void ${sig.fnName}(double *out${
    params.length > 0 ? ", " + params.join(", ") : ""
  })`;
  const fn = lib.func(decl);
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

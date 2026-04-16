/**
 * C-JIT compilation driver.
 *
 * Takes generated C source + N-API shim, compiles them into a `.node`
 * shared library, loads it via `createRequire`, and returns the exported
 * native function.
 *
 * Strategy:
 *   1. Content-addressed cache under ~/.cache/numbl/c-jit/ — the hash
 *      includes the source bytes plus compiler/node/platform/numbl
 *      versions, so any input change (including switching Node versions)
 *      forces a recompile.
 *   2. On cache miss, write `src.c` into a fresh tmpdir and shell out to
 *      the C compiler (`$NUMBL_CC` or `cc`) with `-shared -fPIC`.
 *   3. Load with `createRequire(import.meta.url)(cachedPath)`.
 *
 * All failures return null so the caller falls back transparently to the
 * JS-JIT path.
 */

import { createRequire } from "module";
import { execFileSync } from "child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  writeFileSync,
  copyFileSync,
  readFileSync,
} from "fs";
import { createHash } from "crypto";
import { homedir, tmpdir } from "os";
import { join } from "path";

import { NUMBL_VERSION } from "../../../version.js";

export interface CompiledCFn {
  fn: (...args: number[]) => number;
  cachedPath: string;
}

// ── Environment / tool discovery ───────────────────────────────────────

/** Resolved C-JIT environment. `null` means C-JIT is unavailable. */
interface CEnv {
  cc: string;
  ccVersion: string;
  nodeHeadersDir: string;
  cacheDir: string;
  /** All non-path compile flags that will be passed on every invocation
   *  (everything except the `-o <out>` and `<src>` positional args). */
  flags: string[];
}

let _envCache: CEnv | null | undefined;
let _envFailureReason: string | undefined;

function discoverNodeHeaders(): string | undefined {
  const ver = process.versions.node;
  const candidates: string[] = [];
  // node-gyp caches headers here on Linux/macOS:
  candidates.push(
    join(homedir(), ".cache", "node-gyp", ver, "include", "node")
  );
  // npm's node-gyp may use ~/.electron-gyp or similar — ignore those.
  // System-installed node on some distros:
  const execDir = process.execPath;
  // <exec>/.. is typically bin/, so headers may be at ../include/node
  candidates.push(
    join(execDir, "..", "..", "include", "node"),
    join(execDir, "..", "include", "node")
  );
  for (const p of candidates) {
    if (existsSync(join(p, "node_api.h"))) return p;
  }
  return undefined;
}

/**
 * Probe whether the compiler accepts a flag. Done via a no-link preprocess
 * of an empty source with `-Werror` so unknown/rejected flags cause a
 * non-zero exit. Fast (~10ms) and portable across gcc/clang.
 */
function compilerAcceptsFlag(cc: string, flag: string): boolean {
  try {
    execFileSync(
      cc,
      ["-Werror", flag, "-E", "-x", "c", "-", "-o", "/dev/null"],
      {
        input: "",
        stdio: ["pipe", "ignore", "ignore"],
        timeout: 5_000,
      }
    );
    return true;
  } catch {
    return false;
  }
}

function tryInstallHeaders(log?: (m: string) => void): string | undefined {
  const ver = process.versions.node;
  const headerRoot = join(homedir(), ".cache", "node-gyp", ver);
  log?.(`C-JIT: installing Node headers for v${ver} via node-gyp...`);
  try {
    execFileSync("npx", ["node-gyp", "install", `--target=${ver}`], {
      stdio: "pipe",
      timeout: 120_000,
    });
  } catch {
    return undefined;
  }
  const inc = join(headerRoot, "include", "node");
  if (existsSync(join(inc, "node_api.h"))) return inc;
  return undefined;
}

function getCEnv(log?: (m: string) => void): CEnv | null {
  if (_envCache !== undefined) return _envCache;

  const cc = process.env.NUMBL_CC || "cc";
  let ccVersion: string;
  try {
    ccVersion = execFileSync(cc, ["--version"], {
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 5_000,
    })
      .toString()
      .split("\n")[0]
      .trim();
  } catch {
    _envFailureReason = `C compiler '${cc}' not found (set NUMBL_CC to override)`;
    log?.(`C-JIT disabled: ${_envFailureReason}`);
    _envCache = null;
    return null;
  }

  let nodeHeadersDir = discoverNodeHeaders();
  if (!nodeHeadersDir) {
    nodeHeadersDir = tryInstallHeaders(log);
  }
  if (!nodeHeadersDir) {
    _envFailureReason =
      "Node API headers not found (tried ~/.cache/node-gyp and exec dir; " +
      "npx node-gyp install also failed). Run `npx node-gyp install` manually.";
    log?.(`C-JIT disabled: ${_envFailureReason}`);
    _envCache = null;
    return null;
  }

  const cacheDir = join(homedir(), ".cache", "numbl", "c-jit");
  try {
    mkdirSync(cacheDir, { recursive: true });
  } catch (e) {
    _envFailureReason = `cannot create cache dir ${cacheDir}: ${String(e)}`;
    log?.(`C-JIT disabled: ${_envFailureReason}`);
    _envCache = null;
    return null;
  }

  const flags: string[] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    `-I${nodeHeadersDir}`,
  ];
  if (process.platform === "darwin") {
    flags.push("-undefined", "dynamic_lookup");
  }

  // Probed-default flags. Each is added only if the compiler accepts it,
  // so defaults stay portable across gcc/clang on Linux/macOS (including
  // Apple Silicon clang, which accepts `-march=native` on modern Xcode
  // but rejected it on older versions).
  //
  // `-march=native` tunes for the host CPU — safe for JIT, since the
  // compiled artifact lives in a per-host cache (~/.cache/numbl/c-jit)
  // and is never shipped to another machine.
  //
  // Opt out with `NUMBL_NO_NATIVE_CFLAGS=1` if debugging a portability
  // issue or comparing output across machines.
  const skipNativeDefaults = !!process.env.NUMBL_NO_NATIVE_CFLAGS;
  if (!skipNativeDefaults) {
    for (const flag of ["-march=native"]) {
      if (compilerAcceptsFlag(cc, flag)) flags.push(flag);
    }
  }

  const userFlags = process.env.NUMBL_CFLAGS;
  if (userFlags) {
    for (const f of userFlags.split(/\s+/).filter(Boolean)) flags.push(f);
  }

  _envCache = { cc, ccVersion, nodeHeadersDir, cacheDir, flags };

  // Print the compiler banner once per process so users of --opt 2 can
  // see exactly what flags their code is being built with. Goes to stderr
  // so it never contaminates stdout script output.
  process.stderr.write(`C-JIT: ${cc} ${flags.join(" ")} -lm  (${ccVersion})\n`);

  return _envCache;
}

// ── Cache key ──────────────────────────────────────────────────────────

function computeSourceHash(cSource: string, shim: string, env: CEnv): string {
  const h = createHash("sha256");
  h.update(cSource);
  h.update("\n---shim---\n");
  h.update(shim);
  h.update("\n---env---\n");
  h.update(env.ccVersion);
  h.update(process.version);
  h.update(process.platform);
  h.update(process.arch);
  h.update(NUMBL_VERSION);
  return h.digest("hex");
}

// ── Public API ─────────────────────────────────────────────────────────

/**
 * Compile + load a C function. Returns null on any failure; caller should
 * fall back to JS-JIT.
 */
export function compileAndLoad(
  cSource: string,
  shim: string,
  exportName: string,
  log?: (m: string) => void
): CompiledCFn | null {
  const env = getCEnv(log);
  if (!env) return null;

  const hash = computeSourceHash(cSource, shim, env);
  const cachedPath = join(env.cacheDir, `${hash}.node`);

  // Fast path: cached compilation.
  if (!existsSync(cachedPath)) {
    const built = compileToCache(cSource, shim, cachedPath, env, log);
    if (!built) return null;
  }

  try {
    const req = createRequire(import.meta.url);
    // Bust the require cache so re-loads of a freshly-written path pick up.
    // createRequire's cache is keyed on path — first load is always fresh.
    const mod = req(cachedPath) as Record<string, unknown>;
    const fn = mod[exportName];
    if (typeof fn !== "function") {
      log?.(
        `C-JIT: loaded module missing export '${exportName}' at ${cachedPath}`
      );
      return null;
    }
    return {
      fn: fn as (...args: number[]) => number,
      cachedPath,
    };
  } catch (e) {
    log?.(`C-JIT: load failed for ${cachedPath}: ${String(e)}`);
    return null;
  }
}

function compileToCache(
  cSource: string,
  shim: string,
  cachedPath: string,
  env: CEnv,
  log?: (m: string) => void
): boolean {
  let buildDir: string;
  try {
    buildDir = mkdtempSync(join(tmpdir(), "numbl-c-jit-"));
  } catch (e) {
    log?.(`C-JIT: mkdtemp failed: ${String(e)}`);
    return false;
  }

  const srcPath = join(buildDir, "src.c");
  const outPath = join(buildDir, "out.node");

  // Concatenate user C code + N-API shim into a single TU so the shim can
  // reference the user function directly.
  const full = `${cSource}\n\n${shim}\n`;
  try {
    writeFileSync(srcPath, full);
  } catch (e) {
    log?.(`C-JIT: write src failed: ${String(e)}`);
    return false;
  }

  const args = [...env.flags, "-o", outPath, srcPath, "-lm"];

  try {
    execFileSync(env.cc, args, {
      stdio: ["ignore", "ignore", "pipe"],
      timeout: 60_000,
    });
  } catch (e) {
    const err = e as { stderr?: Buffer; message: string };
    const stderr = err.stderr ? err.stderr.toString() : err.message;
    log?.(
      `C-JIT: compile failed\ncommand: ${env.cc} ${args.join(" ")}\nstderr:\n${stderr}`
    );
    // Leave the build dir for post-mortem when verbose.
    return false;
  }

  try {
    copyFileSync(outPath, cachedPath);
  } catch (e) {
    log?.(`C-JIT: cache copy failed: ${String(e)}`);
    return false;
  }

  return true;
}

// ── Introspection / testing helpers ────────────────────────────────────

/** Clear the in-memory cEnv cache (for tests). */
export function resetCEnvForTesting(): void {
  _envCache = undefined;
  _envFailureReason = undefined;
}

/** Return why C-JIT isn't available, or undefined if it is. */
export function cJitUnavailableReason(): string | undefined {
  if (_envCache === null) return _envFailureReason;
  return undefined;
}

/** Count cached .node files for debugging. */
export function cJitCacheSize(): number {
  const env = _envCache;
  if (!env) return 0;
  try {
    return readdirSync(env.cacheDir).filter(n => n.endsWith(".node")).length;
  } catch {
    return 0;
  }
}

/** Read a cached .node file's compiled hash name (for smoke test). */
export function readCachedBuild(cachedPath: string): Buffer {
  return readFileSync(cachedPath);
}

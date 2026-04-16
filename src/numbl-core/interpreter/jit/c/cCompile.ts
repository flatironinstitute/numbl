/**
 * C-JIT compilation driver (koffi path).
 *
 * Takes generated C source, compiles it into a `.so` shared library,
 * loads it via koffi, and returns the declared function.
 *
 * Strategy:
 *   1. Content-addressed cache under ~/.cache/numbl/c-jit/ — the hash
 *      includes the source bytes plus compiler/platform/numbl versions,
 *      so any input change forces a recompile.
 *   2. On cache miss, write `src.c` into a fresh tmpdir and shell out to
 *      the C compiler (`$NUMBL_CC` or `cc`) with `-shared -fPIC`.
 *   3. Load with koffi.load() and declare the function.
 *
 * No Node API headers are needed — functions are plain C with raw types.
 * No NAPI_MODULE_INIT, no module registration, no exit hooks.
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
import { dirname, join } from "path";
import { fileURLToPath } from "url";

import { NUMBL_VERSION } from "../../../version.js";

export interface CompiledCFn {
  fn: (...args: unknown[]) => unknown;
  cachedPath: string;
}

// ── Environment / tool discovery ───────────────────────────────────────

interface CEnv {
  cc: string;
  ccVersion: string;
  cacheDir: string;
  flags: string[];
  opsLibPath: string;
  opsHeaderDir: string;
}

let _envCache: CEnv | null | undefined;
let _envFailureReason: string | undefined;

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

function discoverNumblRepoRoot(): string | undefined {
  let cur = dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 12; i++) {
    const pkg = join(cur, "package.json");
    if (existsSync(pkg)) {
      try {
        const j = JSON.parse(readFileSync(pkg, "utf-8")) as { name?: string };
        if (j.name === "numbl") return cur;
      } catch {
        /* malformed package.json — keep walking */
      }
    }
    const parent = dirname(cur);
    if (parent === cur) return undefined;
    cur = parent;
  }
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

  const cacheDir = join(homedir(), ".cache", "numbl", "c-jit");
  try {
    mkdirSync(cacheDir, { recursive: true });
  } catch (e) {
    _envFailureReason = `cannot create cache dir ${cacheDir}: ${String(e)}`;
    log?.(`C-JIT disabled: ${_envFailureReason}`);
    _envCache = null;
    return null;
  }

  const repoRoot = discoverNumblRepoRoot();
  if (!repoRoot) {
    throw new Error(
      "C-JIT (--opt 2): cannot locate numbl repo root from " +
        fileURLToPath(import.meta.url) +
        " (searched up 12 levels for a package.json with name='numbl')."
    );
  }
  const opsLibPath = join(repoRoot, "build", "Release", "numbl_ops.a");
  const opsHeaderDir = join(repoRoot, "native", "ops");
  if (!existsSync(opsLibPath)) {
    throw new Error(
      `C-JIT (--opt 2): missing prebuilt libnumbl_ops static archive at\n  ${opsLibPath}\n` +
        `Run \`npm run build:addon\` to build it (this also produces numbl_addon.node).`
    );
  }
  if (!existsSync(join(opsHeaderDir, "numbl_ops.h"))) {
    throw new Error(
      `C-JIT (--opt 2): missing libnumbl_ops header at\n  ${join(opsHeaderDir, "numbl_ops.h")}`
    );
  }

  const flags: string[] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    `-I${opsHeaderDir}`,
  ];
  if (process.platform === "darwin") {
    flags.push("-undefined", "dynamic_lookup");
  }

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

  _envCache = {
    cc,
    ccVersion,
    cacheDir,
    flags,
    opsLibPath,
    opsHeaderDir,
  };

  process.stderr.write(
    `C-JIT: ${cc} ${flags.join(" ")} ${opsLibPath} -lm  (${ccVersion})\n`
  );

  return _envCache;
}

// ── Cache key ──────────────────────────────────────────────────────────

function computeSourceHash(cSource: string, env: CEnv): string {
  const h = createHash("sha256");
  h.update(cSource);
  h.update("\n---env---\n");
  h.update(env.ccVersion);
  h.update(process.version);
  h.update(process.platform);
  h.update(process.arch);
  h.update(NUMBL_VERSION);
  try {
    h.update(readFileSync(env.opsLibPath));
  } catch {
    /* getCEnv already validated existence */
  }
  return h.digest("hex");
}

// ── koffi loading ─────────────────────────────────────────────────────

let _koffi: typeof import("koffi") | null | undefined;

function getKoffi(): typeof import("koffi") | null {
  if (_koffi !== undefined) return _koffi;
  try {
    const req = createRequire(import.meta.url);
    _koffi = req("koffi") as typeof import("koffi");
    return _koffi;
  } catch {
    _koffi = null;
    return null;
  }
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Compile + load a C function via koffi. Returns null on any failure.
 *
 * `koffiSignature` is the koffi type-string for the C function, e.g.:
 *   "void jit_fn(double, double *, int64_t, double *)"
 */
export function compileAndLoad(
  cSource: string,
  koffiSignature: string,
  _cFnName: string,
  log?: (m: string) => void
): CompiledCFn | null {
  const env = getCEnv(log);
  if (!env) return null;

  const koffi = getKoffi();
  if (!koffi) {
    log?.("C-JIT: koffi not available (install with `npm install koffi`)");
    return null;
  }

  const hash = computeSourceHash(cSource, env);
  const cachedPath = join(env.cacheDir, `${hash}.so`);

  if (!existsSync(cachedPath)) {
    const built = compileToCache(cSource, cachedPath, env, log);
    if (!built) return null;
  }

  try {
    const lib = koffi.load(cachedPath);
    const fn = lib.func(koffiSignature);
    return {
      fn: fn as (...args: unknown[]) => unknown,
      cachedPath,
    };
  } catch (e) {
    log?.(`C-JIT: koffi load failed for ${cachedPath}: ${String(e)}`);
    return null;
  }
}

function compileToCache(
  cSource: string,
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
  const outPath = join(buildDir, "out.so");

  try {
    writeFileSync(srcPath, cSource);
  } catch (e) {
    log?.(`C-JIT: write src failed: ${String(e)}`);
    return false;
  }

  const args = [...env.flags, "-o", outPath, srcPath, env.opsLibPath, "-lm"];

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

// ── Introspection / testing helpers ───────────────────────────────────

export function resetCEnvForTesting(): void {
  _envCache = undefined;
  _envFailureReason = undefined;
}

export function cJitUnavailableReason(): string | undefined {
  if (_envCache === null) return _envFailureReason;
  return undefined;
}

export function cJitCacheSize(): number {
  const env = _envCache;
  if (!env) return 0;
  try {
    return readdirSync(env.cacheDir).filter(n => n.endsWith(".so")).length;
  } catch {
    return 0;
  }
}

export function readCachedBuild(cachedPath: string): Buffer {
  return readFileSync(cachedPath);
}

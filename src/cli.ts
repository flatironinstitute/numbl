#!/usr/bin/env node

import {
  readFileSync,
  readSync,
  readdirSync,
  statSync,
  existsSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  rmSync,
  copyFileSync,
} from "fs";
import { delimiter, dirname, join, relative, resolve } from "path";
import { homedir } from "os";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import { execSync } from "child_process";
import { type PlotServerOptions } from "./cli-plot-server.js";
import { createPlotHandler } from "./cli-plot-handler.js";
import {
  getAllIBuiltinNames,
  getIBuiltinHelp,
} from "./numbl-core/interpreter/builtins/index.js";
import { setLapackBridge } from "./numbl-core/native/lapack-bridge.js";
import { setLapackBridge as setLapackBridgeNew } from "./numbl-core/native/lapack-bridge.js";
import {
  diagnoseErrors,
  formatDiagnostics,
  getSourceForFile,
} from "./numbl-core/diagnostics";
import { NUMBL_VERSION } from "./numbl-core/version.js";
import { runRepl } from "./cli-repl.js";
import { NodeFileIOAdapter } from "./cli-fileio.js";
import { NodeSystemAdapter } from "./cli-system.js";

import { executeCode } from "./numbl-core/executeCode.js";
import { parseMFile } from "./numbl-core/parser/index.js";
import { WorkspaceFile, NativeBridge } from "./numbl-core/workspace/types.js";
import { isOptLevel } from "./numbl-core/executors/plugins.js";
import { registerNodeCompileC } from "./numbl-core/executors/jit/compileC.node.js";
import { scanMFiles } from "./cli-scan.js";
import { unzipToFiles } from "./vfs/unzipToFiles.js";

// Install the Node-side koffi C-JIT implementation into the
// browser-safe stub in `compileC.ts`. Keeping the `node:fs` /
// `node:child_process` imports inside `compileC.node.ts` (only loaded
// here) is what lets the browser bundle stay green.
registerNodeCompileC();

// ── Package directory & native addon paths ───────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const packageDir = join(__dirname, "..");
const addonBuildPath = join(packageDir, "build", "Release", "numbl_addon.node");

/** Variant-specific addon path. Build-addon copies the freshly built
 *  `numbl_addon.node` here so both fast-math and no-fast-math variants
 *  can coexist; the loader picks the one matching the runtime
 *  `--no-fast-math` choice. */
function variantAddonPath(fastMath: boolean): string {
  const name = fastMath
    ? "numbl_addon.fastmath.node"
    : "numbl_addon.nofastmath.node";
  return join(packageDir, "build", "Release", name);
}

// ── Try to load the native LAPACK addon ──────────────────────────────────────

import { NATIVE_ADDON_EXPECTED_VERSION } from "./numbl-core/native/lapack-bridge.js";

function loadNativeAddon(fastMath: boolean): void {
  if (process.env.NUMBL_NO_NATIVE) return;
  const variantPath = variantAddonPath(fastMath);
  const flag = fastMath ? " --fast-math" : "";
  if (!existsSync(variantPath)) {
    console.error(
      `Warning: native addon not built${fastMath ? " with fast-math" : " with --no-fast-math"}. ` +
        `Run "npx numbl build-addon${flag}" to build it. Using JS fallbacks.`
    );
    return;
  }
  try {
    const req = createRequire(import.meta.url);
    const addon = req(variantPath);
    const addonVer =
      typeof addon.addonVersion === "function" ? addon.addonVersion() : 0;
    if (addonVer !== NATIVE_ADDON_EXPECTED_VERSION) {
      console.error(
        `Warning: native addon version mismatch (got ${addonVer}, expected ${NATIVE_ADDON_EXPECTED_VERSION}). ` +
          `Run "npx numbl build-addon${flag}" to rebuild. Using JS fallbacks.`
      );
      return;
    }
    setLapackBridge(addon);
    setLapackBridgeNew(addon);
  } catch (e) {
    console.error(
      `Warning: failed to load native addon at ${variantPath}: ${(e as Error).message}. Using JS fallbacks.`
    );
  }
}

// ── Try to load koffi for native FFI bridge ───────────────────────────────────

let nativeBridge: NativeBridge | undefined;
try {
  const req = createRequire(import.meta.url);
  const koffi = req("koffi");
  nativeBridge = { load: (path: string) => koffi.load(path), koffi };
} catch {
  // koffi not installed — native shared library support disabled
}

// scanMFiles is in cli-scan.ts (separate module to avoid circular deps with cli-fileio)
export { scanMFiles } from "./cli-scan.js";

// ── MIP core package auto-install ───────────────────────────────────────────

const MHL_URL =
  "https://github.com/mip-org/mip-core/releases/download/mip-numbl/mip-numbl-any.mhl";
const mipCoreDir = join(
  homedir(),
  ".numbl",
  "mip",
  "packages",
  "gh",
  "mip-org",
  "core",
  "mip"
);
const mipCoreSearchPath = join(mipCoreDir, "mip");
const mipCoreStampPath = join(homedir(), ".numbl", "mip", ".last-refreshed");
const MIP_REFRESH_INTERVAL_MS = 60 * 60 * 1000;

function readMipRefreshStamp(): number {
  try {
    const raw = readFileSync(mipCoreStampPath, "utf8");
    const n = parseInt(raw, 10);
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

async function ensureMipCorePackage(): Promise<void> {
  const installed = existsSync(join(mipCoreSearchPath, "mip.m"));
  const fresh = Date.now() - readMipRefreshStamp() < MIP_REFRESH_INTERVAL_MS;
  if (installed && fresh) return;
  try {
    if (!installed) console.error("Installing mip core package...");
    const resp = await fetch(MHL_URL);
    if (!resp.ok) return;
    const buf = new Uint8Array(await resp.arrayBuffer());
    const files = unzipToFiles(buf);
    rmSync(mipCoreDir, { recursive: true, force: true });
    for (const f of files) {
      const dest = join(mipCoreDir, f.path);
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, f.content);
    }
    mkdirSync(dirname(mipCoreStampPath), { recursive: true });
    writeFileSync(mipCoreStampPath, String(Date.now()));
    if (!installed) console.error("mip core package installed.");
  } catch {
    // silently ignore download / install failures; previous cache (if any) stays intact
  }
}

// /** JSON replacer that serializes Maps and Sets as plain objects/arrays. */
// function jsonReplacer(_key: string, value: unknown): unknown {
//   if (value instanceof Map) {
//     return Object.fromEntries(value);
//   }
//   if (value instanceof Set) {
//     return [...value];
//   }
//   return value;
// }

// ── Internal test runner ─────────────────────────────────────────────────────

function findTestFiles(dir: string): string[] {
  const results: string[] = [];

  function walk(current: string) {
    let entries: string[];
    try {
      entries = readdirSync(current);
    } catch {
      return;
    }
    entries.sort();
    for (const entry of entries) {
      const fullPath = join(current, entry);
      let stat;
      try {
        stat = statSync(fullPath);
      } catch {
        continue;
      }
      if (stat.isDirectory()) {
        if (
          !entry.startsWith("@") &&
          !entry.startsWith("+") &&
          entry !== "wasm"
        ) {
          walk(fullPath);
        }
      } else if (stat.isFile() && entry.endsWith(".m")) {
        const content = readFileSync(fullPath, "utf-8");
        if (content.includes("SUCCESS")) {
          results.push(fullPath);
        }
      }
    }
  }

  walk(dir);
  return results;
}

async function runTests(
  dir: string,
  optimization?: import("./numbl-core/executors/plugins.js").OptLevel,
  fastMath?: boolean
) {
  loadNativeAddon(fastMath ?? false);
  const absDir = resolve(process.cwd(), dir);
  const testFiles = findTestFiles(absDir);

  if (testFiles.length === 0) {
    console.error(`No .m test files found in ${dir}`);
    process.exit(1);
  }

  let pass = 0;
  let fail = 0;
  const failedScripts: string[] = [];

  for (const filepath of testFiles) {
    const rel = relative(process.cwd(), filepath);
    const source = readFileSync(filepath, "utf-8");
    const mainFileName = filepath;
    const scriptDir = dirname(filepath);
    const searchPaths = [scriptDir];
    const workspaceFiles = scanMFiles(scriptDir, filepath);

    try {
      const result = executeCode(
        source,
        {
          displayResults: true,
          optimization: optimization ?? "1",
          fileIO: new NodeFileIOAdapter(),
          system: new NodeSystemAdapter(),
        },
        workspaceFiles,
        mainFileName,
        searchPaths,
        nativeBridge
      );

      const outputText = result.output.join("");
      const lines = outputText.split("\n").filter(l => l.length > 0);
      const lastLine = lines.length > 0 ? lines[lines.length - 1] : "";

      if (lastLine === "SUCCESS") {
        console.log(`PASS  ${rel}`);
        pass++;
      } else {
        console.log(`FAIL  ${rel}`);
        console.log(outputText.replace(/^/gm, "      "));
        fail++;
        failedScripts.push(rel);
      }
    } catch (error) {
      const diags = diagnoseErrors(error, source, mainFileName, workspaceFiles);
      const getSource = (file: string) =>
        getSourceForFile(file, mainFileName, source, workspaceFiles);
      console.log(`FAIL  ${rel}`);
      console.log(
        `      ${formatDiagnostics(diags, getSource).replace(/\n/g, "\n      ")}`
      );
      fail++;
      failedScripts.push(rel);
    }
  }

  console.log("");
  console.log(`Results: ${pass} passed, ${fail} failed`);

  if (fail > 0) {
    console.log("");
    console.log("Failed scripts:");
    for (const s of failedScripts) {
      console.log(`  ${s}`);
    }
    process.exit(1);
  }
}

// ── Help ─────────────────────────────────────────────────────────────────────

function printHelp() {
  console.log(`Usage: numbl <command> [options]

Commands:
  run <file.m>       Run a .m file
  eval "<code>"      Evaluate inline code
  parse <file.m>     Tokenize, lex, and parse a .m file (no execution)
  run-tests [dir]    Run .m test scripts (default: numbl_test_scripts/)
  build-addon        Build native LAPACK addon (pass --fast-math to enable -ffast-math)
  info               Print machine-readable info (JSON)
  list-builtins      List available built-in functions (--no-help: only those without help text)
  serve              Start local execution server for the browser IDE
  (no command)       Start interactive REPL

Global options:
  --version, -V      Print version and exit
  --help, -h         Print this help message

Options (for serve):
  --passkey <key>    Required passkey (generated by the browser IDE)
  --port <port>      Set server port (default: 3001)

Options (for parse):
  --dump-ast <file>  Write the AST as indented JSON to <file> (default: stdout)

Options (for REPL):
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)

Options (for run and eval):
  --dump-js <file>   Write JS-JIT-generated JavaScript to file (--opt 1, and
                       the JS fallback at --opt 2)
  --dump-c <file>    Write C-JIT-generated C to file (--opt 2)
  --dump-ast         Print AST as JSON
  --verbose          Detailed logging to stderr
  --stream           NDJSON output mode
  --path <dir>       Add extra workspace directory
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)
  --opt <mode>       Optimization mode (default: 1)
                       0  — interpreter (no JIT)
                       1  — JS-JIT: type-specialize hot user functions to JS
                       2  — C-JIT: scalar/tensor kernels via cc + koffi
                            (Node only; falls back to JS-JIT otherwise)
  --fast-math        Build/load the native LAPACK addon WITH -ffast-math
                     (libmvec-vectorized transcendentals, reorder-allowed
                     reductions — faster but FP results drift and diverge
                     across --opt levels). Off by default so all --opt
                     levels agree; --no-fast-math is accepted as the
                     (now default) opt-out.

Environment variables:
  NUMBL_PATH              Extra workspace directories (separated by ${delimiter})`);
}

// ── Option parsing helpers ───────────────────────────────────────────────────

interface ParsedOptions {
  dumpJs: string | undefined;
  dumpC: string | undefined;
  dumpAst: boolean;
  verbose: boolean;
  stream: boolean;
  plot: boolean;
  plotPort: number | undefined;
  extraPaths: string[];
  positional: string[];
  profileOutput: string | undefined;
  optimization: import("./numbl-core/executors/plugins.js").OptLevel;
  fastMath: boolean;
}

function parseOptions(args: string[]): ParsedOptions {
  const opts: ParsedOptions = {
    dumpJs: undefined,
    dumpC: undefined,
    dumpAst: false,
    verbose: false,
    stream: false,
    plot: false,
    plotPort: undefined,
    extraPaths: [],
    positional: [],
    profileOutput: undefined,
    optimization: "1",
    // Default OFF: the native addon's -ffast-math reorders reductions
    // and vectorizes transcendentals, so results drift from the JIT
    // kernels and diverge across --opt levels. Opt back in with
    // `--fast-math` (and a matching `build-addon --fast-math`).
    fastMath: false,
  };

  // Seed extraPaths from NUMBL_PATH environment variable (platform path separator)
  const envPath = process.env.NUMBL_PATH;
  if (envPath) {
    const home = process.env.HOME || process.env.USERPROFILE || "";
    for (const p of envPath.split(delimiter)) {
      let trimmed = p.trim();
      if (!trimmed) continue;
      // Expand leading ~ to home directory
      if (trimmed.startsWith("~/") || trimmed === "~") {
        trimmed = home + trimmed.slice(1);
      }
      opts.extraPaths.push(resolve(process.cwd(), trimmed));
    }
  }

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--dump-js":
        i++;
        if (i >= args.length) {
          console.error("Error: --dump-js requires an output filename");
          process.exit(1);
        }
        opts.dumpJs = resolve(process.cwd(), args[i]);
        break;
      case "--dump-c":
        i++;
        if (i >= args.length) {
          console.error("Error: --dump-c requires an output filename");
          process.exit(1);
        }
        opts.dumpC = resolve(process.cwd(), args[i]);
        break;
      case "--fast-math":
        opts.fastMath = true;
        break;
      case "--no-fast-math":
        // Accepted for back-compat; this is now the default.
        opts.fastMath = false;
        break;
      case "--dump-ast":
        opts.dumpAst = true;
        break;
      case "--verbose":
        opts.verbose = true;
        break;
      case "--stream":
        opts.stream = true;
        break;
      case "--plot":
        opts.plot = true;
        break;
      case "--plot-port":
        i++;
        if (i >= args.length) {
          console.error("Error: --plot-port requires a port number");
          process.exit(1);
        }
        opts.plotPort = parseInt(args[i], 10);
        if (isNaN(opts.plotPort)) {
          console.error("Error: --plot-port requires a valid port number");
          process.exit(1);
        }
        opts.plot = true;
        break;
      case "--path":
        i++;
        if (i >= args.length) {
          console.error("Error: --path requires a directory argument");
          process.exit(1);
        }
        opts.extraPaths.push(resolve(process.cwd(), args[i]));
        break;
      case "--profile":
        i++;
        if (i >= args.length) {
          console.error("Error: --profile requires an output filename");
          process.exit(1);
        }
        opts.profileOutput = resolve(process.cwd(), args[i]);
        break;
      case "--opt":
        i++;
        if (i >= args.length) {
          console.error("Error: --opt requires a value (0 or 1)");
          process.exit(1);
        }
        {
          const v = args[i];
          if (!isOptLevel(v)) {
            console.error(`Error: --opt must be 0 or 1 (got ${v}).`);
            process.exit(1);
          }
          opts.optimization = v;
        }
        break;
      default:
        if (args[i].startsWith("-")) {
          console.error(`Unknown option: ${args[i]}`);
          process.exit(1);
        }
        opts.positional.push(args[i]);
    }
  }

  return opts;
}

// ── Command handlers ─────────────────────────────────────────────────────────

async function cmdRun(args: string[]) {
  const opts = parseOptions(args);

  if (opts.positional.length === 0) {
    console.error("Error: numbl run requires a file argument");
    process.exit(1);
  }
  if (opts.positional.length > 1) {
    console.error("Error: numbl run accepts only one file argument");
    process.exit(1);
  }

  await ensureMipCorePackage();

  const filepath = resolve(process.cwd(), opts.positional[0]);
  const code = readFileSync(filepath, "utf-8");
  const mainFileName = filepath;

  // Auto-cd into the script's directory before executing (matches MATLAB
  // "F5 in editor" behavior). The script directory then becomes the
  // first-priority implicit search path via the cwd-as-search-path feature.
  process.chdir(dirname(filepath));

  const searchPaths: string[] = [];
  const workspaceFiles: WorkspaceFile[] = [];
  for (const p of opts.extraPaths) {
    searchPaths.push(p);
    workspaceFiles.push(...scanMFiles(p));
  }
  if (existsSync(mipCoreSearchPath)) {
    searchPaths.push(mipCoreSearchPath);
    workspaceFiles.push(...scanMFiles(mipCoreSearchPath));
  }

  await executeWithOptions(
    code,
    mainFileName,
    workspaceFiles,
    opts,
    searchPaths
  );
}

async function cmdParse(args: string[]) {
  let dumpAst: string | undefined;
  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--dump-ast") {
      i++;
      if (i >= args.length) {
        console.error("Error: --dump-ast requires an output filename");
        process.exit(1);
      }
      dumpAst = resolve(process.cwd(), args[i]);
    } else if (args[i].startsWith("-")) {
      console.error(`Unknown option: ${args[i]}`);
      process.exit(1);
    } else {
      positional.push(args[i]);
    }
  }

  if (positional.length === 0) {
    console.error("Error: numbl parse requires a file argument");
    process.exit(1);
  }
  if (positional.length > 1) {
    console.error("Error: numbl parse accepts only one file argument");
    process.exit(1);
  }

  const filepath = resolve(process.cwd(), positional[0]);
  const code = readFileSync(filepath, "utf-8");

  try {
    const ast = parseMFile(code, filepath);
    const json = JSON.stringify(ast, null, 2);
    if (dumpAst) {
      writeFileSync(dumpAst, json + "\n");
    } else {
      process.stdout.write(json + "\n");
    }
    process.exit(0);
  } catch (error) {
    const diags = diagnoseErrors(error, code, filepath, []);
    const getSource = (file: string) =>
      getSourceForFile(file, filepath, code, []);
    console.error(formatDiagnostics(diags, getSource));
    process.exit(1);
  }
}

async function cmdEval(args: string[]) {
  const opts = parseOptions(args);

  if (opts.positional.length === 0) {
    console.error("Error: numbl eval requires a code argument");
    process.exit(1);
  }
  if (opts.positional.length > 1) {
    console.error("Error: numbl eval accepts only one code argument");
    process.exit(1);
  }

  await ensureMipCorePackage();

  const searchPaths: string[] = [];
  const workspaceFiles: WorkspaceFile[] = [];
  for (const p of opts.extraPaths) {
    searchPaths.push(p);
    workspaceFiles.push(...scanMFiles(p));
  }
  if (existsSync(mipCoreSearchPath)) {
    searchPaths.push(mipCoreSearchPath);
    workspaceFiles.push(...scanMFiles(mipCoreSearchPath));
  }

  await executeWithOptions(
    opts.positional[0],
    "eval.m",
    workspaceFiles,
    opts,
    searchPaths
  );
}

async function executeWithOptions(
  code: string,
  mainFileName: string,
  workspaceFiles: WorkspaceFile[],
  opts: ParsedOptions,
  searchPaths?: string[]
) {
  loadNativeAddon(opts.fastMath);
  const profiling = !!opts.profileOutput;
  const totalStart = performance.now();

  // Helper to write profile data after execution completes
  const writeProfileIfNeeded = (result: {
    profileData?: import("./numbl-core/executeCode.js").ProfileData;
  }) => {
    if (!opts.profileOutput || !result.profileData) return;
    const pd = result.profileData;
    const totalTimeMs = performance.now() - totalStart;
    const builtins = Object.entries(pd.builtins)
      .map(([name, b]) => ({
        name,
        fallback: b.fallback,
        interp: b.interp,
        jit: b.jit,
      }))
      .sort((a, b) => {
        const aTotal =
          a.fallback.totalTimeMs + a.interp.totalTimeMs + a.jit.totalTimeMs;
        const bTotal =
          b.fallback.totalTimeMs + b.interp.totalTimeMs + b.jit.totalTimeMs;
        return bTotal - aTotal;
      });
    const dispatches = Object.entries(pd.dispatches)
      .map(([name, b]) => ({
        name,
        totalTimeMs: b.totalTimeMs,
        callCount: b.callCount,
      }))
      .sort((a, b) => b.totalTimeMs - a.totalTimeMs);
    const profile = {
      totalTimeMs,
      executionTimeMs: pd.executionTimeMs,
      jitCompileTimeMs: pd.jitCompileTimeMs,
      builtins,
      dispatches,
      hotLoops: pd.hotLoops ?? [],
    };
    writeFileSync(opts.profileOutput, JSON.stringify(profile, null, 2) + "\n");
    console.error(`Profile written to ${opts.profileOutput}`);
  };

  // If --dump-ast is used alone (without running), just dump and exit
  if (
    opts.dumpAst &&
    !opts.dumpJs &&
    !opts.dumpC &&
    !opts.stream &&
    !opts.verbose
  ) {
    const ast = parseMFile(code, mainFileName);
    process.stdout.write(JSON.stringify(ast, null, 2) + "\n");
    process.exit(0);
  }

  const fileIO = new NodeFileIOAdapter();
  const system = new NodeSystemAdapter();

  /** Synchronous line reader for input() builtin. */
  const onInput = (prompt: string): string => {
    process.stdout.write(prompt);
    const buf = Buffer.alloc(1);
    let line = "";
    while (true) {
      const bytesRead = readSync(0, buf, 0, 1, null);
      if (bytesRead === 0) break; // EOF
      const ch = buf.toString("utf8");
      if (ch === "\n") break;
      if (ch === "\r") continue;
      line += ch;
    }
    return line;
  };

  // Set up --dump-js / --dump-c. Each JIT section is streamed to disk the
  // moment it compiles (in onJitCompile, which fires before the compiled
  // code runs) rather than written in one shot at the end. This keeps the
  // dump intact even when a hard error in compiled C terminates the
  // process via exit(1) from inside the FFI call — that path never unwinds
  // back to JS, so an end-of-run write would be lost. Write the header now
  // so a crash still leaves a self-describing file.
  const dumpHeader =
    "// " +
    "=".repeat(60) +
    "\n// MAIN SCRIPT: " +
    mainFileName +
    "\n// " +
    "=".repeat(60) +
    "\n\n";
  const dumpCounts = { js: 0, c: 0 };
  // Start each file fresh with the header. `/dev/*` targets (e.g.
  // `/dev/stdout`) can't be truncated, so append there instead.
  const initDump = (file: string) => {
    if (file.startsWith("/dev/")) appendFileSync(file, dumpHeader);
    else writeFileSync(file, dumpHeader);
  };
  if (opts.dumpJs) initDump(opts.dumpJs);
  if (opts.dumpC) initDump(opts.dumpC);
  const onJitCompile:
    | ((description: string, code: string, lang: "js" | "c") => void)
    | undefined =
    opts.dumpJs || opts.dumpC
      ? (description, code, lang) => {
          const file = lang === "c" ? opts.dumpC : opts.dumpJs;
          if (!file) return;
          const banner = `// ${"=".repeat(60)}\n// JIT: ${description}\n// ${"=".repeat(60)}\n\n${code}\n`;
          const prefix =
            dumpCounts[lang] === 0
              ? "// Interpreter mode — JIT compiled sections:\n\n"
              : "\n";
          appendFileSync(file, prefix + banner);
          dumpCounts[lang]++;
        }
      : undefined;
  // Append the "nothing generated" placeholder for any enabled dump file
  // that received no sections. Safe to call more than once (guarded on
  // count), so it runs on both the success and error paths.
  const finalizeDumps = () => {
    if (opts.dumpJs && dumpCounts.js === 0) {
      appendFileSync(opts.dumpJs, "// No JS generated\n");
      dumpCounts.js = -1;
    }
    if (opts.dumpC && dumpCounts.c === 0) {
      appendFileSync(opts.dumpC, "// No C generated\n");
      dumpCounts.c = -1;
    }
  };

  // Surface a runtime JIT bail (e.g. an indexed-store array growth the
  // JIT can't model) as a stderr warning. The scope still runs
  // correctly — the interpreter takes over — so this is informational,
  // not an error; it goes to stderr to leave stdout (program output)
  // untouched.
  const onJitBail = (message: string): void => {
    process.stderr.write(`warning: JIT bailed to interpreter — ${message}\n`);
  };

  try {
    if (opts.stream) {
      // Force stdout to blocking mode so large writes (e.g. drawnow events
      // carrying megabytes of plot data) are fully flushed before
      // process.exit(0) below. With the default non-blocking pipe behavior,
      // exit can drop the tail of a write, which the parent then sees as
      // an unparseable line and surfaces as raw output text.
      const stdoutHandle = (
        process.stdout as unknown as {
          _handle?: { setBlocking?: (b: boolean) => void };
        }
      )._handle;
      stdoutHandle?.setBlocking?.(true);
      const streamLine = (obj: Record<string, unknown>) => {
        process.stdout.write(JSON.stringify(obj) + "\n");
      };
      try {
        const result = executeCode(
          code,
          {
            displayResults: true,
            profile: profiling,
            onOutput: (text: string) => {
              streamLine({ type: "output", text });
            },
            onDrawnow: plotInstructions => {
              streamLine({ type: "drawnow", plotInstructions });
            },
            onJitCompile,
            onJitBail,

            fileIO,
            system,
            onInput,

            optimization: opts.optimization,
          },
          workspaceFiles,
          mainFileName,
          searchPaths,
          nativeBridge
        );
        if (result.plotInstructions.length > 0) {
          streamLine({
            type: "drawnow",
            plotInstructions: result.plotInstructions,
          });
        }
        writeProfileIfNeeded(result);
        finalizeDumps();
        streamLine({
          type: "done",
          generatedJS: result.generatedJS || undefined,
          generatedC: result.generatedC || undefined,
        });
      } catch (error) {
        const diags = diagnoseErrors(error, code, mainFileName, workspaceFiles);
        for (const diag of diags) {
          streamLine({
            type: "error",
            message: diag.message,
            errorType: diag.errorType,
            file: diag.file,
            line: diag.line,
            snippet: diag.snippet,
          });
        }
      }
      process.exit(0);
    } else if (opts.verbose) {
      const log = (msg: string) => console.error(`[verbose] ${msg}`);
      const { onDrawnow, flushAndWait } = createPlotHandler(
        !opts.plot || opts.stream
      );
      const result = executeCode(
        code,
        {
          displayResults: true,
          profile: profiling,
          onOutput: (text: string) => {
            process.stdout.write(text);
          },
          onDrawnow,
          log,
          onJitCompile,
          onJitBail,

          fileIO,
          system,
          onInput,
          optimization: opts.optimization,
        },
        workspaceFiles,
        mainFileName,
        searchPaths,
        nativeBridge
      );
      writeProfileIfNeeded(result);
      finalizeDumps();
      await flushAndWait(result.plotInstructions);
    } else {
      const asyncPlotOpts: PlotServerOptions | undefined =
        opts.plotPort !== undefined
          ? { port: opts.plotPort, host: "0.0.0.0" }
          : undefined;
      const { onDrawnow, flushAndWait } = createPlotHandler(
        !opts.plot,
        asyncPlotOpts
      );
      const result = executeCode(
        code,
        {
          displayResults: true,
          profile: profiling,
          onOutput: (text: string) => {
            process.stdout.write(text);
          },
          onDrawnow,
          onJitCompile,
          onJitBail,

          fileIO,
          system,
          onInput,
          optimization: opts.optimization,
        },
        workspaceFiles,
        mainFileName,
        searchPaths,
        nativeBridge
      );
      writeProfileIfNeeded(result);
      finalizeDumps();
      await flushAndWait(result.plotInstructions);
    }

    process.exit(0);
  } catch (error) {
    const diags = diagnoseErrors(error, code, mainFileName, workspaceFiles);
    const getSource = (file: string) =>
      getSourceForFile(file, mainFileName, code, workspaceFiles);
    console.error(formatDiagnostics(diags, getSource));

    // The dump files were streamed as each section compiled (see
    // onJitCompile above), so anything generated before the error is
    // already on disk. Just cap off any file that got no sections.
    finalizeDumps();

    process.exit(1);
  }
}

async function cmdBuildAddon(args: string[]) {
  // Default OFF (deterministic, cross-mode-consistent). `--fast-math`
  // opts in; `--no-fast-math` is accepted for back-compat (now default).
  const fastMath = args.includes("--fast-math");
  for (const a of args) {
    if (a !== "--no-fast-math" && a !== "--fast-math") {
      console.error(`Unknown option: ${a}`);
      process.exit(1);
    }
  }
  const bindingGyp = join(packageDir, "binding.gyp");
  if (!existsSync(bindingGyp)) {
    console.error(
      "Error: binding.gyp not found in package directory: " + packageDir
    );
    process.exit(1);
  }
  console.log("Building native LAPACK addon...");
  console.log("Package directory: " + packageDir);
  console.log("Prerequisites: C++ compiler, libopenblas-dev (or equivalent)");
  console.log(
    `-ffast-math: ${fastMath ? "ENABLED (--fast-math)" : "disabled (default)"}`
  );
  console.log("");
  try {
    // node-gyp picks up NUMBL_FAST_MATH at gyp-time via the
    // `fast_math%` variable in binding.gyp.
    const env = {
      ...process.env,
      NUMBL_FAST_MATH: fastMath ? "true" : "false",
    };
    execSync("npx node-gyp rebuild", {
      cwd: packageDir,
      stdio: "inherit",
      env,
    });
    // Copy the freshly built addon to a variant-specific path so both
    // fast-math and no-fast-math builds can coexist. The unversioned
    // numbl_addon.node gets overwritten on each rebuild; the loader
    // always reads the variant-specific path.
    if (!existsSync(addonBuildPath)) {
      throw new Error(`expected built addon at ${addonBuildPath}`);
    }
    const variantPath = variantAddonPath(fastMath);
    copyFileSync(addonBuildPath, variantPath);
    console.log("");
    console.log(`Native LAPACK addon built: ${variantPath}`);
    console.log("Restart numbl to use it.");
  } catch {
    console.error("");
    console.error("Failed to build native LAPACK addon.");
    console.error(
      "Ensure you have a C++ compiler, libopenblas-dev, and libfftw3-dev installed."
    );
    process.exit(1);
  }
}

function cmdInfo() {
  const fastMathPath = variantAddonPath(true);
  const noFastMathPath = variantAddonPath(false);
  const fastMathPresent = existsSync(fastMathPath);
  const noFastMathPresent = existsSync(noFastMathPath);
  process.stdout.write(
    JSON.stringify({
      version: NUMBL_VERSION,
      nativeAddon: fastMathPresent || noFastMathPresent,
      nativeAddonVariants: {
        fastMath: fastMathPresent ? fastMathPath : null,
        noFastMath: noFastMathPresent ? noFastMathPath : null,
      },
      packageDir,
    }) + "\n"
  );
}

function cmdListBuiltins(flags: string[]) {
  const noHelp = flags.includes("--no-help");
  const names = getAllIBuiltinNames().sort();
  for (const name of names) {
    if (noHelp && getIBuiltinHelp(name)) continue;
    console.log(name);
  }
}

async function cmdRepl(args: string[]) {
  const opts = parseOptions(args);

  loadNativeAddon(opts.fastMath);
  await ensureMipCorePackage();

  const replSearchPaths = [...opts.extraPaths];
  const replFiles = opts.extraPaths.flatMap(d => scanMFiles(d));
  if (existsSync(mipCoreSearchPath)) {
    replSearchPaths.push(mipCoreSearchPath);
    replFiles.push(...scanMFiles(mipCoreSearchPath));
  }
  const replPlotOpts: PlotServerOptions | undefined =
    opts.plotPort !== undefined
      ? { port: opts.plotPort, host: "0.0.0.0" }
      : undefined;
  const { onDrawnow: replDrawnow } = createPlotHandler(
    !opts.plot,
    replPlotOpts
  );
  await runRepl(
    replFiles,
    replDrawnow,
    replSearchPaths,
    nativeBridge,
    opts.optimization
  );
}

// ── Show Profile ─────────────────────────────────────────────────────────────

function cmdShowProfile(args: string[]) {
  if (args.length === 0) {
    console.error("Error: show-profile requires a profile JSON file argument");
    process.exit(1);
  }
  const filepath = resolve(process.cwd(), args[0]);
  if (!existsSync(filepath)) {
    console.error(`Error: profile file not found: ${filepath}`);
    process.exit(1);
  }
  const data = JSON.parse(readFileSync(filepath, "utf-8"));

  const fmt = (n: number, decimals = 1) => n.toFixed(decimals);
  const pad = (s: string, w: number) => s.padStart(w);

  const execTime = data.executionTimeMs || 1;
  const jitTime = data.jitCompileTimeMs ?? 0;

  // Compute disjoint breakdown of execution time
  const dispatches: { name: string; totalTimeMs: number; callCount: number }[] =
    data.dispatches ?? [];
  const builtins: {
    name: string;
    fallback: { totalTimeMs: number; callCount: number };
    interp: { totalTimeMs: number; callCount: number };
    jit: { totalTimeMs: number; callCount: number };
  }[] = data.builtins ?? [];
  const sumDispatches = dispatches.reduce((s, d) => s + d.totalTimeMs, 0);
  const sumBuiltins = builtins.reduce(
    (s, b) =>
      s + b.fallback.totalTimeMs + b.interp.totalTimeMs + b.jit.totalTimeMs,
    0
  );
  const otherTime =
    data.executionTimeMs - jitTime - sumDispatches - sumBuiltins;

  console.log(`Total time:      ${fmt(data.totalTimeMs)} ms`);
  console.log(`Execution time:  ${fmt(data.executionTimeMs)} ms`);
  console.log(
    `  JIT compile:   ${fmt(jitTime)} ms (${fmt((jitTime / execTime) * 100)}%)`
  );
  console.log(
    `  Dispatches:    ${fmt(sumDispatches)} ms (${fmt((sumDispatches / execTime) * 100)}%)`
  );
  console.log(
    `  Builtins:      ${fmt(sumBuiltins)} ms (${fmt((sumBuiltins / execTime) * 100)}%)`
  );
  console.log(
    `  Other:         ${fmt(otherTime)} ms (${fmt((otherTime / execTime) * 100)}%)`
  );
  console.log("\nAll timings are self-time (exclusive) and mutually disjoint.");

  // Dispatched user functions
  if (dispatches.length > 0) {
    const nameW = Math.max(20, ...dispatches.map(b => b.name.length)) + 2;
    const header =
      "Function".padEnd(nameW) +
      pad("Calls", 10) +
      pad("Self (ms)", 13) +
      pad("Avg (ms)", 11) +
      pad("% Exec", 9);
    console.log("\nDispatched functions (user-defined / workspace):");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const b of dispatches) {
      const avg = b.callCount > 0 ? b.totalTimeMs / b.callCount : 0;
      const pct = (b.totalTimeMs / execTime) * 100;
      const line =
        b.name.padEnd(nameW) +
        pad(String(b.callCount), 10) +
        pad(fmt(b.totalTimeMs, 2), 13) +
        pad(fmt(avg, 3), 11) +
        pad(fmt(pct, 1) + "%", 9);
      console.log(line);
    }
  }

  // Builtin functions
  if (builtins.length > 0) {
    const nameW = Math.max(20, ...builtins.map(b => b.name.length)) + 2;
    const header =
      "Builtin".padEnd(nameW) +
      pad("Interp", 10) +
      pad("Fallback", 10) +
      pad("JIT", 10) +
      pad("Self (ms)", 13) +
      pad("% Exec", 9);
    console.log("\nBuiltin functions (call counts by source):");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const b of builtins) {
      const totalTime =
        b.fallback.totalTimeMs + b.interp.totalTimeMs + b.jit.totalTimeMs;
      const pct = (totalTime / execTime) * 100;
      const line =
        b.name.padEnd(nameW) +
        pad(String(b.interp.callCount), 10) +
        pad(String(b.fallback.callCount), 10) +
        pad(String(b.jit.callCount), 10) +
        pad(fmt(totalTime, 2), 13) +
        pad(fmt(pct, 1) + "%", 9);
      console.log(line);
    }
  } else {
    console.log("\nNo builtin functions were called.");
  }

  // Hot interpreted loops (>1000 iterations)
  const hotLoops: {
    file: string;
    line: number;
    kind: string;
    iterations: number;
    callCount: number;
    totalTimeMs: number;
  }[] = data.hotLoops ?? [];
  if (hotLoops.length > 0) {
    hotLoops.sort((a, b) => b.totalTimeMs - a.totalTimeMs);
    const locW =
      Math.max(20, ...hotLoops.map(l => `${l.file}:${l.line}`.length)) + 2;
    const header =
      "Location".padEnd(locW) +
      pad("Kind", 8) +
      pad("Calls", 8) +
      pad("Max Iters", 12) +
      pad("Total (ms)", 13);
    console.log("\nInterpreted loops (>1000 iterations):");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const l of hotLoops) {
      const loc = `${l.file}:${l.line}`;
      const line =
        loc.padEnd(locW) +
        pad(l.kind, 8) +
        pad(String(l.callCount ?? 1), 8) +
        pad(String(l.iterations), 12) +
        pad(fmt(l.totalTimeMs ?? 0, 2), 13);
      console.log(line);
    }
  }
}

// ── Serve ───────────────────────────────────────────────────────────────────

function cmdServe(args: string[]): void {
  let port = 3001;
  let passkey: string | undefined;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port" && i + 1 < args.length) {
      port = parseInt(args[++i]);
      if (isNaN(port)) {
        console.error("Invalid port number");
        process.exit(1);
      }
    } else if (args[i] === "--passkey" && i + 1 < args.length) {
      passkey = args[++i];
    }
  }
  if (!passkey) {
    console.error("Error: --passkey <key> is required");
    console.error("The passkey is generated by the browser IDE.");
    process.exit(1);
  }
  // Always use the bundled CLI so the spawned child process works with
  // plain `node` (the source .ts files require tsx).
  const bundledCli = join(packageDir, "dist-cli", "cli.js");
  if (!existsSync(bundledCli)) {
    console.error("dist-cli/cli.js not found. Run `npm run build:cli` first.");
    process.exit(1);
  }
  import("../server/execution-service.js").then(({ startServer }) => {
    startServer({ port, numblCliPath: bundledCli, passkey: passkey! });
  });
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);

  // --version / -V
  if (args.includes("--version") || args.includes("-V")) {
    console.log(NUMBL_VERSION);
    process.exit(0);
  }

  // --help / -h at any position
  if (args.includes("--help") || args.includes("-h")) {
    printHelp();
    process.exit(0);
  }

  // No args or starts with a flag (not a command) → REPL
  if (args.length === 0 || args[0].startsWith("--")) {
    await cmdRepl(args);
    return;
  }

  const command = args[0];
  const rest = args.slice(1);

  switch (command) {
    case "run":
      await cmdRun(rest);
      break;
    case "eval":
      await cmdEval(rest);
      break;
    case "parse":
      await cmdParse(rest);
      break;
    case "run-tests": {
      const testOpts = parseOptions(rest);
      const dir =
        testOpts.positional.length > 0
          ? testOpts.positional[0]
          : join(packageDir, "numbl_test_scripts");
      await runTests(dir, testOpts.optimization, testOpts.fastMath);
      break;
    }
    case "build-addon":
      await cmdBuildAddon(rest);
      break;
    case "info":
      cmdInfo();
      break;
    case "list-builtins":
      cmdListBuiltins(rest);
      break;
    case "show-profile":
      cmdShowProfile(rest);
      break;
    case "serve":
      cmdServe(rest);
      break;
    default:
      console.error(`Unknown command: ${command}`);
      console.error('Run "numbl --help" for usage information.');
      process.exit(1);
  }
}

main();

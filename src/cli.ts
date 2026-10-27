#!/usr/bin/env node

import {
  readFileSync,
  readSync,
  existsSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  rmSync,
  copyFileSync,
} from "fs";
import { delimiter, dirname, join, resolve } from "path";
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
import {
  setLapackBridge,
  setBridgeLog,
} from "./numbl-core/native/lapack-bridge.js";
import {
  diagnoseErrors,
  formatDiagnostics,
  getSourceForFile,
} from "./numbl-core/diagnostics";
import { NUMBL_VERSION } from "./numbl-core/version.js";
import { runRepl } from "./cli-repl.js";
import { NodeFileIOAdapter } from "./cli-fileio.js";
import { NodeSystemAdapter } from "./cli-system.js";
import {
  parseCommandArgs,
  parseNumberOption,
  parseRunOptions,
  type RunOptions,
} from "./cli-args.js";
import { runTests } from "./cli-run-tests.js";
import { cmdBuildSite } from "./cli-build-site.js";
import { cmdShowProfile, writeProfile } from "./cli-profile.js";

import { executeCode } from "./numbl-core/executeCode.js";
import { parseMFile } from "./numbl-core/parser/index.js";
import { WorkspaceFile, NativeBridge } from "./numbl-core/workspace/types.js";
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

function loadNativeAddon(fastMath: boolean, noNative = false): void {
  if (noNative || process.env.NUMBL_NO_NATIVE) return;
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
  } catch (e) {
    console.error(
      `Warning: failed to load native addon at ${variantPath}: ${(e as Error).message}. Using JS fallbacks.`
    );
  }
}

// ── Try to load the qhull WASM Delaunay backend ──────────────────────────────

import { loadQhullNodeBackend } from "./numbl-core/native/qhull-node.js";

let qhullLoaded = false;
async function loadQhullBackend(): Promise<void> {
  if (qhullLoaded || process.env.NUMBL_NO_QHULL) return;
  qhullLoaded = true;
  try {
    await loadQhullNodeBackend();
  } catch (e) {
    console.error(
      `Warning: failed to load qhull WASM backend: ${(e as Error).message}. ` +
        `delaunay/delaunayn will be unavailable.`
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

// ── Help ─────────────────────────────────────────────────────────────────────

function printHelp() {
  console.log(`Usage: numbl <command> [options]

Commands:
  run <file.m>       Run a .m file
  eval "<code>"      Evaluate inline code
  parse <file.m>     Tokenize, lex, and parse a .m file (no execution)
  run-tests [dir]    Run .m test scripts (default: numbl_test_scripts/)
  build-addon        Build native LAPACK addon (pass --fast-math to enable -ffast-math)
  build-site <dir>   Build a static, browser-runnable site for a numbl project
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

Options (for build-site):
  <dir>              Project directory to bundle (default: current directory)
  --out <dir>        Output directory for the static site (default: dist)
  --base <path>      Deploy base path, e.g. /my-repo/ (for GitHub Pages project
                       sites; omit for root/custom-domain deploys)
  --title <text>     Site title (default: from numbl-project.json or dir name)
  --entry <file>     Default file to open, e.g. main.m (default: README.md or
                       the first script)
  --repo-url <url>   Source repository URL, shown as a link in the deployed
                       site (default: from numbl-project.json)
  --max-initial-output-panel-height <px>
                     Cap the initial height (px) of the output panel so the
                       figure panel below it starts larger (desktop layout)

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
  --no-native        Do not load the native LAPACK addon; use the pure-JS
                     fallbacks (same effect as NUMBL_NO_NATIVE=1)

Environment variables:
  NUMBL_PATH              Extra workspace directories (separated by ${delimiter})
  NUMBL_NO_NATIVE         If set, do not load the native LAPACK addon
  NUMBL_JIT_LOG           If set, print every JIT decline (executor, source
                          position, reason) to stderr`);
}

// ── Command handlers ─────────────────────────────────────────────────────────

/** Extra --path / NUMBL_PATH directories plus the mip core package (installed
 *  or refreshed first), with the workspace files found in them. */
async function prepareSearchPaths(
  extraPaths: string[]
): Promise<{ searchPaths: string[]; workspaceFiles: WorkspaceFile[] }> {
  await ensureMipCorePackage();
  const searchPaths = [...extraPaths];
  if (existsSync(mipCoreSearchPath)) searchPaths.push(mipCoreSearchPath);
  return {
    searchPaths,
    workspaceFiles: searchPaths.flatMap(d => scanMFiles(d)),
  };
}

/** The single positional argument of `numbl <command>`. */
function singlePositional(
  positional: string[],
  command: string,
  what: string
): string {
  if (positional.length === 0) {
    console.error(`Error: numbl ${command} requires a ${what} argument`);
    process.exit(1);
  }
  if (positional.length > 1) {
    console.error(`Error: numbl ${command} accepts only one ${what} argument`);
    process.exit(1);
  }
  return positional[0];
}

async function cmdRun(args: string[]) {
  const opts = parseRunOptions(args);
  const filepath = resolve(
    process.cwd(),
    singlePositional(opts.positional, "run", "file")
  );
  const code = readFileSync(filepath, "utf-8");

  // Auto-cd into the script's directory before executing (matches MATLAB
  // "F5 in editor" behavior). The script directory then becomes the
  // first-priority implicit search path via the cwd-as-search-path feature.
  process.chdir(dirname(filepath));

  const { searchPaths, workspaceFiles } = await prepareSearchPaths(
    opts.extraPaths
  );
  await executeWithOptions(code, filepath, workspaceFiles, opts, searchPaths);
}

async function cmdParse(args: string[]) {
  const { values, positionals } = parseCommandArgs(args, {
    "dump-ast": { type: "string" },
  });
  const dumpAst =
    values["dump-ast"] === undefined
      ? undefined
      : resolve(process.cwd(), values["dump-ast"]);
  const filepath = resolve(
    process.cwd(),
    singlePositional(positionals, "parse", "file")
  );
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
  const opts = parseRunOptions(args);
  const code = singlePositional(opts.positional, "eval", "code");
  const { searchPaths, workspaceFiles } = await prepareSearchPaths(
    opts.extraPaths
  );
  await executeWithOptions(code, "eval.m", workspaceFiles, opts, searchPaths);
}

/** NDJSON writer on stdout for --stream. */
function ndjsonStdout(): (obj: Record<string, unknown>) => void {
  // Force stdout to blocking mode so large writes (e.g. drawnow events
  // carrying megabytes of plot data) are fully flushed before
  // process.exit(0). With the default non-blocking pipe behavior, exit can
  // drop the tail of a write, which the parent then sees as an unparseable
  // line and surfaces as raw output text.
  const stdoutHandle = (
    process.stdout as unknown as {
      _handle?: { setBlocking?: (b: boolean) => void };
    }
  )._handle;
  stdoutHandle?.setBlocking?.(true);
  return obj => {
    process.stdout.write(JSON.stringify(obj) + "\n");
  };
}

/** Plot handler for --plot / --plot-port (disabled without them). */
function plotHandlerFor(opts: RunOptions) {
  const serverOpts: PlotServerOptions | undefined =
    opts.plotPort !== undefined
      ? { port: opts.plotPort, host: "0.0.0.0" }
      : undefined;
  return createPlotHandler(!opts.plot, serverOpts);
}

async function executeWithOptions(
  code: string,
  mainFileName: string,
  workspaceFiles: WorkspaceFile[],
  opts: RunOptions,
  searchPaths?: string[]
) {
  loadNativeAddon(opts.fastMath, opts.noNative);
  await loadQhullBackend();
  const totalStart = performance.now();

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

  // --stream emits NDJSON on stdout; otherwise output goes to stdout and
  // figures to the plot handler (a no-op unless --plot).
  const stream = opts.stream ? ndjsonStdout() : undefined;
  const plot = stream ? undefined : plotHandlerFor(opts);
  const log =
    opts.verbose && !stream
      ? (msg: string) => console.error(`[verbose] ${msg}`)
      : undefined;
  if (log) setBridgeLog(log);

  try {
    const result = executeCode(
      code,
      {
        displayResults: true,
        profile: !!opts.profileOutput,
        onOutput: stream
          ? text => stream({ type: "output", text })
          : text => process.stdout.write(text),
        onDrawnow: stream
          ? plotInstructions => stream({ type: "drawnow", plotInstructions })
          : plot?.onDrawnow,
        onHtmlSourceEvent: plot?.sendUihtmlEvent,
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
    if (opts.profileOutput && result.profileData) {
      writeProfile(
        opts.profileOutput,
        result.profileData,
        performance.now() - totalStart
      );
    }
    finalizeDumps();
    if (stream) {
      if (result.plotInstructions.length > 0) {
        stream({ type: "drawnow", plotInstructions: result.plotInstructions });
      }
      stream({
        type: "done",
        generatedJS: result.generatedJS || undefined,
        generatedC: result.generatedC || undefined,
      });
    } else {
      plot?.setUihtmlSession(result.uihtmlSession ?? null);
      await plot?.flushAndWait(result.plotInstructions);
    }
    process.exit(0);
  } catch (error) {
    const diags = diagnoseErrors(error, code, mainFileName, workspaceFiles);
    // The dump files were streamed as each section compiled (see
    // onJitCompile above), so anything generated before the error is
    // already on disk. Just cap off any file that got no sections.
    finalizeDumps();
    if (stream) {
      for (const diag of diags) {
        stream({
          type: "error",
          message: diag.message,
          errorType: diag.errorType,
          file: diag.file,
          line: diag.line,
          snippet: diag.snippet,
        });
      }
      process.exit(0);
    }
    const getSource = (file: string) =>
      getSourceForFile(file, mainFileName, code, workspaceFiles);
    console.error(formatDiagnostics(diags, getSource));
    process.exit(1);
  }
}

async function cmdBuildAddon(args: string[]) {
  // Default OFF (deterministic, cross-mode-consistent). `--fast-math`
  // opts in; `--no-fast-math` is accepted for back-compat (now default).
  const { values, positionals } = parseCommandArgs(args, {
    "fast-math": { type: "boolean" },
    "no-fast-math": { type: "boolean" },
  });
  if (positionals.length > 0) {
    console.error(`Unknown option: ${positionals[0]}`);
    process.exit(1);
  }
  const fastMath = values["fast-math"] ?? false;
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

function cmdListBuiltins(args: string[]) {
  const { values } = parseCommandArgs(args, {
    "no-help": { type: "boolean" },
  });
  const noHelp = values["no-help"] ?? false;
  const names = getAllIBuiltinNames().sort();
  for (const name of names) {
    if (noHelp && getIBuiltinHelp(name)) continue;
    console.log(name);
  }
}

async function cmdRepl(args: string[]) {
  const opts = parseRunOptions(args);
  loadNativeAddon(opts.fastMath, opts.noNative);
  const { searchPaths, workspaceFiles } = await prepareSearchPaths(
    opts.extraPaths
  );
  await runRepl(
    workspaceFiles,
    plotHandlerFor(opts).onDrawnow,
    searchPaths,
    nativeBridge,
    opts.optimization
  );
}

// ── Serve ───────────────────────────────────────────────────────────────────

function cmdServe(args: string[]): void {
  const { values } = parseCommandArgs(args, {
    port: { type: "string" },
    passkey: { type: "string" },
  });
  const port =
    values.port === undefined
      ? 3001
      : parseNumberOption(values.port, parseInt, "Invalid port number");
  const passkey = values.passkey;
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
    startServer({ port, numblCliPath: bundledCli, passkey });
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
      const opts = parseRunOptions(rest);
      loadNativeAddon(opts.fastMath, opts.noNative);
      await loadQhullBackend();
      runTests(
        opts.positional[0] ?? join(packageDir, "numbl_test_scripts"),
        opts.optimization,
        nativeBridge
      );
      break;
    }
    case "build-addon":
      await cmdBuildAddon(rest);
      break;
    case "build-site":
      await cmdBuildSite(rest, packageDir);
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

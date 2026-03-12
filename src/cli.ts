#!/usr/bin/env node

import {
  readFileSync,
  readdirSync,
  statSync,
  existsSync,
  writeFileSync,
  appendFileSync,
} from "fs";
import { delimiter, dirname, join, relative, resolve } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import { execSync } from "child_process";
import {
  startPlotServer,
  type PlotServer,
  type PlotServerOptions,
} from "./cli-plot-server.js";
import { cmdMip } from "./cli-mip.js";
import { extractMipDirectives, processMipLoad } from "./mip-directives.js";
import { getAllBuiltinNames } from "./numbl-core/builtins/index.js";
import { setLapackBridge } from "./numbl-core/native/lapack-bridge.js";
import { setLapackBridge as setLapackBridgeNew } from "./numbl-core/native/lapack-bridge.js";
import { diagnoseErrors, formatDiagnostics } from "./numbl-core/diagnostics";
import { runRepl } from "./cli-repl.js";
import { NodeFileIOAdapter } from "./cli-fileio.js";

import { executeCode, generateCode } from "./numbl-core/executeCode.js";
import { WorkspaceFile } from "./numbl-core/workspace/types.js";
import { PlotInstruction } from "./numbl-core/executor/types.js";

// ── Package directory & native addon paths ───────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const packageDir = join(__dirname, "..");
const addonPath = join(packageDir, "build", "Release", "lapack_addon.node");

// ── Try to load the native LAPACK addon ──────────────────────────────────────

let nativeAddonLoaded = false;
try {
  const req = createRequire(import.meta.url);
  const addon = req(addonPath);
  setLapackBridge(addon);
  setLapackBridgeNew(addon);
  nativeAddonLoaded = true;
} catch {
  // Native addon not available — JS fallbacks will be used
}

/**
 * Recursively scan a directory for .m files.
 * File names are stored as absolute paths.
 */
export function scanMFiles(
  dirPath: string,
  excludeFile?: string
): WorkspaceFile[] {
  const files: WorkspaceFile[] = [];

  try {
    const entries = readdirSync(dirPath);

    for (const entry of entries) {
      const fullPath = join(dirPath, entry);

      if (excludeFile && fullPath === excludeFile) {
        continue;
      }

      const stat = statSync(fullPath);

      if (stat.isDirectory()) {
        if (
          entry.startsWith("@") ||
          entry.startsWith("+") ||
          entry === "private"
        ) {
          files.push(...scanMFiles(fullPath, excludeFile));
        }
      } else if (
        stat.isFile() &&
        (entry.endsWith(".m") || entry.endsWith(".js"))
      ) {
        const source = readFileSync(fullPath, "utf-8");
        files.push({
          name: fullPath,
          source,
        });
      }
    }
  } catch {
    // Silently skip directories we can't read
  }

  return files;
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
        if (!entry.startsWith("@") && !entry.startsWith("+")) {
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

async function runTests(dir: string) {
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
        { displayResults: true },
        workspaceFiles,
        mainFileName,
        searchPaths
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
      console.log(`FAIL  ${rel}`);
      console.log(
        `      ${formatDiagnostics(diags).replace(/\n/g, "\n      ")}`
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
  run-tests [dir]    Run .m test scripts (default: numbl_test_scripts/)
  build-addon        Build native LAPACK addon
  info               Print machine-readable info (JSON)
  list-builtins      List available built-in functions
  mip <subcommand>   Package manager (install, uninstall, list, avail, info)
  (no command)       Start interactive REPL

Options (for REPL):
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)

Options (for run and eval):
  --dump-js <file>   Write all generated JavaScript (main + JIT) to file
  --dump-ast         Print AST as JSON
  --verbose          Detailed logging to stderr
  --stream           NDJSON output mode
  --path <dir>       Add extra workspace directory
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)
  --add-script-path  Add the script's directory to the workspace (run only)
  --no-line-tracking  Omit $rt.$file/$rt.$line from generated JS

Environment variables:
  NUMBL_PATH         Extra workspace directories (separated by ${delimiter})`);
}

// ── Option parsing helpers ───────────────────────────────────────────────────

interface ParsedOptions {
  dumpJs: string | undefined;
  dumpAst: boolean;
  verbose: boolean;
  stream: boolean;
  plot: boolean;
  plotPort: number | undefined;
  addScriptPath: boolean;
  extraPaths: string[];
  positional: string[];
  profileOutput: string | undefined;
  noLineTracking: boolean;
}

function parseOptions(args: string[]): ParsedOptions {
  const opts: ParsedOptions = {
    dumpJs: undefined,
    dumpAst: false,
    verbose: false,
    stream: false,
    plot: false,
    plotPort: undefined,
    addScriptPath: false,
    extraPaths: [],
    positional: [],
    profileOutput: undefined,
    noLineTracking: false,
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
      case "--add-script-path":
        opts.addScriptPath = true;
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
      case "--no-line-tracking":
        opts.noLineTracking = true;
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

  const filepath = resolve(process.cwd(), opts.positional[0]);
  const rawCode = readFileSync(filepath, "utf-8");
  const mainFileName = filepath;

  // Extract mip directives (before any parsing)
  const { directives, cleanedSource: code } = extractMipDirectives(
    rawCode,
    filepath
  );

  // Process mip directives — collect additional search paths
  const mipPaths: string[] = [];
  for (const d of directives) {
    if (d.type === "load") {
      for (const result of processMipLoad(d.packageName)) {
        mipPaths.push(...result.paths);
      }
    }
  }

  const searchPaths: string[] = [];
  let workspaceFiles: WorkspaceFile[] = [];
  if (opts.addScriptPath) {
    const scriptDir = dirname(filepath);
    searchPaths.push(scriptDir);
    workspaceFiles = scanMFiles(scriptDir, filepath);
  }
  for (const p of [...opts.extraPaths, ...mipPaths]) {
    searchPaths.push(p);
    workspaceFiles.push(...scanMFiles(p));
  }

  await executeWithOptions(
    code,
    mainFileName,
    workspaceFiles,
    opts,
    searchPaths
  );
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

  const searchPaths: string[] = [];
  const workspaceFiles: WorkspaceFile[] = [];
  for (const p of opts.extraPaths) {
    searchPaths.push(p);
    workspaceFiles.push(...scanMFiles(p));
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
  const profiling = !!opts.profileOutput;
  const totalStart = performance.now();

  // Helper to write profile data after execution completes
  const writeProfileIfNeeded = (result: {
    profileData?: import("./numbl-core/executor/types.js").ProfileData;
  }) => {
    if (!opts.profileOutput || !result.profileData) return;
    const pd = result.profileData;
    const totalTimeMs = performance.now() - totalStart;
    const builtins = Object.entries(pd.builtins)
      .map(([name, b]) => ({
        name,
        totalTimeMs: b.totalTimeMs,
        callCount: b.callCount,
      }))
      .sort((a, b) => b.totalTimeMs - a.totalTimeMs);
    const dispatches = Object.entries(pd.dispatches)
      .map(([name, b]) => ({
        name,
        totalTimeMs: b.totalTimeMs,
        callCount: b.callCount,
      }))
      .sort((a, b) => b.totalTimeMs - a.totalTimeMs);
    const profile = {
      totalTimeMs,
      codegenTimeMs: pd.codegenTimeMs,
      codegenBreakdown: pd.codegenBreakdown,
      executionTimeMs: pd.executionTimeMs,
      jitCompileTimeMs: pd.jitCompileTimeMs,
      builtins,
      dispatches,
    };
    writeFileSync(opts.profileOutput, JSON.stringify(profile, null, 2) + "\n");
    console.error(`Profile written to ${opts.profileOutput}`);
  };

  // If --dump-ast is used alone (without running), just dump and exit
  if (opts.dumpAst && !opts.dumpJs && !opts.stream && !opts.verbose) {
    const { ast } = generateCode(
      code,
      mainFileName,
      workspaceFiles,
      undefined,
      searchPaths
    );
    process.stdout.write(JSON.stringify(ast, null, 2) + "\n");
    process.exit(0);
  }

  const fileIO = new NodeFileIOAdapter();

  // Set up --dump-js: clear file, append JIT pieces during execution, then prepend main script.
  let onJitCompile: ((description: string, jsCode: string) => void) | undefined;
  if (opts.dumpJs) {
    // Clear any previous dump file
    writeFileSync(opts.dumpJs, "");
    onJitCompile = (description: string, jsCode: string) => {
      const header =
        "\n// " +
        "=".repeat(60) +
        "\n" +
        "// JIT: " +
        description +
        "\n" +
        "// " +
        "=".repeat(60) +
        "\n\n";
      appendFileSync(opts.dumpJs!, header + jsCode + "\n");
    };
  }

  try {
    if (opts.stream) {
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
            noLineTracking: opts.noLineTracking,
            fileIO,
          },
          workspaceFiles,
          mainFileName,
          searchPaths
        );
        if (result.plotInstructions.length > 0) {
          streamLine({
            type: "drawnow",
            plotInstructions: result.plotInstructions,
          });
        }
        writeProfileIfNeeded(result);
        if (opts.dumpJs) {
          finalizeDumpFile(opts.dumpJs, mainFileName, result.generatedJS);
        }
        streamLine({ type: "done" });
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
          noLineTracking: opts.noLineTracking,
          fileIO,
        },
        workspaceFiles,
        mainFileName,
        searchPaths
      );
      writeProfileIfNeeded(result);
      if (opts.dumpJs) {
        finalizeDumpFile(opts.dumpJs, mainFileName, result.generatedJS);
      }
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
          noLineTracking: opts.noLineTracking,
          fileIO,
        },
        workspaceFiles,
        mainFileName,
        searchPaths
      );
      writeProfileIfNeeded(result);
      if (opts.dumpJs) {
        finalizeDumpFile(opts.dumpJs, mainFileName, result.generatedJS);
      }
      await flushAndWait(result.plotInstructions);
    }

    process.exit(0);
  } catch (error) {
    const diags = diagnoseErrors(error, code, mainFileName, workspaceFiles);
    console.error(formatDiagnostics(diags));

    // Still finalize the dump file on error so the user can inspect the generated JS
    if (opts.dumpJs) {
      const errWithInfo = error as Error & { generatedJS?: string };
      finalizeDumpFile(
        opts.dumpJs,
        mainFileName,
        errWithInfo.generatedJS ?? "// (main script JS not available)"
      );
    }

    process.exit(1);
  }
}

/** Finalize the dump file: prepend the main script header before JIT pieces. */
function finalizeDumpFile(
  dumpFile: string,
  mainFileName: string,
  jsCode: string
) {
  const header =
    "// " +
    "=".repeat(60) +
    "\n" +
    "// MAIN SCRIPT: " +
    mainFileName +
    "\n" +
    "// " +
    "=".repeat(60) +
    "\n\n";
  // JIT pieces were appended during execution; prepend the main script
  let jitContent = "";
  try {
    jitContent = readFileSync(dumpFile, "utf-8");
  } catch {
    // File may not exist yet if no JIT compilations happened
  }
  writeFileSync(dumpFile, header + jsCode + "\n" + jitContent);
}

async function cmdBuildAddon() {
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
  console.log("");
  try {
    execSync("npx node-gyp rebuild", {
      cwd: packageDir,
      stdio: "inherit",
    });
    console.log("");
    console.log("Native LAPACK addon built successfully.");
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
  process.stdout.write(
    JSON.stringify({
      nativeAddon: nativeAddonLoaded,
      nativeAddonPath: addonPath,
      packageDir,
    }) + "\n"
  );
}

function cmdListBuiltins() {
  const names = getAllBuiltinNames().sort();
  for (const name of names) {
    console.log(name);
  }
}

async function cmdRepl(args: string[]) {
  const opts = parseOptions(args);

  const replSearchPaths = [...opts.extraPaths];
  const replFiles = opts.extraPaths.flatMap(d => scanMFiles(d));
  const replPlotOpts: PlotServerOptions | undefined =
    opts.plotPort !== undefined
      ? { port: opts.plotPort, host: "0.0.0.0" }
      : undefined;
  const { onDrawnow: replDrawnow } = createPlotHandler(
    !opts.plot,
    replPlotOpts
  );
  await runRepl(replFiles, replDrawnow, replSearchPaths);
}

// ── Plot handler ─────────────────────────────────────────────────────────────

function createPlotHandler(disabled: boolean, plotOpts?: PlotServerOptions) {
  if (disabled) {
    return {
      onDrawnow: undefined,
      flushAndWait: async () => {},
    };
  }

  let plotServer: PlotServer | null = null;
  let serverStarting: Promise<PlotServer> | null = null;
  const pendingBatches: PlotInstruction[][] = [];

  const onDrawnow = (instructions: PlotInstruction[]) => {
    if (plotServer) {
      plotServer.sendInstructions(instructions);
    } else {
      pendingBatches.push(instructions);
      if (!serverStarting) {
        serverStarting = startPlotServer(plotOpts).then(ps => {
          plotServer = ps;
          for (const batch of pendingBatches) {
            ps.sendInstructions(batch);
          }
          pendingBatches.length = 0;
          return ps;
        });
      }
    }
  };

  const flushAndWait = async (remaining: PlotInstruction[]) => {
    if (remaining.length > 0) {
      onDrawnow(remaining);
    }
    if (serverStarting) {
      plotServer = await serverStarting;
    }
    if (plotServer) {
      plotServer.scriptDone();
      await plotServer.closed;
    }
  };

  return { onDrawnow, flushAndWait };
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
  const builtins: { name: string; totalTimeMs: number; callCount: number }[] =
    data.builtins ?? [];
  const sumDispatches = dispatches.reduce((s, d) => s + d.totalTimeMs, 0);
  const sumBuiltins = builtins.reduce((s, b) => s + b.totalTimeMs, 0);
  const otherTime =
    data.executionTimeMs - jitTime - sumDispatches - sumBuiltins;

  console.log(`Total time:      ${fmt(data.totalTimeMs)} ms`);
  console.log(`Codegen time:    ${fmt(data.codegenTimeMs)} ms`);
  if (data.codegenBreakdown) {
    const cb = data.codegenBreakdown;
    const cgTime = data.codegenTimeMs || 1;
    const cgPct = (v: number) => fmt((v / cgTime) * 100);
    console.log(
      `  Parse main:    ${fmt(cb.parseMainMs)} ms (${cgPct(cb.parseMainMs)}%)`
    );
    console.log(
      `  Parse workspace: ${fmt(cb.parseWorkspaceMs)} ms (${cgPct(cb.parseWorkspaceMs)}%)`
    );
    console.log(
      `  Load JS funcs: ${fmt(cb.loadJsUserFunctionsMs)} ms (${cgPct(cb.loadJsUserFunctionsMs)}%)`
    );
    console.log(
      `  Registration:  ${fmt(cb.registrationMs)} ms (${cgPct(cb.registrationMs)}%)`
    );
    console.log(
      `  Func index:    ${fmt(cb.buildFunctionIndexMs)} ms (${cgPct(cb.buildFunctionIndexMs)}%)`
    );
    console.log(
      `  Lower main:    ${fmt(cb.lowerMainMs)} ms (${cgPct(cb.lowerMainMs)}%)`
    );
    console.log(
      `  Codegen emit:  ${fmt(cb.codegenMs)} ms (${cgPct(cb.codegenMs)}%)`
    );
  }
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
      pad("Calls", 10) +
      pad("Self (ms)", 13) +
      pad("Avg (ms)", 11) +
      pad("% Exec", 9);
    console.log("\nBuiltin functions:");
    console.log(header);
    console.log("─".repeat(header.length));

    for (const b of builtins) {
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
  } else {
    console.log("\nNo builtin functions were called.");
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);

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
    case "run-tests": {
      const dir =
        rest.length > 0 ? rest[0] : join(packageDir, "numbl_test_scripts");
      await runTests(dir);
      break;
    }
    case "build-addon":
      await cmdBuildAddon();
      break;
    case "info":
      cmdInfo();
      break;
    case "list-builtins":
      cmdListBuiltins();
      break;
    case "mip":
      await cmdMip(rest);
      break;
    case "show-profile":
      cmdShowProfile(rest);
      break;
    default:
      console.error(`Unknown command: ${command}`);
      console.error('Run "numbl --help" for usage information.');
      process.exit(1);
  }
}

main();

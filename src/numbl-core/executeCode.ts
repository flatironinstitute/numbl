/**
 * Entry point for code execution.
 *
 * Parses source, sets up the LoweringContext and Runtime,
 * creates an Interpreter, and runs the AST.
 */

import type { RuntimeValue } from "./runtime/index.js";
import type { PlotInstruction } from "../graphics/types.js";
import type { FileIOAdapter } from "./fileIOAdapter.js";
import type { SystemAdapter } from "./systemAdapter.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { Runtime } from "./runtime/runtime.js";
import { RTV } from "./runtime/constructors.js";
import { RuntimeError } from "../numbl-core/runtime/index.js";
import { loadJsUserFunctions } from "./jsUserFunctions.js";
import {
  registerDynamicIBuiltin,
  unregisterIBuiltin,
  getIBuiltin,
} from "./interpreter/builtins/types.js";
import type { IBuiltin } from "./interpreter/builtins/types.js";
import type { NativeBridge } from "./workspace/index.js";
import { parseMFile, type Stmt } from "./parser/index.js";
import { SyntaxError } from "./parser/errors.js";
import { Interpreter } from "./interpreter/interpreter.js";
import { LoweringContext } from "./lowering/loweringContext.js";
import { stdlibFiles, shimFiles } from "./stdlib-bundle.js";
import { jitHelpers } from "./interpreter/jit/jitHelpers.js";
import { resetAppdataStore } from "./interpreter/builtins/misc.js";

// ── Public API types ────────────────────────────────────────────────────

export interface ExecOptions {
  onOutput?: (text: string) => void;
  onDrawnow?: (plotInstructions: PlotInstruction[]) => void;
  displayResults?: boolean;
  maxIterations?: number;
  initialVariableValues?: Record<string, RuntimeValue>;
  /** Optional callback for verbose compilation logging. */
  log?: (message: string) => void;
  /** Enable profiling of builtin function calls. */
  profile?: boolean;
  /** Called each time a JIT function is compiled, with a description and the generated JS. */
  onJitCompile?: (description: string, jsCode: string) => void;
  /** Initial hold state for plotting (persisted across REPL executions). */
  initialHoldState?: boolean;
  /** Override or add builtins for this execution only. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  customBuiltins?: Record<string, (nargout: number, args: any[]) => any>;
  /** Platform-specific file I/O adapter (e.g. Node.js fs). */
  fileIO?: FileIOAdapter;
  /** Platform-specific system adapter (env vars, cwd, platform info). */
  system?: SystemAdapter;
  /** Synchronous callback for the `input()` builtin. Displays prompt, returns user's line. */
  onInput?: (prompt: string) => string;
  /** Optimization level for interpreter (0 = none, >=1 = JIT scalar functions). */
  optimization?: number;
  /**
   * Initial implicit cwd path for the MATLAB-style "cwd is the first search path" feature.
   * - undefined → auto-detect from `system.cwd()` and scan its files.
   * - a string → use this absolute path as the implicit cwd (REPL persistence).
   * - null → opt out of the implicit-cwd behavior entirely.
   */
  implicitCwdPath?: string | null;
}

export interface BuiltinProfileEntry {
  totalTimeMs: number;
  callCount: number;
}

export interface BuiltinProfileBreakdown {
  /** Calls from the registry fallback (rt.builtins). */
  fallback: BuiltinProfileEntry;
  /** Calls from the interpreter (IBuiltin.resolve → apply). */
  interp: BuiltinProfileEntry;
  /** Calls from JIT-compiled code (ib_* helpers). */
  jit: BuiltinProfileEntry;
}

export interface ProfileData {
  executionTimeMs: number;
  jitCompileTimeMs: number;
  builtins: Record<string, BuiltinProfileBreakdown>;
  dispatches: Record<string, BuiltinProfileEntry>;
}

export interface ExecResult {
  output: string[];
  generatedJS: string;
  plotInstructions: PlotInstruction[];
  returnValue: RuntimeValue;
  variableValues: Record<string, RuntimeValue>;
  holdState: boolean;
  profileData?: ProfileData;
  dispatchUnknownCounts?: Record<string, number>;
  /** Updated search paths (set when addpath/rmpath was called). */
  searchPaths?: string[];
  /** Updated workspace files (set when addpath/rmpath was called). */
  workspaceFiles?: WorkspaceFile[];
  /** Final implicit cwd path (for REPL persistence across commands). */
  implicitCwdPath?: string | null;
}

// ── Implementation ──────────────────────────────────────────────────────

/** Virtual search path prefix for bundled shim files. */
const SHIM_SEARCH_PATH = "__numbl_shims__";

export function executeCode(
  source: string,
  options: ExecOptions = {},
  workspaceFiles?: WorkspaceFile[],
  mainFileName: string = "script.m",
  searchPaths?: string[],
  nativeBridge?: NativeBridge
): ExecResult {
  // Reset module-level mutable state so separate runs don't bleed
  resetAppdataStore();

  // ── 1. Parse main file ──────────────────────────────────────────────
  const ast = parseMFile(source, mainFileName);

  if (options.log) {
    options.log("AST parsed, starting interpretation");
  }

  // Separate local functions and class definitions from main body
  const isRepl = mainFileName === "repl";
  const localFunctions: (Stmt & { type: "Function" })[] = [];
  const localClasses: (Stmt & { type: "ClassDef" })[] = [];
  for (const stmt of ast.body) {
    if (stmt.type === "Function") {
      if (isRepl) {
        throw new RuntimeError(
          "Function definitions are not supported in the REPL. Save the function to a .m file instead.",
          stmt.span
        );
      }
      localFunctions.push(stmt);
    } else if (stmt.type === "ClassDef") {
      localClasses.push(stmt);
    }
  }

  // ── 2. Set up LoweringContext for resolution ────────────────────────
  const ctx = new LoweringContext(source, mainFileName);

  // Separate .m workspace files from .js/.wasm
  const mWorkspaceFiles: WorkspaceFile[] = [];
  const jsWorkspaceFiles: WorkspaceFile[] = [];
  const wasmWorkspaceFiles: WorkspaceFile[] = [];
  if (workspaceFiles) {
    for (const f of workspaceFiles) {
      if (f.name.endsWith(".m")) {
        mWorkspaceFiles.push(f);
      } else if (f.name.endsWith(".js")) {
        jsWorkspaceFiles.push(f);
      } else if (f.name.endsWith(".wasm")) {
        wasmWorkspaceFiles.push(f);
      }
    }
  }

  // ── Implicit cwd as first-priority search path (MATLAB semantics) ──
  // The implicit cwd scan only collects .m files. Loading arbitrary .js
  // files from cwd would be too aggressive (e.g. eslint.config.js in a
  // project root would be misinterpreted as a numbl user function); .js
  // user functions remain explicit via `addpath`.
  let implicitCwdPath: string | null = null;
  if (
    options.implicitCwdPath !== null &&
    options.system &&
    options.fileIO?.scanDirectory
  ) {
    try {
      const cwd = options.implicitCwdPath ?? options.system.cwd();
      const absCwd = options.fileIO.resolvePath?.(cwd) ?? cwd;
      // Skip if this dir is already explicitly on the search path
      const explicitPaths = searchPaths ?? [];
      if (!explicitPaths.includes(absCwd)) {
        implicitCwdPath = absCwd;
        // Skip scan if we already received files from this dir via the
        // workspaceFiles argument (REPL persistence path).
        const prefix = absCwd.endsWith("/") ? absCwd : absCwd + "/";
        const alreadyHave = mWorkspaceFiles.some(
          f => f.name === absCwd || f.name.startsWith(prefix)
        );
        if (!alreadyHave) {
          const cwdFiles = options.fileIO.scanDirectory(absCwd);
          for (const f of cwdFiles) {
            if (f.name.endsWith(".m")) {
              mWorkspaceFiles.push(f);
            }
          }
        }
      }
    } catch {
      // cwd inaccessible or scan failed — silently disable implicit cwd
      implicitCwdPath = null;
    }
  }

  // Load .js user functions
  const jsUserFunctions = loadJsUserFunctions(
    jsWorkspaceFiles,
    wasmWorkspaceFiles,
    nativeBridge
  );
  const jsUserFunctionNames = jsUserFunctions.map(ib => ib.name);

  // Add stdlib and shim files (track names for filtering in ExecResult)
  const stdlibShimNames = new Set<string>();
  for (const f of stdlibFiles) {
    mWorkspaceFiles.push(f);
    stdlibShimNames.add(f.name);
  }
  for (const f of shimFiles) {
    const shimName = `${SHIM_SEARCH_PATH}/${f.name}`;
    mWorkspaceFiles.push({ name: shimName, source: f.source });
    stdlibShimNames.add(shimName);
  }
  ctx.registry.searchPaths = [...(searchPaths ?? []), SHIM_SEARCH_PATH];

  // Prepend implicit cwd at position 0 (highest priority).
  if (implicitCwdPath !== null) {
    ctx.registry.searchPaths.unshift(implicitCwdPath);
  }

  // Pre-parse all .m workspace files into the shared AST cache.
  // Files that fail to parse are skipped with a warning instead of aborting.
  ctx.fileASTCache.set(mainFileName, ast);
  const skippedFiles = new Set<string>();
  for (const f of mWorkspaceFiles) {
    try {
      ctx.fileASTCache.set(f.name, parseMFile(f.source, f.name));
    } catch (e) {
      skippedFiles.add(f.name);
      if (e instanceof SyntaxError) {
        console.warn(
          `Warning: skipping ${f.name} (syntax error at line ${e.line ?? "?"})`
        );
      } else {
        console.warn(`Warning: skipping ${f.name} (parse error)`);
      }
    }
  }
  // Remove skipped files so they aren't registered in the workspace
  if (skippedFiles.size > 0) {
    for (let i = mWorkspaceFiles.length - 1; i >= 0; i--) {
      if (skippedFiles.has(mWorkspaceFiles[i].name)) {
        mWorkspaceFiles.splice(i, 1);
      }
    }
  }

  // Register local functions from main file
  for (const fn of localFunctions) {
    ctx.registerLocalFunctionAST(fn);
  }

  // Register local classes from main file
  for (const cls of localClasses) {
    ctx.registerLocalClass(cls);
  }

  // Register .m workspace files for resolution
  if (mWorkspaceFiles.length > 0) {
    ctx.registerWorkspaceFiles(mWorkspaceFiles);
  }

  // Build function index (enables definitive resolution)
  const functionIndex = ctx.buildFunctionIndex(jsUserFunctionNames);

  // ── 3. Create runtime ──────────────────────────────────────────────
  const rt = new Runtime(options, options.initialVariableValues);

  // Register .js user functions as IBuiltins (save originals for cleanup)
  const savedIBuiltins = new Map<string, IBuiltin>();
  for (const ib of jsUserFunctions) {
    const orig = getIBuiltin(ib.name);
    if (orig) savedIBuiltins.set(ib.name, orig);
    registerDynamicIBuiltin(ib);
  }

  // Apply custom builtins (both to rt.builtins fallback and to customBuiltins
  // which take priority over IBuiltins in the interpreter)
  if (options.customBuiltins) {
    Object.assign(rt.builtins, options.customBuiltins);
    Object.assign(rt.customBuiltins, options.customBuiltins);
  }

  // ── 4. Create interpreter and wire up runtime callbacks ────────────
  const interpreter = new Interpreter(
    rt,
    ctx,
    functionIndex,
    mainFileName,
    options.initialVariableValues
  );

  // Populate file→source map for line number computation in error messages
  interpreter.fileSources.set(mainFileName, source);
  for (const f of mWorkspaceFiles) {
    interpreter.fileSources.set(f.name, f.source);
  }

  interpreter.optimization = options.optimization ?? 1;

  // Collect JIT compilations for generatedJS output and profiling
  const jitSections: string[] = [];
  interpreter.onJitCompile = (description: string, jsCode: string) => {
    jitSections.push(
      `// ${"=".repeat(60)}\n// JIT: ${description}\n// ${"=".repeat(60)}\n\n${jsCode}`
    );
    options.onJitCompile?.(description, jsCode);
  };

  // Wire up JIT builtin profiling hooks when profiling is enabled
  if (options.profile) {
    (jitHelpers as Record<string, unknown>)._profileEnter = (key: string) =>
      rt.profileEnter(key);
    (jitHelpers as Record<string, unknown>)._profileLeave = () =>
      rt.profileLeave();
  }

  // Wire up compileSpecialized so runtime dispatch routes through interpreter
  interpreter.installRuntimeCallbacks();

  // Expose search paths reference on the runtime (for path() builtin)
  rt.searchPaths = ctx.registry.searchPaths;

  // Track whether paths were modified during execution
  let pathsModified = false;

  // Shared rebuild logic used by both onPathChange and onCwdChange.
  // Re-sorts mWorkspaceFiles by search path order, then re-registers
  // and rebuilds the function index.
  const rebuildWorkspace = () => {
    const paths = ctx.registry.searchPaths;
    // Compute file → priority by LONGEST matching prefix.
    // (A simple findIndex would mis-classify a file in /a/b/c.m as
    //  belonging to /a if /a is also on the path — when /a is the cwd
    //  and /a/b is an explicit addpath, the file should belong to /a/b.)
    const priorityOf = (name: string): number => {
      let bestIdx = paths.length;
      let bestLen = -1;
      for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        const withSep = p.endsWith("/") ? p : p + "/";
        if (name === p || name.startsWith(withSep)) {
          if (p.length > bestLen) {
            bestLen = p.length;
            bestIdx = i;
          }
        }
      }
      return bestIdx;
    };
    mWorkspaceFiles.sort((a, b) => priorityOf(a.name) - priorityOf(b.name));

    ctx.clearWorkspaceRegistrations();
    ctx.registerWorkspaceFiles(mWorkspaceFiles);
    const newIndex = ctx.buildFunctionIndex(jsUserFunctionNames);
    interpreter.functionIndex = newIndex;
    interpreter.clearAllCaches();
    pathsModified = true;
  };

  // Wire up addpath/rmpath callback
  rt.onPathChange = (action, dir, position) => {
    const fileIO = options.fileIO;
    const absDir = fileIO?.resolvePath?.(dir) ?? dir;

    if (action === "add") {
      // Skip if already on the path
      if (ctx.registry.searchPaths.includes(absDir)) return;

      // Add to search paths (before the shim path for '-end').
      // The implicit cwd entry, if present, must remain at position 0 — so
      // a "begin" insertion goes to index 1 when cwd is currently first.
      if (position === "begin") {
        const cwdIsFirst =
          implicitCwdPath !== null &&
          ctx.registry.searchPaths[0] === implicitCwdPath;
        ctx.registry.searchPaths.splice(cwdIsFirst ? 1 : 0, 0, absDir);
      } else {
        const shimIdx = ctx.registry.searchPaths.indexOf(SHIM_SEARCH_PATH);
        if (shimIdx >= 0) {
          ctx.registry.searchPaths.splice(shimIdx, 0, absDir);
        } else {
          ctx.registry.searchPaths.push(absDir);
        }
      }

      // Scan and parse new files
      if (fileIO?.scanDirectory) {
        const newFiles = fileIO.scanDirectory(absDir);
        const newJsFiles: WorkspaceFile[] = [];
        const newWasmFiles: WorkspaceFile[] = [];
        for (const f of newFiles) {
          if (f.name.endsWith(".m") && !ctx.fileASTCache.has(f.name)) {
            try {
              ctx.fileASTCache.set(f.name, parseMFile(f.source, f.name));
            } catch (e) {
              if (e instanceof SyntaxError && e.file === null) {
                e.file = f.name;
              }
              throw e;
            }
            interpreter.fileSources.set(f.name, f.source);
            mWorkspaceFiles.push(f);
          } else if (f.name.endsWith(".js")) {
            newJsFiles.push(f);
          } else if (f.name.endsWith(".wasm")) {
            newWasmFiles.push(f);
          }
        }
        // Load any new .js/.wasm user functions
        if (newJsFiles.length > 0) {
          const newIBuiltins = loadJsUserFunctions(
            newJsFiles,
            newWasmFiles,
            nativeBridge
          );
          for (const ib of newIBuiltins) {
            const orig = getIBuiltin(ib.name);
            if (orig) savedIBuiltins.set(ib.name, orig);
            registerDynamicIBuiltin(ib);
            if (!jsUserFunctionNames.includes(ib.name)) {
              jsUserFunctionNames.push(ib.name);
            }
          }
        }
      }
    } else {
      // rmpath: remove from search paths
      const idx = ctx.registry.searchPaths.indexOf(absDir);
      if (idx >= 0) {
        ctx.registry.searchPaths.splice(idx, 1);
      }
      // Remove files belonging to this path from AST cache and mWorkspaceFiles
      const prefix = absDir.endsWith("/") ? absDir : absDir + "/";
      for (let i = mWorkspaceFiles.length - 1; i >= 0; i--) {
        if (mWorkspaceFiles[i].name.startsWith(prefix)) {
          ctx.fileASTCache.delete(mWorkspaceFiles[i].name);
          interpreter.fileSources.delete(mWorkspaceFiles[i].name);
          mWorkspaceFiles.splice(i, 1);
        }
      }
    }

    rebuildWorkspace();
  };

  // Wire up cd callback — updates the implicit cwd entry in searchPaths
  // and rebuilds the workspace using the same pipeline as addpath/rmpath.
  rt.onCwdChange = (newCwd: string) => {
    const fileIO = options.fileIO;
    const absNewCwd = fileIO?.resolvePath?.(newCwd) ?? newCwd;

    // No-op if cwd is already the implicit path
    if (implicitCwdPath === absNewCwd) return;

    // Remove files from the OLD implicit cwd (if any). Only remove files
    // whose CLOSEST search path is the old implicit cwd — files in
    // addpath'd subdirectories of the old cwd belong to those subdirs and
    // must survive (e.g. addpath('/a/b/lib') then cd('/a/b') then cd
    // elsewhere should keep /a/b/lib/foo.m visible via the addpath entry).
    if (implicitCwdPath !== null) {
      const oldPrefix = implicitCwdPath.endsWith("/")
        ? implicitCwdPath
        : implicitCwdPath + "/";
      // Collect all explicit search paths that are deeper than the old
      // implicit cwd; a file under one of those paths "belongs" to it.
      const deeperPaths = ctx.registry.searchPaths.filter(
        p =>
          p !== implicitCwdPath &&
          p !== SHIM_SEARCH_PATH &&
          (p === implicitCwdPath || p.startsWith(oldPrefix))
      );
      const fileBelongsToDeeperPath = (fname: string): boolean => {
        for (const p of deeperPaths) {
          const withSep = p.endsWith("/") ? p : p + "/";
          if (fname === p || fname.startsWith(withSep)) return true;
        }
        return false;
      };
      for (let i = mWorkspaceFiles.length - 1; i >= 0; i--) {
        const fname = mWorkspaceFiles[i].name;
        if (
          (fname === implicitCwdPath || fname.startsWith(oldPrefix)) &&
          !fileBelongsToDeeperPath(fname)
        ) {
          ctx.fileASTCache.delete(fname);
          interpreter.fileSources.delete(fname);
          mWorkspaceFiles.splice(i, 1);
        }
      }
      const oldIdx = ctx.registry.searchPaths.indexOf(implicitCwdPath);
      if (oldIdx >= 0) ctx.registry.searchPaths.splice(oldIdx, 1);
      implicitCwdPath = null;
    }

    // If new cwd is already an explicit search path, just rebuild — no
    // need to add it again or scan its files.
    if (ctx.registry.searchPaths.includes(absNewCwd)) {
      rebuildWorkspace();
      return;
    }

    // Insert new cwd at position 0 (highest priority)
    ctx.registry.searchPaths.unshift(absNewCwd);
    implicitCwdPath = absNewCwd;

    // Scan and parse .m files in the new cwd. As with the initial setup,
    // .js/.wasm files from the implicit cwd are NOT auto-loaded — that
    // remains an explicit `addpath` opt-in to avoid misinterpreting random
    // .js files (e.g. eslint configs) as numbl user functions.
    if (fileIO?.scanDirectory) {
      let newFiles: WorkspaceFile[] = [];
      try {
        newFiles = fileIO.scanDirectory(absNewCwd);
      } catch {
        // cwd inaccessible — leave the entry but skip scanning
      }
      for (const f of newFiles) {
        if (f.name.endsWith(".m") && !ctx.fileASTCache.has(f.name)) {
          try {
            ctx.fileASTCache.set(f.name, parseMFile(f.source, f.name));
          } catch (e) {
            // Warn-and-skip (matches startup behavior at the top of executeCode);
            // a single broken .m file should not make `cd` unusable.
            if (e instanceof SyntaxError) {
              console.warn(
                `Warning: skipping ${f.name} (syntax error at line ${e.line ?? "?"})`
              );
            } else {
              console.warn(`Warning: skipping ${f.name} (parse error)`);
            }
            continue;
          }
          interpreter.fileSources.set(f.name, f.source);
          mWorkspaceFiles.push(f);
        }
      }
    }

    rebuildWorkspace();
  };

  // Wire up eval callback
  rt.evalLocalCallback = (code, initialVars, onOutput, fileName) => {
    // Propagate the parent's effective workspace state into the nested call
    // so functions resolved via addpath remain visible inside eval()/run().
    // Filter out the SHIM sentinel and the implicit cwd from searchPaths
    // (the nested call will set up its own implicit cwd from `implicitCwdPath`),
    // and filter out stdlib/shim files from workspaceFiles (the nested call
    // re-adds them itself).
    const nestedSearchPaths = ctx.registry.searchPaths.filter(
      p => p !== SHIM_SEARCH_PATH && p !== implicitCwdPath
    );
    const nestedWorkspaceFiles = mWorkspaceFiles.filter(
      f => !stdlibShimNames.has(f.name)
    );
    const evalResult = executeCode(
      code,
      {
        onOutput,
        displayResults: false,
        initialVariableValues: initialVars,
        fileIO: options.fileIO,
        system: options.system,
        onInput: options.onInput,
        implicitCwdPath,
      },
      nestedWorkspaceFiles,
      fileName,
      nestedSearchPaths,
      nativeBridge
    );
    // If the nested execution updated its implicit cwd (e.g. cd inside the
    // evaluated code), reflect it in the parent so subsequent statements see
    // the same effective search path.
    if (evalResult.implicitCwdPath !== undefined) {
      implicitCwdPath = evalResult.implicitCwdPath;
    }
    return {
      returnValue: evalResult.returnValue,
      variableValues: evalResult.variableValues,
      searchPaths: evalResult.searchPaths,
      workspaceFiles: evalResult.workspaceFiles,
    };
  };

  // ── 5. Run ─────────────────────────────────────────────────────────
  try {
    const execStart = performance.now();
    interpreter.run(ast);
    const executionTimeMs = performance.now() - execStart;

    const result: ExecResult = {
      output: rt.outputLines,
      generatedJS:
        jitSections.length > 0
          ? `// Interpreter mode — JIT compiled sections:\n\n${jitSections.join("\n\n")}`
          : "// No JS generated",
      plotInstructions: rt.plotInstructions,
      returnValue: interpreter.ans ?? RTV.num(0),
      variableValues: interpreter.getVariableValues(),
      holdState: rt.holdState,
    };

    if (options.profile) {
      result.profileData = {
        executionTimeMs,
        jitCompileTimeMs: rt.getJitCompileTimeMs(),
        builtins: rt.getBuiltinProfile(),
        dispatches: rt.getDispatchProfile(),
      };
    }

    // Propagate path changes so callers (e.g. REPL) can persist them
    if (pathsModified) {
      result.searchPaths = ctx.registry.searchPaths.filter(
        p => p !== SHIM_SEARCH_PATH && p !== implicitCwdPath
      );
      result.workspaceFiles = mWorkspaceFiles.filter(
        f => !stdlibShimNames.has(f.name)
      );
    }
    // Always propagate the implicit cwd so REPL callers can persist it
    // across commands (cd inside one command should affect the next).
    result.implicitCwdPath = implicitCwdPath;

    return result;
  } catch (e) {
    // Attach collected JIT code to the error so callers (e.g. --dump-js) can inspect it
    const generatedJS =
      jitSections.length > 0
        ? `// Interpreter mode — JIT compiled sections:\n\n${jitSections.join("\n\n")}`
        : "// No JS generated";
    if (e instanceof RuntimeError) {
      // Annotate with file/line info
      if (e.line === null && rt.$file && rt.$line > 0) {
        e.file = rt.$file;
        e.line = rt.$line;
      }
      if (!e.fileSources) e.fileSources = interpreter.fileSources;
      (e as RuntimeError & { generatedJS?: string }).generatedJS = generatedJS;
      throw e;
    }
    const re = new RuntimeError(e instanceof Error ? e.message : String(e));
    if (rt.$file && rt.$line > 0) {
      re.file = rt.$file;
      re.line = rt.$line;
    }
    re.fileSources = interpreter.fileSources;
    (re as RuntimeError & { generatedJS?: string }).generatedJS = generatedJS;
    throw re;
  } finally {
    // Reset JIT profiling hooks to no-ops
    if (options.profile) {
      (jitHelpers as Record<string, unknown>)._profileEnter =
        Function.prototype;
      (jitHelpers as Record<string, unknown>)._profileLeave =
        Function.prototype;
    }
    // Restore or unregister .js user function IBuiltins
    for (const ib of jsUserFunctions) {
      const orig = savedIBuiltins.get(ib.name);
      if (orig) {
        registerDynamicIBuiltin(orig);
      } else {
        unregisterIBuiltin(ib.name);
      }
    }
  }
}

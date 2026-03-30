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
  const localFunctions: (Stmt & { type: "Function" })[] = [];
  const localClasses: (Stmt & { type: "ClassDef" })[] = [];
  for (const stmt of ast.body) {
    if (stmt.type === "Function") {
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

  // Wire up addpath/rmpath callback
  rt.onPathChange = (action, dir, position) => {
    const fileIO = options.fileIO;
    const absDir = fileIO?.resolvePath?.(dir) ?? dir;

    if (action === "add") {
      // Skip if already on the path
      if (ctx.registry.searchPaths.includes(absDir)) return;

      // Add to search paths (before the shim path for '-end')
      if (position === "begin") {
        ctx.registry.searchPaths.unshift(absDir);
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

    // Clear workspace registrations and rebuild.
    // Sort files by search path order so earlier paths win in first-wins registration.
    const paths = ctx.registry.searchPaths;
    mWorkspaceFiles.sort((a, b) => {
      const ai = paths.findIndex(
        p => a.name.startsWith(p.endsWith("/") ? p : p + "/") || a.name === p
      );
      const bi = paths.findIndex(
        p => b.name.startsWith(p.endsWith("/") ? p : p + "/") || b.name === p
      );
      // Files not matching any search path (e.g. stdlib) go last
      const aPri = ai >= 0 ? ai : paths.length;
      const bPri = bi >= 0 ? bi : paths.length;
      return aPri - bPri;
    });

    ctx.clearWorkspaceRegistrations();
    ctx.registerWorkspaceFiles(mWorkspaceFiles);
    const newIndex = ctx.buildFunctionIndex(jsUserFunctionNames);
    interpreter.functionIndex = newIndex;
    interpreter.clearAllCaches();
    pathsModified = true;
  };

  // Wire up eval callback
  rt.evalLocalCallback = (code, initialVars, onOutput, fileName) => {
    const evalResult = executeCode(
      code,
      {
        onOutput,
        displayResults: false,
        initialVariableValues: initialVars,
        fileIO: options.fileIO,
        onInput: options.onInput,
      },
      undefined,
      fileName
    );
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
          : "// interpreted mode — no JS generated",
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
        p => p !== SHIM_SEARCH_PATH
      );
      result.workspaceFiles = mWorkspaceFiles.filter(
        f => !stdlibShimNames.has(f.name)
      );
    }

    return result;
  } catch (e) {
    // Attach collected JIT code to the error so callers (e.g. --dump-js) can inspect it
    const generatedJS =
      jitSections.length > 0
        ? `// Interpreter mode — JIT compiled sections:\n\n${jitSections.join("\n\n")}`
        : "// interpreted mode — no JS generated";
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

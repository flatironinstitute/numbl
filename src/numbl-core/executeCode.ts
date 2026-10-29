/**
 * Entry point for code execution.
 *
 * Parses source, indexes the workspace, sets up the Runtime,
 * creates an Interpreter, and runs the AST.
 */

// legacy user-defined functions
(
  globalThis as typeof globalThis & { FloatXArray: typeof Float64Array }
).FloatXArray = Float64Array;

import type { RuntimeValue } from "./runtime/index.js";
import type { PlotInstruction } from "./plotting/types.js";
import type { FileIOAdapter } from "./fileIOAdapter.js";
import { fileIOView } from "./fileTable.js";
import type { SystemAdapter } from "./systemAdapter.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { Runtime } from "./runtime/runtime.js";
import { RTV } from "./runtime/constructors.js";
import {
  RuntimeError,
  CancellationError,
} from "../numbl-core/runtime/index.js";
import type { NativeBridge } from "./workspace/index.js";
import { Workspace, SHIM_SEARCH_PATH } from "./workspace/workspace.js";
import { parseMFile } from "./parser/index.js";
import { Interpreter } from "./interpreter/interpreter.js";
import { stdlibFiles, shimFiles } from "./stdlib-bundle.js";
import { SessionState } from "./sessionState.js";
import { defaultHost, type NumblHost } from "./host.js";
import { registerExecutorsForOpt } from "./executors/plugins.js";
import { pushCurrentRuntime, popCurrentRuntime } from "./runtime/refcount.js";
import {
  createUihtmlSession,
  type UihtmlSession,
} from "./runtime/uihtmlSession.js";

export type { UihtmlSession } from "./runtime/uihtmlSession.js";

// ── Public API types ────────────────────────────────────────────────────

export interface ExecOptions {
  onOutput?: (text: string) => void;
  /** Standard-error text (`fprintf(2, ...)`); when omitted it goes to
   *  `onOutput`. */
  onErrorOutput?: (text: string) => void;
  onDrawnow?: (plotInstructions: PlotInstruction[]) => void;
  displayResults?: boolean;
  maxIterations?: number;
  /** State that outlives the run (base workspace, globals, persistents,
   *  warning and PRNG state, ...). An `ExecSession` passes its own; when
   *  omitted the run gets a fresh one. */
  session?: SessionState;
  /** Variables to add to the base workspace before the run. */
  initialVariableValues?: Record<string, RuntimeValue>;
  /** Optional callback for verbose compilation logging. */
  log?: (message: string) => void;
  /** Enable profiling of builtin function calls. */
  profile?: boolean;
  /** Called each time a JIT function is compiled, with a description, the
   *  generated source, and the backend language (`"js"` or `"c"`). */
  onJitCompile?: (description: string, code: string, lang: "js" | "c") => void;
  /** Called when a JIT-compiled unit bails to the interpreter at runtime
   *  (e.g. an indexed-store array growth the JIT can't model). Surfaced
   *  as a warning; the CLI routes it to stderr. */
  onJitBail?: (message: string) => void;
  /** Initial hold state for plotting (persisted across REPL executions). */
  initialHoldState?: boolean;
  /** Current and largest figure handle so far (persisted across REPL
   *  executions, so a bare `figure` keeps allocating new figures). */
  initialFigureHandles?: FigureHandles;
  /** Override or add builtins for this execution only. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  customBuiltins?: Record<string, (nargout: number, args: any[]) => any>;
  /** The platform's capabilities (numeric providers, geometry backends,
   *  C-JIT, native libraries, gpu; see host.ts). An `ExecSession` passes its
   *  own; when omitted the run uses the process-default host. */
  host?: NumblHost;
  /** Platform-specific file I/O adapter (e.g. Node.js fs); default
   *  `host.fileIO`. */
  fileIO?: FileIOAdapter;
  /** Platform-specific system adapter (env vars, cwd, platform info);
   *  default `host.system`. */
  system?: SystemAdapter;
  /** Synchronous callback for the `input()` builtin. Displays prompt, returns user's line. */
  onInput?: (prompt: string) => string;
  /** Optimization mode: `"0"` interpreter, `"1"` JS-JIT, `"2"` C-JIT. */
  optimization?: import("./executors/plugins.js").OptLevel;
  /**
   * Initial implicit cwd path for the MATLAB-style "cwd is the first search path" feature.
   * - undefined → auto-detect from `system.cwd()` and scan its files.
   * - a string → use this absolute path as the implicit cwd (REPL persistence).
   * - null → opt out of the implicit-cwd behavior entirely.
   */
  implicitCwdPath?: string | null;
  /** SharedArrayBuffer for cooperative cancellation. Int32[0] != 0 means
   *  cancelled. Make it with `createCancelBuffer` (8 bytes) so Stop also wakes
   *  a blocked input(), pause or gpu.call at once (see blockingChannel.ts). */
  cancelSAB?: SharedArrayBuffer;
  /** Hook for `sendEventToHTMLSource(src,name,data)` — pushes an event from a
   *  uihtml callback back to the component's page (MATLAB → JS). `dataJson` is
   *  the data already `jsonencode`d. The host forwards it to the iframe. */
  onHtmlSourceEvent?: (compId: string, name: string, dataJson: string) => void;
}

export interface BuiltinProfileEntry {
  totalTimeMs: number;
  callCount: number;
}

export interface BuiltinProfileBreakdown {
  /** Calls from the registry fallback (rt.builtins). */
  fallback: BuiltinProfileEntry;
  /** Calls from the interpreter (invokeBuiltin). */
  interp: BuiltinProfileEntry;
  /** Calls from JIT-compiled code (ib_* helpers). */
  jit: BuiltinProfileEntry;
}

export interface HotLoopEntry {
  file: string;
  line: number;
  kind: "for" | "while";
  iterations: number;
  callCount: number;
  totalTimeMs: number;
}

export interface ProfileData {
  executionTimeMs: number;
  jitCompileTimeMs: number;
  builtins: Record<string, BuiltinProfileBreakdown>;
  dispatches: Record<string, BuiltinProfileEntry>;
  hotLoops: HotLoopEntry[];
}

export interface ExecResult {
  output: string[];
  generatedJS: string;
  /** C source emitted by the C-JIT backend (`--opt 2`). Empty placeholder
   *  when no C was generated (e.g. `--opt 0`/`1`). */
  generatedC: string;
  plotInstructions: PlotInstruction[];
  returnValue: RuntimeValue;
  variableValues: Record<string, RuntimeValue>;
  holdState: boolean;
  figureHandles: FigureHandles;
  profileData?: ProfileData;
  dispatchUnknownCounts?: Record<string, number>;
  /** Updated search paths (set when addpath/rmpath was called). */
  searchPaths?: string[];
  /** Updated workspace files (set when addpath/rmpath was called). */
  workspaceFiles?: WorkspaceFile[];
  /** Final implicit cwd path (for REPL persistence across commands). */
  implicitCwdPath?: string | null;
  /** Present when the run left `uihtml` reverse-channel callbacks registered.
   *  The host retains this to dispatch later iframe events into the still-live
   *  interpreter (see UihtmlSession). Absent for normal runs (no overhead). */
  uihtmlSession?: UihtmlSession;
}

/** Figure-handle state of a run (see Runtime.currentFigureHandle). */
export interface FigureHandles {
  current: number;
  max: number;
}

// ── Implementation ──────────────────────────────────────────────────────

/**
 * Parse and run `source`. `nativeBridge` is the older way to give the run a
 * koffi bridge; it overrides `host.nativeBridge` for this run.
 */
export function executeCode(
  source: string,
  options: ExecOptions = {},
  workspaceFiles?: WorkspaceFile[],
  mainFileName: string = "script.m",
  searchPaths?: string[],
  /** @deprecated Set `nativeBridge` on the host (`initNodePlatform` does). */
  nativeBridge?: NativeBridge
): ExecResult {
  const host = runHost(options.host ?? defaultHost(), nativeBridge);
  options = {
    ...options,
    host,
    fileIO: options.fileIO ?? host.fileIO,
    system: options.system ?? host.system,
  };
  // ── 1. Parse main file ──────────────────────────────────────────────
  const ast = parseMFile(source, mainFileName);

  if (options.log) {
    options.log("AST parsed, starting interpretation");
  }

  if (mainFileName === "repl") {
    const fn = ast.body.find(s => s.type === "Function");
    if (fn) {
      throw new RuntimeError(
        "Function definitions are not supported in the REPL. Save the function to a .m file instead.",
        fn.span
      );
    }
  }

  // ── 2. Index the workspace for function resolution ──────────────────
  const ws = new Workspace(
    mainFileName,
    source,
    options.fileIO && fileIOView(options.fileIO),
    host
  );
  ws.load(
    workspaceFiles ?? [],
    searchPaths ?? [],
    options.implicitCwdPath,
    options.system,
    [
      ...stdlibFiles,
      ...shimFiles.map(f => ({
        name: `${SHIM_SEARCH_PATH}/${f.name}`,
        source: f.source,
      })),
    ]
  );
  ws.code.setMainFile(ast);
  ws.register();
  ws.code.buildFunctionIndex();

  // ── 3. Create runtime ──────────────────────────────────────────────
  const session = options.session ?? new SessionState();
  const rt = new Runtime(options, session, host);
  const baseEnv = session.baseEnv;
  baseEnv.rt = rt;
  if (options.initialVariableValues) {
    for (const [name, value] of Object.entries(options.initialVariableValues))
      baseEnv.set(name, value);
  }

  // Persistents of functions whose file was edited since the last run are
  // cleared, as MATLAB does when it reloads a changed file.
  const sourceOf = (file: string): string | undefined =>
    file === mainFileName
      ? source
      : ws.mFiles.find(f => f.name === file)?.source;
  session.syncPersistents(sourceOf);

  // Custom builtins take priority over registered builtins.
  if (options.customBuiltins) {
    Object.assign(rt.customBuiltins, options.customBuiltins);
  }

  // ── 4. Create interpreter and wire up runtime callbacks ────────────
  const interpreter = new Interpreter(rt, ws.code, baseEnv);
  interpreter.baseWorkspacePersists = options.session !== undefined;

  // Populate file→source map for line number computation in error messages
  interpreter.fileSources.set(mainFileName, source);
  for (const f of ws.mFiles) {
    interpreter.fileSources.set(f.name, f.source);
  }
  ws.onFileAdded = f => interpreter.fileSources.set(f.name, f.source);
  ws.onFileRemoved = name => interpreter.fileSources.delete(name);

  interpreter.optimization = options.optimization ?? "1";
  interpreter.log = options.log;

  // Register mode-specific executor plugins. The AST interpreter is
  // the dispatcher's hardcoded fallback (not a registered executor).
  registerExecutorsForOpt(interpreter.registry, interpreter.optimization, host);

  // Collect JIT compilations for generatedJS/generatedC output and
  // profiling. JS-JIT and C-JIT sections are kept apart so the CLI's
  // `--dump-js` and `--dump-c` flags route to the right file — at
  // `--opt 2` both backends can fire in a single run.
  const jsSections: string[] = [];
  const cSections: string[] = [];
  interpreter.onJitCompile = (
    description: string,
    code: string,
    lang: "js" | "c"
  ) => {
    const banner = `// ${"=".repeat(60)}\n// JIT: ${description}\n// ${"=".repeat(60)}\n\n${code}`;
    (lang === "c" ? cSections : jsSections).push(banner);
    options.onJitCompile?.(description, code, lang);
  };
  const buildDump = (sections: string[], label: "JS" | "C"): string =>
    sections.length > 0
      ? `// Interpreter mode — JIT compiled sections:\n\n${sections.join("\n\n")}`
      : `// No ${label} generated`;
  if (options.onJitBail) interpreter.onJitBail = options.onJitBail;

  // Wire up the runtime hooks so dispatch by name routes through the interpreter
  interpreter.installRuntimeCallbacks();

  // Expose search paths reference on the runtime (for path() builtin)
  rt.searchPaths = ws.code.searchPaths;

  const applyWorkspace = () => {
    ws.rebuild();
    interpreter.clearAllCaches();
  };
  rt.onPathChange = (action, dir, position) => {
    if (action === "add") {
      if (ws.addPath(dir, position)) applyWorkspace();
    } else {
      ws.removePath(dir);
      applyWorkspace();
    }
  };
  rt.onCwdChange = newCwd => {
    if (ws.changeCwd(newCwd)) applyWorkspace();
  };

  // ── 5. Run ─────────────────────────────────────────────────────────
  pushCurrentRuntime(rt);
  try {
    const execStart = performance.now();
    interpreter.run(ast);
    const executionTimeMs = performance.now() - execStart;

    const result: ExecResult = {
      output: rt.outputLines,
      generatedJS: buildDump(jsSections, "JS"),
      generatedC: buildDump(cSections, "C"),
      plotInstructions: rt.plotInstructions,
      returnValue: interpreter.ans ?? RTV.num(0),
      variableValues: interpreter.getVariableValues(),
      holdState: rt.holdState,
      figureHandles: {
        current: rt.currentFigureHandle,
        max: rt.maxFigureHandle,
      },
    };

    // If the run registered uihtml reverse-channel callbacks, hand the caller a
    // session over this still-live runtime.
    if (rt.uihtmlCallbacks.size > 0) {
      result.uihtmlSession = createUihtmlSession(rt, options.onDrawnow);
    }

    if (options.profile) {
      result.profileData = {
        executionTimeMs,
        jitCompileTimeMs: rt.getJitCompileTimeMs(),
        builtins: rt.getBuiltinProfile(),
        dispatches: rt.getDispatchProfile(),
        hotLoops: [...rt.hotLoops.values()],
      };
    }

    // Propagate path changes so callers (e.g. REPL) can persist them
    if (ws.modified) {
      result.searchPaths = ws.explicitSearchPaths();
      result.workspaceFiles = ws.userFiles();
    }
    // Always propagate the implicit cwd so REPL callers can persist it
    // across commands (cd inside one command should affect the next).
    result.implicitCwdPath = ws.implicitCwd;

    return result;
  } catch (e) {
    // Cooperative cancellation is not an error: propagate it unchanged so
    // callers can detect it (wrapping it in a RuntimeError below would hide
    // it behind a bogus "Runtime error at line N").
    if (e instanceof CancellationError) {
      throw e;
    }
    // Attach collected JIT code to the error so callers (e.g. --dump-js /
    // --dump-c) can inspect it.
    const generatedJS = buildDump(jsSections, "JS");
    const generatedC = buildDump(cSections, "C");
    type WithGen = { generatedJS?: string; generatedC?: string };
    if (e instanceof RuntimeError) {
      // Annotate with file/line info
      if (e.line === null && rt.$file && rt.$line > 0) {
        e.file = rt.$file;
        e.line = rt.$line;
      }
      if (!e.fileSources) e.fileSources = interpreter.fileSources;
      (e as RuntimeError & WithGen).generatedJS = generatedJS;
      (e as RuntimeError & WithGen).generatedC = generatedC;
      throw e;
    }
    const re = new RuntimeError(e instanceof Error ? e.message : String(e));
    if (rt.$file && rt.$line > 0) {
      re.file = rt.$file;
      re.line = rt.$line;
    }
    re.fileSources = interpreter.fileSources;
    (re as RuntimeError & WithGen).generatedJS = generatedJS;
    (re as RuntimeError & WithGen).generatedC = generatedC;
    throw re;
  } finally {
    popCurrentRuntime(rt);
    session.syncPersistents(sourceOf);
    // Files left open stay open (a REPL may keep using them), but their
    // contents reach the host now.
    rt.files?.flush();
  }
}

/** `host`, or a copy of it with a different koffi bridge (the deprecated
 *  `nativeBridge` argument). The copy shares everything else. */
function runHost(host: NumblHost, nativeBridge?: NativeBridge): NumblHost {
  if (!nativeBridge || nativeBridge === host.nativeBridge) return host;
  return { ...host, nativeBridge };
}

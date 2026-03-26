/**
 * Entry point for interpreted execution.
 *
 * Parallel to executeCode.ts but walks the AST directly
 * instead of lowering → codegen → eval.
 *
 * Reuses the LoweringContext for workspace registration, class extraction,
 * and function index building — same resolution logic as the compiler,
 * but dispatches to the interpreter instead of codegen.
 */

import { type ExecOptions, type ExecResult } from "./executor/types.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { Runtime } from "./runtime/runtime.js";
import { RTV } from "./runtime/constructors.js";
import { RuntimeError } from "../numbl-core/runtime/index.js";
import { loadJsUserFunctions } from "./jsUserFunctions.js";
import {
  registerDynamicIBuiltin,
  unregisterIBuiltin,
} from "./interpreter/builtins/types.js";
import type { NativeBridge } from "./workspace/index.js";
import { parseMFile, type Stmt } from "./parser/index.js";
import { SyntaxError } from "./parser/errors.js";
import { Interpreter } from "./interpreter/interpreter.js";
import { LoweringContext } from "./lowering/loweringContext.js";
import { stdlibFiles, shimFiles } from "./stdlib-bundle.js";
import { jitHelpers } from "./interpreter/jit/jitHelpers.js";

/** Virtual search path prefix for bundled shim files. */
const SHIM_SEARCH_PATH = "__numbl_shims__";

export function interpretCode(
  source: string,
  options: ExecOptions = {},
  workspaceFiles?: WorkspaceFile[],
  mainFileName: string = "script.m",
  searchPaths?: string[],
  nativeBridge?: NativeBridge
): ExecResult {
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

  // Add stdlib and shim files
  for (const f of stdlibFiles) {
    mWorkspaceFiles.push(f);
  }
  for (const f of shimFiles) {
    mWorkspaceFiles.push({
      name: `${SHIM_SEARCH_PATH}/${f.name}`,
      source: f.source,
    });
  }
  ctx.registry.searchPaths = [...(searchPaths ?? []), SHIM_SEARCH_PATH];

  // Pre-parse all .m workspace files into the shared AST cache
  ctx.fileASTCache.set(mainFileName, ast);
  for (const f of mWorkspaceFiles) {
    try {
      ctx.fileASTCache.set(f.name, parseMFile(f.source, f.name));
    } catch (e) {
      if (e instanceof SyntaxError && e.file === null) {
        e.file = f.name;
      }
      throw e;
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

  // Register .js user functions as IBuiltins
  for (const ib of jsUserFunctions) {
    registerDynamicIBuiltin(ib);
  }

  // Apply custom builtins
  if (options.customBuiltins) {
    Object.assign(rt.builtins, options.customBuiltins);
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

  interpreter.optimization = options.optimization ?? 0;

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

  // Wire up eval callback
  rt.evalLocalCallback = (code, initialVars, onOutput) => {
    const evalResult = interpretCode(code, {
      onOutput,
      displayResults: false,
      initialVariableValues: initialVars,
    });
    return {
      returnValue: evalResult.returnValue,
      variableValues: evalResult.variableValues,
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
        codegenTimeMs: 0,
        codegenBreakdown: {
          parseMainMs: 0,
          parseWorkspaceMs: 0,
          loadJsUserFunctionsMs: 0,
          registrationMs: 0,
          buildFunctionIndexMs: 0,
          lowerMainMs: 0,
          codegenMs: 0,
        },
        executionTimeMs,
        jitCompileTimeMs: rt.getJitCompileTimeMs(),
        builtins: rt.getBuiltinProfile(),
        dispatches: rt.getDispatchProfile(),
      };
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
    // Unregister .js user function IBuiltins to avoid polluting the global registry
    for (const ib of jsUserFunctions) {
      unregisterIBuiltin(ib.name);
    }
  }
}

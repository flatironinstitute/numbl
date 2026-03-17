import { AbstractSyntaxTree, parseMFile, type Stmt } from "../parser/index.js";
import { SyntaxError } from "../parser/errors.js";
import type { WorkspaceFile, NativeBridge } from "../workspace/index.js";
import { offsetToLine } from "../runtime/index.js";
import { SemanticError } from "../lowering/errors.js";
import {
  LoweringContext,
  type FileASTCache,
  type FunctionIndex,
} from "../lowering/loweringContext.js";
import { lowerStmts } from "../lowering/lowerStmt.js";
import { Codegen } from "./codegen.js";
import { collectVarIds } from "../lowering/varIdCollect.js";
import { typeToString } from "../lowering/itemTypes.js";
import type { IRVariable } from "../lowering/loweringTypes.js";
import {
  register,
  unregister,
  getBuiltin,
  type BuiltinFn,
} from "../builtins/registry.js";
import { loadJsUserFunctions } from "../jsUserFunctions.js";
import { stdlibFiles, shimFiles } from "../stdlib-bundle.js";

/** Virtual search path prefix for bundled shim files. */
const SHIM_SEARCH_PATH = "__numbl_shims__";

export interface CodegenBreakdown {
  parseMainMs: number;
  parseWorkspaceMs: number;
  loadJsUserFunctionsMs: number;
  registrationMs: number;
  buildFunctionIndexMs: number;
  lowerMainMs: number;
  codegenMs: number;
}

export function generateMainScriptCode(
  source: string,
  mainFileName: string = "script.m",
  workspaceFiles?: WorkspaceFile[],
  initialVariableNames?: string[],
  searchPaths?: string[],
  opts?: { noLineTracking?: boolean; nativeBridge?: NativeBridge }
): {
  jsCode: string;
  ctx: LoweringContext;
  fileSources: Map<string, string>;
  ast: AbstractSyntaxTree;
  functionIndex: FunctionIndex;
  fileASTCache: FileASTCache;
  jsUserFunctions: Map<string, BuiltinFn>;
  codegenBreakdown: CodegenBreakdown;
} {
  // ── 1. Parse ────────────────────────────────────────────────────────
  let _t0 = performance.now();
  const ast = parseMFile(source, mainFileName);

  // Separate local functions and class definitions from main body statements
  const localFunctions: (Stmt & { type: "Function" })[] = [];
  const localClasses: (Stmt & { type: "ClassDef" })[] = [];
  const mainStmts: Stmt[] = [];
  for (const stmt of ast.body) {
    if (stmt.type === "Function") {
      localFunctions.push(stmt);
    } else if (stmt.type === "ClassDef") {
      localClasses.push(stmt);
    } else {
      mainStmts.push(stmt);
    }
  }

  // Validate classdef constraints: at most one classdef, no mixing with main body
  if (localClasses.length > 1) {
    throw new SemanticError("Only one classdef per file is allowed");
  }
  if (localClasses.length === 1 && mainStmts.length > 0) {
    throw new SemanticError(
      "A classdef file cannot contain top-level statements"
    );
  }

  const _parseMainMs = performance.now() - _t0;

  // ── 2. Set up lowering context ──────────────────────────────────────
  // source and mainFileName are stored for error reporting and workspace class detection
  const ctx = new LoweringContext(source, mainFileName);
  // Separate .js user function files, .wasm files, and .m workspace files
  const mWorkspaceFiles: WorkspaceFile[] = [];
  const jsWorkspaceFiles: WorkspaceFile[] = [];
  const wasmWorkspaceFiles: WorkspaceFile[] = [];
  if (workspaceFiles) {
    for (const f of workspaceFiles) {
      if (f.name.endsWith(".js")) {
        jsWorkspaceFiles.push(f);
      } else if (f.name.endsWith(".wasm")) {
        wasmWorkspaceFiles.push(f);
      } else {
        mWorkspaceFiles.push(f);
      }
    }
  }

  // Add stdlib files (bundled .m implementations of builtins like inputParser)
  for (const f of stdlibFiles) {
    mWorkspaceFiles.push(f);
  }

  // Add shim files (bundled .m shims for internal classes/functions)
  // Shim file names are relative paths (e.g. "+matlab/+internal/+decomposition/DenseLU.m"),
  // so we prefix them with a virtual search path to enable package resolution.
  for (const f of shimFiles) {
    mWorkspaceFiles.push({
      name: `${SHIM_SEARCH_PATH}/${f.name}`,
      source: f.source,
    });
  }
  ctx.registry.searchPaths = [...(searchPaths ?? []), SHIM_SEARCH_PATH];

  // Load .js user functions
  _t0 = performance.now();
  const jsUserFunctions = loadJsUserFunctions(
    jsWorkspaceFiles,
    wasmWorkspaceFiles,
    opts?.nativeBridge
  );
  const _loadJsUserFunctionsMs = performance.now() - _t0;

  // Pre-parse all .m workspace files into the shared AST cache (each file parsed exactly once)
  _t0 = performance.now();
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
  const _parseWorkspaceMs = performance.now() - _t0;

  // Register local functions (available for resolution, lowered on demand)
  _t0 = performance.now();
  for (const fn of localFunctions) {
    ctx.registerLocalFunctionAST(fn);
  }

  // Register local classes (available for resolution)
  for (const cls of localClasses) {
    ctx.registerLocalClass(cls);
  }

  // Register .m workspace files for on-demand resolution
  if (mWorkspaceFiles.length > 0) {
    ctx.registerWorkspaceFiles(mWorkspaceFiles);
  }

  // ── 2b. Build function index ───────────────────────────────────────
  // Register .js user functions as builtins so they appear in the index
  // and are available during lowering (for type inference via check()).
  // They are unregistered at the end to avoid polluting the global singleton.
  const jsUserFuncNames = [...jsUserFunctions.keys()];
  const savedBuiltins = new Map<string, BuiltinFn>();
  for (const name of jsUserFuncNames) {
    const existing = getBuiltin(name);
    if (existing) savedBuiltins.set(name, existing);
    register(name, jsUserFunctions.get(name)!);
  }
  const _registrationMs = performance.now() - _t0;

  // Built after workspace/local function registration, before lowering so that
  // the resolver is available during both lowering and code generation.
  _t0 = performance.now();
  const functionIndex = ctx.buildFunctionIndex(jsUserFuncNames);
  const _buildFunctionIndexMs = performance.now() - _t0;

  // ── 3. Lower main body ─────────────────────────────────────────────
  // isTopLevel tags variables so only script-scope vars are exported to the workspace
  ctx.isTopLevel = true;

  // Pre-define initial variables so they're visible during lowering
  // (used for REPL persistence: variables from previous executions)
  const initialVarIRVars: IRVariable[] = [];
  if (initialVariableNames) {
    for (const name of initialVariableNames) {
      initialVarIRVars.push(ctx.defineVariable(name, undefined));
    }
  }

  _t0 = performance.now();
  const irBody = lowerStmts(ctx, mainStmts);
  ctx.isTopLevel = false;
  const _lowerMainMs = performance.now() - _t0;

  // Auto-invoke: if no top-level code and no classdef, call the first function
  const hasTopLevelCode = irBody.length > 0;
  let autoInvokeFunction: (Stmt & { type: "Function" }) | null = null;
  if (
    !hasTopLevelCode &&
    localClasses.length === 0 &&
    localFunctions.length > 0
  ) {
    autoInvokeFunction = localFunctions.reduce((a, b) =>
      a.span.start < b.span.start ? a : b
    );
  }

  // ── 4. Codegen ─────────────────────────────────────────────────────
  _t0 = performance.now();
  // fileSources maps filename → source text; used by codegen for error spans
  // and by the runtime for annotating errors with file/line info
  const fileSources = new Map<string, string>();
  fileSources.set(mainFileName, source);
  for (const f of mWorkspaceFiles) {
    fileSources.set(f.name, f.source);
  }

  const codegen = new Codegen(ctx, fileSources);
  if (opts?.noLineTracking) {
    codegen.noLineTracking = true;
  }

  // Collect script-level VarIds
  const scriptVarIds = new Set<string>();
  collectVarIds(irBody, scriptVarIds);
  // Include initial variable VarIds so they get declared and exported
  for (const v of initialVarIRVars) {
    scriptVarIds.add(v.id.id);
  }

  // Declare script-level variables
  codegen.emit(`var $ret;`);
  const varById = new Map(ctx.allVariables.map(v => [v.id.id, v]));
  const sortedVarIds = [...scriptVarIds].sort();
  for (const id of sortedVarIds) {
    const v = varById.get(id);
    const vTy = v ? codegen.typeEnv.get(v.id) : undefined;
    const tc = vTy ? ` /* ${typeToString(vTy)} */` : "";
    codegen.emit(`var ${codegen.varRef(id)};${tc}`);
  }

  // Initialize variables from initialVariableValues
  for (const v of initialVarIRVars) {
    codegen.emit(
      `${codegen.varRef(v.id.id)} = $rt.getInitialVariableValue(${JSON.stringify(v.name)});`
    );
  }

  // Generate main body statements
  codegen.genStmts(irBody);

  // Auto-invoke: if file has only function defs and no main body, call the first one
  if (autoInvokeFunction) {
    // Lower+codegen the local function specialized for zero args
    const jsId = codegen.ensureSpecializedFunctionGenerated(
      autoInvokeFunction.name,
      []
    );
    const invokeLine = offsetToLine(source, autoInvokeFunction.span.start);
    if (!codegen.noLineTracking) {
      codegen.emit(`$rt.$file = ${JSON.stringify(mainFileName)};`);
      codegen.emit(`$rt.$line = ${invokeLine};`);
    }
    if (jsId) {
      codegen.emit(`${jsId}(0); /* 0 = nargout */`);
    } else {
      codegen.emit(
        `throw $rt.error("Failed to auto-invoke function ${autoInvokeFunction.name}");`
      );
    }
  }

  // Export top-level variable values to the runtime workspace
  // (used by REPL and web UI to inspect state after execution)
  for (const variable of ctx.allVariables) {
    if (variable.isTopLevel && scriptVarIds.has(variable.id.id)) {
      codegen.emit(
        `$rt.setVariableValue(${JSON.stringify(variable.name)}, ${codegen.varRef(variable.id.id)});`
      );
    }
  }

  // $ret holds the last expression statement's value (set by ExprStmt codegen)
  codegen.emit(`return $ret;`);

  const _codegenMs = performance.now() - _t0;

  // Restore original builtins that were temporarily overwritten by .js user functions
  // (the js user functions are registered on rt.builtins at runtime in executeCode)
  for (const name of jsUserFuncNames) {
    const original = savedBuiltins.get(name);
    if (original) {
      register(name, original);
    } else {
      unregister(name);
    }
  }

  return {
    jsCode: codegen.getCode(),
    ctx,
    fileSources,
    ast,
    functionIndex,
    fileASTCache: ctx.fileASTCache,
    jsUserFunctions,
    codegenBreakdown: {
      parseMainMs: _parseMainMs,
      parseWorkspaceMs: _parseWorkspaceMs,
      loadJsUserFunctionsMs: _loadJsUserFunctionsMs,
      registrationMs: _registrationMs,
      buildFunctionIndexMs: _buildFunctionIndexMs,
      lowerMainMs: _lowerMainMs,
      codegenMs: _codegenMs,
    },
  };
}

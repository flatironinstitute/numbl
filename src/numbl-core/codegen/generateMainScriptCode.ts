import {
  AbstractSyntaxTree,
  parseMFile,
  type Stmt,
  type Expr,
} from "../parser/index.js";
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
import type { IRStmt, IRExpr } from "../lowering/nodes.js";
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

  // ── 2c. Scan ASTs for assignin/evalin usage (before lowering) ────────
  // This must happen before lowering so the type inference can reset
  // types of externally-settable variables after function calls.
  const { workspaceNames: workspaceAccessedNames, callerAccessMap } =
    collectAccessedNames(ast, ctx.fileASTCache);
  ctx.workspaceAccessedVarNames = workspaceAccessedNames;
  ctx.callerAccessMap = callerAccessMap;

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

  // Register workspace accessors only for variables referenced by assignin/evalin
  for (const id of sortedVarIds) {
    const v = varById.get(id);
    if (v && ctx.workspaceAccessedVarNames.has(v.name)) {
      const jsRef = codegen.varRef(id);
      codegen.emit(
        `$rt.setWorkspaceAccessor(${JSON.stringify(v.name)}, () => ${jsRef}, ($v) => { ${jsRef} = $v; });`
      );
    }
  }

  // Push caller accessors for the main script so functions called from
  // top level can use evalin/assignin('caller', ...) to access script vars
  if (ctx.callerAccessMap.size > 0) {
    const callerVars = computeMainScriptCallerVars(
      ctx.callerAccessMap,
      irBody,
      sortedVarIds,
      varById,
      codegen
    );
    if (callerVars.length > 0) {
      const entries = callerVars
        .map(({ name, jsRef }) => {
          return `${JSON.stringify(name)}: [() => ${jsRef}, ($v) => { ${jsRef} = $v; }]`;
        })
        .join(", ");
      codegen.emit(`$rt.pushCallerAccessors({${entries}});`);
    } else {
      codegen.emit(`$rt.pushCallerAccessors(null);`);
    }
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

  // Pop caller accessors if we pushed them
  if (ctx.callerAccessMap.size > 0) {
    codegen.emit(`$rt.popCallerAccessors();`);
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

/**
 * Determine which main-script variables need caller accessors.
 * Walks the main script IR body for function call names, cross-references
 * with callerAccessMap, and returns matching script-level variables.
 */
function computeMainScriptCallerVars(
  callerAccessMap: Map<string, Set<string>>,
  irBody: IRStmt[],
  sortedVarIds: string[],
  varById: Map<string, IRVariable>,
  codegen: Codegen
): Array<{ name: string; jsRef: string }> {
  // Collect function call names from the main script IR body
  const callNames = new Set<string>();
  collectIRCallNames(irBody, callNames);

  // Find which variable names are needed
  const neededVarNames = new Set<string>();
  for (const callName of callNames) {
    const baseName = callName.includes(".")
      ? callName.slice(callName.lastIndexOf(".") + 1)
      : callName;
    const vars = callerAccessMap.get(baseName);
    if (vars) {
      for (const v of vars) neededVarNames.add(v);
    }
  }

  if (neededVarNames.size === 0) return [];

  // Intersect with script-level variables
  const result: Array<{ name: string; jsRef: string }> = [];
  for (const id of sortedVarIds) {
    const v = varById.get(id);
    if (v && neededVarNames.has(v.name)) {
      result.push({ name: v.name, jsRef: codegen.varRef(id) });
    }
  }
  return result;
}

/** Recursively collect function call names from IR statements. */
function collectIRCallNames(stmts: IRStmt[], out: Set<string>): void {
  for (const s of stmts) {
    if (s.type === "Function") continue;
    collectIRCallNamesFromStmt(s, out);
  }
}

function collectIRCallNamesFromStmt(stmt: IRStmt, out: Set<string>): void {
  switch (stmt.type) {
    case "ExprStmt":
      collectIRCallNamesFromExpr(stmt.expr, out);
      break;
    case "Assign":
      collectIRCallNamesFromExpr(stmt.expr, out);
      break;
    case "MultiAssign":
      collectIRCallNamesFromExpr(stmt.expr, out);
      break;
    case "AssignLValue":
      collectIRCallNamesFromExpr(stmt.expr, out);
      break;
    case "If":
      collectIRCallNamesFromExpr(stmt.cond, out);
      collectIRCallNames(stmt.thenBody, out);
      for (const c of stmt.elseifBlocks) {
        collectIRCallNamesFromExpr(c.cond, out);
        collectIRCallNames(c.body, out);
      }
      if (stmt.elseBody) collectIRCallNames(stmt.elseBody, out);
      break;
    case "For":
      collectIRCallNamesFromExpr(stmt.expr, out);
      collectIRCallNames(stmt.body, out);
      break;
    case "While":
      collectIRCallNamesFromExpr(stmt.cond, out);
      collectIRCallNames(stmt.body, out);
      break;
    case "Switch":
      collectIRCallNamesFromExpr(stmt.expr, out);
      for (const c of stmt.cases) {
        collectIRCallNamesFromExpr(c.value, out);
        collectIRCallNames(c.body, out);
      }
      if (stmt.otherwise) collectIRCallNames(stmt.otherwise, out);
      break;
    case "TryCatch":
      collectIRCallNames(stmt.tryBody, out);
      collectIRCallNames(stmt.catchBody, out);
      break;
    default:
      break;
  }
}

function collectIRCallNamesFromExpr(expr: IRExpr, out: Set<string>): void {
  const k = expr.kind;
  switch (k.type) {
    case "FuncCall":
      out.add(k.name);
      for (const a of k.args) collectIRCallNamesFromExpr(a, out);
      if (k.instanceBase) collectIRCallNamesFromExpr(k.instanceBase, out);
      break;
    case "MethodCall":
      out.add(k.name);
      collectIRCallNamesFromExpr(k.base, out);
      for (const a of k.args) collectIRCallNamesFromExpr(a, out);
      break;
    case "Binary":
      collectIRCallNamesFromExpr(k.left, out);
      collectIRCallNamesFromExpr(k.right, out);
      break;
    case "Unary":
      collectIRCallNamesFromExpr(k.operand, out);
      break;
    case "Range":
      collectIRCallNamesFromExpr(k.start, out);
      if (k.step) collectIRCallNamesFromExpr(k.step, out);
      collectIRCallNamesFromExpr(k.end, out);
      break;
    case "Index":
    case "IndexCell":
      collectIRCallNamesFromExpr(k.base, out);
      for (const i of k.indices) collectIRCallNamesFromExpr(i, out);
      break;
    case "Member":
      collectIRCallNamesFromExpr(k.base, out);
      break;
    case "MemberDynamic":
      collectIRCallNamesFromExpr(k.base, out);
      collectIRCallNamesFromExpr(k.nameExpr, out);
      break;
    case "SuperConstructorCall":
      for (const a of k.args) collectIRCallNamesFromExpr(a, out);
      break;
    case "AnonFunc":
      collectIRCallNamesFromExpr(k.body, out);
      break;
    case "Tensor":
    case "Cell":
      for (const row of k.rows)
        for (const e of row) collectIRCallNamesFromExpr(e, out);
      break;
    case "ClassInstantiation":
      for (const a of k.args) collectIRCallNamesFromExpr(a, out);
      break;
    default:
      break;
  }
}

/**
 * Scan all ASTs for assignin/evalin calls, returning:
 * - workspaceNames: variable names accessed with 'workspace' scope
 * - callerAccessMap: function base name → set of variable names accessed with 'caller' scope
 */
function collectAccessedNames(
  mainAst: AbstractSyntaxTree,
  fileASTCache: FileASTCache
): {
  workspaceNames: Set<string>;
  callerAccessMap: Map<string, Set<string>>;
} {
  const workspaceNames = new Set<string>();
  // Maps function base name → set of var names it accesses via caller
  const callerAccessMap = new Map<string, Set<string>>();

  // Track the enclosing function name stack
  const funcNameStack: string[] = [];

  function getStringLiteralValue(expr: Expr): string | null {
    if (expr.type === "Char" || expr.type === "String")
      return expr.value.replace(/^['"]|['"]$/g, "");
    return null;
  }

  function baseName(name: string): string {
    const dot = name.lastIndexOf(".");
    return dot >= 0 ? name.slice(dot + 1) : name;
  }

  function walkExpr(expr: Expr): void {
    switch (expr.type) {
      case "FuncCall":
        if (
          (expr.name === "assignin" || expr.name === "evalin") &&
          expr.args.length >= 2
        ) {
          const scope = getStringLiteralValue(expr.args[0]);
          const varName = getStringLiteralValue(expr.args[1]);
          if (varName) {
            if (scope === "workspace") {
              workspaceNames.add(varName);
            } else if (scope === "caller") {
              const enclosingFunc =
                funcNameStack.length > 0
                  ? funcNameStack[funcNameStack.length - 1]
                  : null;
              if (enclosingFunc) {
                let vars = callerAccessMap.get(enclosingFunc);
                if (!vars) {
                  vars = new Set();
                  callerAccessMap.set(enclosingFunc, vars);
                }
                vars.add(varName);
              }
            }
          }
        }
        for (const a of expr.args) walkExpr(a);
        break;
      case "Binary":
        walkExpr(expr.left);
        walkExpr(expr.right);
        break;
      case "Unary":
        walkExpr(expr.operand);
        break;
      case "Range":
        walkExpr(expr.start);
        if (expr.step) walkExpr(expr.step);
        walkExpr(expr.end);
        break;
      case "Index":
      case "IndexCell":
        walkExpr(expr.base);
        for (const i of expr.indices) walkExpr(i);
        break;
      case "Member":
        walkExpr(expr.base);
        break;
      case "MemberDynamic":
        walkExpr(expr.base);
        walkExpr(expr.nameExpr);
        break;
      case "MethodCall":
        walkExpr(expr.base);
        for (const a of expr.args) walkExpr(a);
        break;
      case "SuperMethodCall":
        for (const a of expr.args) walkExpr(a);
        break;
      case "AnonFunc":
        walkExpr(expr.body);
        break;
      case "Tensor":
      case "Cell":
        for (const row of expr.rows) for (const e of row) walkExpr(e);
        break;
      case "ClassInstantiation":
        for (const a of expr.args) walkExpr(a);
        break;
      default:
        break;
    }
  }

  function walkStmt(stmt: Stmt): void {
    switch (stmt.type) {
      case "ExprStmt":
        walkExpr(stmt.expr);
        break;
      case "Assign":
        walkExpr(stmt.expr);
        break;
      case "MultiAssign":
        walkExpr(stmt.expr);
        break;
      case "AssignLValue":
        walkExpr(stmt.expr);
        break;
      case "If":
        walkExpr(stmt.cond);
        stmt.thenBody.forEach(walkStmt);
        stmt.elseifBlocks.forEach(c => {
          walkExpr(c.cond);
          c.body.forEach(walkStmt);
        });
        stmt.elseBody?.forEach(walkStmt);
        break;
      case "For":
        walkExpr(stmt.expr);
        stmt.body.forEach(walkStmt);
        break;
      case "While":
        walkExpr(stmt.cond);
        stmt.body.forEach(walkStmt);
        break;
      case "Switch":
        walkExpr(stmt.expr);
        stmt.cases.forEach(c => {
          walkExpr(c.value);
          c.body.forEach(walkStmt);
        });
        stmt.otherwise?.forEach(walkStmt);
        break;
      case "TryCatch":
        stmt.tryBody.forEach(walkStmt);
        stmt.catchBody.forEach(walkStmt);
        break;
      case "Function":
        funcNameStack.push(baseName(stmt.name));
        stmt.body.forEach(walkStmt);
        funcNameStack.pop();
        break;
      case "Return":
      case "Break":
      case "Continue":
      case "Global":
      case "Persistent":
        break;
      case "ClassDef":
        for (const m of stmt.members) {
          if (m.type === "Methods") m.body.forEach(walkStmt);
        }
        break;
      default:
        break;
    }
  }

  // Walk the main AST
  for (const stmt of mainAst.body) walkStmt(stmt);

  // Walk all cached workspace file ASTs
  for (const [, ast] of fileASTCache) {
    for (const stmt of ast.body) walkStmt(stmt);
  }

  return { workspaceNames, callerAccessMap };
}

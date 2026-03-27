/**
 * AST-walking interpreter for MATLAB code.
 *
 * Walks the parsed AST directly without lowering or codegen.
 * All dispatch decisions are made at runtime using actual values.
 *
 * Reuses LoweringContext and FunctionIndex from the lowering pipeline
 * for workspace registration, class extraction, and function resolution.
 *
 * The Interpreter class is defined here with all fields.
 * Methods are split across:
 *   - interpreterExec.ts   (statement + expression evaluation)
 *   - interpreterFunctions.ts (function call, resolution, class methods)
 */

import type { AbstractSyntaxTree } from "../parser/types.js";
import type { Runtime } from "../runtime/runtime.js";
import type { RuntimeValue } from "../runtime/types.js";
import { getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import type {
  LoweringContext,
  FunctionIndex,
} from "../lowering/loweringContext.js";
import { resolveFunction, type ResolvedTarget } from "../functionResolve.js";
import type { ItemType } from "../lowering/itemTypes.js";

import {
  Environment,
  funcDefFromStmt,
  type FunctionDef,
  type ControlSignal,
} from "./types.js";
import type { Stmt, Expr, LValue } from "../parser/types.js";
import type { ClassInfo } from "../lowering/classInfo.js";

// ── Interpreter ──────────────────────────────────────────────────────────

export class Interpreter {
  /** @internal */ env: Environment;
  public ans: RuntimeValue | undefined;

  /** @internal */ ctx: LoweringContext;
  /** @internal */ functionIndex: FunctionIndex;
  /** @internal */ rt: Runtime;

  /** @internal */ currentFile: string;
  /** @internal */ currentClassName: string | undefined;
  /** @internal */ currentMethodName: string | undefined;

  /** @internal */ mainLocalFunctions = new Map<string, FunctionDef>();

  /** @internal The main script (workspace) environment — for evalin/assignin('workspace', ...) */
  workspaceEnv: Environment | undefined;
  /** @internal The caller's environment — for evalin/assignin('caller', ...) */
  callerEnv: Environment | undefined;

  /** @internal Stack of [base, dimIndex, numIndices] for resolving `end` keyword in indexing. */
  endContextStack: Array<{
    base: unknown;
    dimIndex: number;
    numIndices: number;
  }> = [];

  /** @internal */
  functionDefCache = new Map<string, FunctionDef>();

  /** @internal Pre-built line break tables for offset→line lookup per file. */
  lineTableCache = new Map<string, number[]>();

  /** @internal file→source mapping for line number computation */
  fileSources = new Map<string, string>();

  /** @internal Guard against infinite recursion in compileSpecialized */
  compileInProgress = new Set<string>();

  /** @internal Per-instance cache for JIT-compiled loops (avoids cross-execution collisions). */
  loopJitCache = new Map<
    string,
    { fn: (...args: unknown[]) => unknown; source: string } | null
  >();

  /** Optimization level (0 = pure interpreter, >=1 = JIT scalar functions). */
  optimization: number = 1;

  /** Callback for JIT compilation logging. */
  onJitCompile?: (description: string, jsCode: string) => void;

  constructor(
    rt: Runtime,
    ctx: LoweringContext,
    functionIndex: FunctionIndex,
    mainFileName: string,
    initialVariableValues?: Record<string, RuntimeValue>
  ) {
    this.rt = rt;
    this.ctx = ctx;
    this.functionIndex = functionIndex;
    this.currentFile = mainFileName;
    this.env = new Environment();
    this.env.rt = rt;
    if (initialVariableValues) {
      for (const [name, value] of Object.entries(initialVariableValues)) {
        this.env.set(name, value);
      }
    }
  }

  /** Clear all JIT and function resolution caches. Called after addpath/rmpath. */
  clearAllCaches(): void {
    for (const [, fd] of this.functionDefCache) {
      fd._jitCache?.clear();
    }
    this.functionDefCache.clear();
    this.loopJitCache.clear();
    this.compileInProgress.clear();
    this.ctx.registry.fileContexts.clear();
    this.rt.classMethodCache.clear();
  }

  /** Wire up runtime callbacks so dispatch() routes through the interpreter. */
  installRuntimeCallbacks(): void {
    this.rt.compileSpecialized = (
      name: string,
      argTypes: ItemType[],
      callSite: CallSite
    ) => {
      const guardKey = JSON.stringify([name, callSite]);
      if (this.compileInProgress.has(guardKey)) return null;
      this.compileInProgress.add(guardKey);
      try {
        const target = resolveFunction(
          name,
          argTypes,
          callSite,
          this.functionIndex
        );
        if (!target || target.kind === "builtin") return null;
        return (nargout: number, ...args: unknown[]) => {
          return this.interpretTarget(target, args, nargout);
        };
      } finally {
        this.compileInProgress.delete(guardKey);
      }
    };

    this.rt.resolveClassMethod = (className: string, methodName: string) => {
      // Check if the method actually exists before returning a wrapper
      if (
        !this.ctx.classHasMethod(className, methodName) &&
        !this.ctx.classHasStaticMethod(className, methodName)
      ) {
        // Check if it's a constructor
        const classInfo = this.ctx.getClassInfo(className);
        if (!classInfo || classInfo.constructorName !== methodName) {
          // Check external method files
          let found = false;
          let current: string | null = className;
          while (current) {
            const info = this.ctx.getClassInfo(current);
            if (!info) break;
            if (info.externalMethodFiles.has(methodName)) {
              found = true;
              break;
            }
            current = info.superClass;
          }
          if (!found) return null;
        }
      }

      const definingClass = this.ctx.findDefiningClass(className, methodName);
      const isStatic = this.ctx.classHasStaticMethod(definingClass, methodName);
      return (nargout: number, ...args: unknown[]) => {
        const target: ResolvedTarget = {
          kind: "classMethod",
          className: definingClass,
          methodName,
          compileArgTypes: args.map(a =>
            getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
          ),
          stripInstance: false,
        };
        if (isStatic) {
          return this.interpretTarget(target, args.slice(1), nargout);
        }
        return this.interpretTarget(target, args, nargout);
      };
    };

    this.rt.getClassParent = (className: string) =>
      this.ctx.getClassInfo(className)?.superClass ?? null;
  }

  /** Run a complete AST (main script). */
  run(ast: AbstractSyntaxTree): void {
    // Remember the workspace (main script) environment
    this.workspaceEnv = this.env;

    // First pass: collect local function definitions
    for (const stmt of ast.body) {
      if (stmt.type === "Function") {
        this.mainLocalFunctions.set(stmt.name, funcDefFromStmt(stmt));
      }
    }

    // Second pass: execute non-function, non-classdef statements
    const nonFuncStmts = ast.body.filter(
      s => s.type !== "Function" && s.type !== "ClassDef"
    );
    if (nonFuncStmts.length === 0 && this.mainLocalFunctions.size > 0) {
      // Function file: call the first function with 0 args
      const firstFn = this.mainLocalFunctions.values().next().value;
      if (firstFn) {
        this.callUserFunction(firstFn, [], 0);
      }
    } else {
      for (const stmt of nonFuncStmts) {
        const signal = this.execStmt(stmt);
        if (signal) break;
      }
    }
  }

  /** Get variable values for ExecResult. */
  getVariableValues(): Record<string, RuntimeValue> {
    return this.env.toRecord();
  }

  // Methods added by interpreterExec.ts
  declare execStmt: (stmt: Stmt) => ControlSignal | null;
  declare execStmts: (stmts: Stmt[]) => ControlSignal | null;
  declare evalExpr: (expr: Expr) => unknown;
  declare evalExprNargout: (expr: Expr, nargout: number) => unknown;
  declare evalBinary: (expr: Extract<Expr, { type: "Binary" }>) => unknown;
  declare evalUnary: (expr: Extract<Expr, { type: "Unary" }>) => unknown;
  declare evalRange: (expr: Extract<Expr, { type: "Range" }>) => unknown;
  declare evalFuncCall: (
    expr: Extract<Expr, { type: "FuncCall" }>,
    nargout: number
  ) => unknown;
  declare evalIndex: (expr: Extract<Expr, { type: "Index" }>) => unknown;
  declare evalIndexCell: (
    expr: Extract<Expr, { type: "IndexCell" }>
  ) => unknown;
  declare evalIndicesWithEnd: (base: unknown, indexExprs: Expr[]) => unknown[];
  declare evalArgs: (argExprs: Expr[]) => unknown[];
  declare evalMember: (
    expr: Extract<Expr, { type: "Member" }>,
    nargout: number
  ) => unknown;
  declare tryExtractDottedName: (expr: Expr) => string | null;
  declare evalAnonFunc: (
    expr: Extract<Expr, { type: "AnonFunc" }>
  ) => RuntimeValue;
  declare makeFuncHandle: (name: string) => RuntimeValue;
  declare evalTensorLiteral: (
    expr: Extract<Expr, { type: "Tensor" }>
  ) => unknown;
  declare evalCellLiteral: (expr: Extract<Expr, { type: "Cell" }>) => unknown;
  declare assignLValue: (lv: LValue, value: RuntimeValue) => void;
  declare writeLValueBase: (base: Expr, value: RuntimeValue) => void;
  declare evalLValueBase: (base: Expr, defaultVal: RuntimeValue) => unknown;
  declare switchMatch: (switchVal: unknown, caseVal: unknown) => boolean;
  declare isOutputExpr: (expr: Expr) => boolean;

  // Methods added by interpreterFunctions.ts
  declare callFunction: (
    name: string,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare interpretTarget: (
    target: ResolvedTarget,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare interpretLocalFunction: (
    target: Extract<ResolvedTarget, { kind: "localFunction" }>,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare interpretWorkspaceFunction: (
    target: Extract<ResolvedTarget, { kind: "workspaceFunction" }>,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare interpretClassMethod: (
    target: Extract<ResolvedTarget, { kind: "classMethod" }>,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare interpretPrivateFunction: (
    target: Extract<ResolvedTarget, { kind: "privateFunction" }>,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare instantiateClass: (
    className: string,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare interpretConstructor: (
    classInfo: ClassInfo,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare callUserFunction: (
    fn: FunctionDef,
    args: unknown[],
    nargout: number,
    narginOverride?: number
  ) => unknown;
  declare callNestedFunction: (
    fn: FunctionDef,
    parentEnv: Environment,
    args: unknown[],
    nargout: number
  ) => unknown;
  declare withFileContext: <T>(
    file: string,
    className: string | undefined,
    methodName: string | undefined,
    fn: () => T
  ) => T;
  declare getWorkspaceFileName: (funcName: string) => string;
  declare getClassFileName: (className: string) => string;
  declare findFunctionInWorkspaceFile: (
    wsName: string,
    funcName: string
  ) => FunctionDef | null;
  declare findFunctionInClassFile: (
    className: string,
    funcName: string,
    methodScope?: string
  ) => FunctionDef | null;
  declare findMethodInClass: (
    classInfo: ClassInfo,
    methodName: string
  ) => FunctionDef | null;
  declare findExternalMethod: (
    classInfo: ClassInfo,
    methodName: string
  ) => FunctionDef | null;
  declare collectClassProperties: (classInfo: ClassInfo) => {
    propertyNames: string[];
    propertyDefaults: Map<string, Expr>;
  };
  declare isHandleClass: (classInfo: ClassInfo) => boolean;
  declare evalInLocalScope: (codeArg: unknown, fileName?: string) => unknown;
  declare processArgumentsBlocks: (
    fn: FunctionDef,
    args: unknown[]
  ) => unknown[];
}

// ── Prototype augmentation ───────────────────────────────────────────────
// Import method implementations from split files and assign to prototype.

import * as Exec from "./interpreterExec.js";
import * as Funcs from "./interpreterFunctions.js";

const p = Interpreter.prototype as unknown as Record<string, unknown>;
for (const mod of [Exec, Funcs]) {
  for (const [key, val] of Object.entries(mod)) {
    if (typeof val === "function") p[key] = val;
  }
}

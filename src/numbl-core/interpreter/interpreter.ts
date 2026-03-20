/**
 * AST-walking interpreter for MATLAB code.
 *
 * Walks the parsed AST directly without lowering or codegen.
 * All dispatch decisions are made at runtime using actual values.
 *
 * Reuses LoweringContext and FunctionIndex from the lowering pipeline
 * for workspace registration, class extraction, and function resolution.
 */

import type {
  Stmt,
  Expr,
  LValue,
  AbstractSyntaxTree,
} from "../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../parser/types.js";
import type { Runtime } from "../runtime/runtime.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeFunction,
  FloatXArray,
} from "../runtime/types.js";
import { RTV, getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import {
  binop,
  uminus,
  uplus,
  ctranspose,
} from "../runtime/runtimeOperators.js";
import {
  makeRangeTensor,
  horzcat,
  vertcat,
} from "../runtime/tensor-construction.js";
import { COLON_SENTINEL, END_SENTINEL } from "../executor/types.js";
import { toNumber } from "../runtime/convert.js";
import { numel } from "../runtime/utils.js";
import type {
  LoweringContext,
  FunctionIndex,
} from "../lowering/loweringContext.js";
import type { ClassInfo } from "../lowering/classInfo.js";
import { resolveFunction, type ResolvedTarget } from "../functionResolve.js";
import type { ItemType } from "../lowering/itemTypes.js";

// ── Control flow signals ─────────────────────────────────────────────────

class BreakSignal {
  readonly _tag = "break";
}
class ContinueSignal {
  readonly _tag = "continue";
}
class ReturnSignal {
  readonly _tag = "return";
  constructor(public values: RuntimeValue[]) {}
}

type ControlSignal = BreakSignal | ContinueSignal | ReturnSignal;

// ── Environment (variable scope) ─────────────────────────────────────────

class Environment {
  private vars = new Map<string, RuntimeValue>();
  /** When true, writes to variables found in parent go to the parent (nested function semantics). */
  isNested = false;
  /** Nested function definitions registered during execution. */
  nestedFunctions = new Map<string, { fn: FunctionDef; env: Environment }>();

  constructor(private parent?: Environment) {}

  get(name: string): RuntimeValue | undefined {
    return this.vars.get(name) ?? this.parent?.get(name);
  }

  /** Set variable — for nested scopes, writes to parent if variable exists there. */
  set(name: string, value: RuntimeValue): void {
    if (this.isNested && !this.vars.has(name) && this.parent) {
      const owner = this.findOwner(name);
      if (owner) {
        owner.setLocal(name, value);
        return;
      }
    }
    this.vars.set(name, value);
  }

  /** Always writes to this scope (for parameter binding). */
  setLocal(name: string, value: RuntimeValue): void {
    this.vars.set(name, value);
  }

  has(name: string): boolean {
    return this.vars.has(name) || (this.parent?.has(name) ?? false);
  }

  /** Find the environment that owns a variable. */
  private findOwner(name: string): Environment | null {
    if (this.vars.has(name)) return this;
    return this.parent?.findOwner(name) ?? null;
  }

  /** Look up a nested function definition in this scope or parent scopes. */
  getNestedFunction(
    name: string
  ): { fn: FunctionDef; env: Environment } | undefined {
    return (
      this.nestedFunctions.get(name) ?? this.parent?.getNestedFunction(name)
    );
  }

  localNames(): string[] {
    return [...this.vars.keys()];
  }

  /** Create a snapshot of this environment (copies all variables by value).
   *  Used for anonymous functions which capture values at definition time. */
  snapshot(): Environment {
    const snap = new Environment();
    // Copy all variables from the entire chain
    const copyVars = (env: Environment) => {
      if (env.parent) copyVars(env.parent);
      for (const [k, v] of env.vars) {
        snap.vars.set(k, v);
      }
      // Also copy nested function registrations
      for (const [k, v] of env.nestedFunctions) {
        snap.nestedFunctions.set(k, v);
      }
    };
    copyVars(this);
    return snap;
  }

  toRecord(): Record<string, RuntimeValue> {
    const result: Record<string, RuntimeValue> = {};
    if (this.parent) {
      Object.assign(result, this.parent.toRecord());
    }
    for (const [k, v] of this.vars) {
      result[k] = v;
    }
    return result;
  }
}

// ── Function definition storage ──────────────────────────────────────────

interface FunctionDef {
  name: string;
  params: string[];
  outputs: string[];
  body: Stmt[];
}

// ── Interpreter ──────────────────────────────────────────────────────────

export class Interpreter {
  private env: Environment;
  public ans: RuntimeValue | undefined;

  // Lowering context and function index (reused from lowering pipeline)
  private ctx: LoweringContext;
  private functionIndex: FunctionIndex;

  // Current execution context (for call site tracking)
  private currentFile: string;
  private currentClassName: string | undefined;
  private currentMethodName: string | undefined;

  // Main script's local functions
  private mainLocalFunctions = new Map<string, FunctionDef>();

  /** Stack of [base, dimIndex, numIndices] for resolving `end` keyword in indexing. */
  private endContextStack: Array<{
    base: unknown;
    dimIndex: number;
    numIndices: number;
  }> = [];

  // Cache for resolved function ASTs (avoid re-parsing)
  private functionDefCache = new Map<string, FunctionDef>();

  // Guard against infinite recursion in compileSpecialized
  private compileInProgress = new Set<string>();

  constructor(
    private rt: Runtime,
    ctx: LoweringContext,
    functionIndex: FunctionIndex,
    mainFileName: string,
    initialVariableValues?: Record<string, RuntimeValue>
  ) {
    this.ctx = ctx;
    this.functionIndex = functionIndex;
    this.currentFile = mainFileName;
    this.env = new Environment();
    if (initialVariableValues) {
      for (const [name, value] of Object.entries(initialVariableValues)) {
        this.env.set(name, value);
      }
    }
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
    // First pass: collect local function definitions
    for (const stmt of ast.body) {
      if (stmt.type === "Function") {
        const fn: FunctionDef = {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        };
        this.mainLocalFunctions.set(stmt.name, fn);
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

  // ── Statement execution ──────────────────────────────────────────────

  private execStmt(stmt: Stmt): ControlSignal | null {
    // Set line tracking for error messages
    if (stmt.span) {
      this.rt.$file = stmt.span.file;
    }

    switch (stmt.type) {
      case "ExprStmt": {
        const val = this.evalExpr(stmt.expr);
        // CSL (comma-separated list) from cell{:} → take first element
        const singleVal = Array.isArray(val) ? val[0] : val;
        const rv = ensureRuntimeValue(singleVal);
        this.ans = rv;
        if (!stmt.suppressed && !this.isOutputExpr(stmt.expr)) {
          this.rt.displayResult(rv);
        }
        return null;
      }

      case "Assign": {
        const rawVal = this.evalExpr(stmt.expr);
        const val = Array.isArray(rawVal) ? rawVal[0] : rawVal;
        const rv = this.rt.share(val) as RuntimeValue;
        this.env.set(stmt.name, rv);
        this.ans = rv;
        if (!stmt.suppressed) {
          this.rt.displayAssign(stmt.name, rv);
        }
        return null;
      }

      case "MultiAssign": {
        const nargout = stmt.lvalues.length;
        const val = this.evalExprNargout(stmt.expr, nargout);
        const values = Array.isArray(val) ? val : [val];
        for (let i = 0; i < stmt.lvalues.length; i++) {
          const lv = stmt.lvalues[i];
          if (lv.type === "Ignore") continue;
          const rv = this.rt.share(
            i < values.length ? values[i] : undefined
          ) as RuntimeValue;
          this.assignLValue(lv, rv);
        }
        if (!stmt.suppressed && values.length > 0) {
          const firstLv = stmt.lvalues[0];
          if (firstLv.type === "Var") {
            this.rt.displayAssign(firstLv.name, ensureRuntimeValue(values[0]));
          }
        }
        return null;
      }

      case "AssignLValue": {
        const val = this.evalExpr(stmt.expr);
        const rv = this.rt.share(val) as RuntimeValue;
        this.assignLValue(stmt.lvalue, rv);
        if (!stmt.suppressed) {
          if (stmt.lvalue.type === "Var") {
            this.rt.displayAssign(stmt.lvalue.name, rv);
          }
        }
        return null;
      }

      case "If": {
        const cond = this.evalExpr(stmt.cond);
        if (this.rt.toBool(cond)) {
          return this.execStmts(stmt.thenBody);
        }
        for (const elseif of stmt.elseifBlocks) {
          const elseifCond = this.evalExpr(elseif.cond);
          if (this.rt.toBool(elseifCond)) {
            return this.execStmts(elseif.body);
          }
        }
        if (stmt.elseBody) {
          return this.execStmts(stmt.elseBody);
        }
        return null;
      }

      case "While": {
        while (true) {
          const cond = this.evalExpr(stmt.cond);
          if (!this.rt.toBool(cond)) break;
          const signal = this.execStmts(stmt.body);
          if (signal instanceof BreakSignal) break;
          if (signal instanceof ContinueSignal) continue;
          if (signal instanceof ReturnSignal) return signal;
        }
        return null;
      }

      case "For": {
        const iterVal = this.evalExpr(stmt.expr);
        const rv = ensureRuntimeValue(iterVal);

        if (isRuntimeTensor(rv)) {
          const shape = rv.shape;
          if (shape.length === 2 && shape[0] === 1) {
            for (let i = 0; i < rv.data.length; i++) {
              this.env.set(stmt.varName, rv.data[i] as unknown as RuntimeValue);
              const signal = this.execStmts(stmt.body);
              if (signal instanceof BreakSignal) break;
              if (signal instanceof ContinueSignal) continue;
              if (signal instanceof ReturnSignal) return signal;
            }
          } else {
            const rows = shape[0];
            const cols = shape.length >= 2 ? shape[1] : 1;
            for (let c = 0; c < cols; c++) {
              const colData = new FloatXArray(rows);
              for (let r = 0; r < rows; r++) {
                colData[r] = rv.data[c * rows + r];
              }
              if (rows === 1) {
                this.env.set(
                  stmt.varName,
                  colData[0] as unknown as RuntimeValue
                );
              } else {
                this.env.set(stmt.varName, RTV.tensor(colData, [rows, 1]));
              }
              const signal = this.execStmts(stmt.body);
              if (signal instanceof BreakSignal) break;
              if (signal instanceof ContinueSignal) continue;
              if (signal instanceof ReturnSignal) return signal;
            }
          }
        } else if (isRuntimeCell(rv)) {
          for (let i = 0; i < rv.data.length; i++) {
            this.env.set(stmt.varName, rv.data[i]);
            const signal = this.execStmts(stmt.body);
            if (signal instanceof BreakSignal) break;
            if (signal instanceof ContinueSignal) continue;
            if (signal instanceof ReturnSignal) return signal;
          }
        } else if (isRuntimeNumber(rv)) {
          this.env.set(stmt.varName, rv);
          const signal = this.execStmts(stmt.body);
          if (signal instanceof ReturnSignal) return signal;
        } else {
          throw new RuntimeError(`Cannot iterate over ${typeof rv}`);
        }
        return null;
      }

      case "Switch": {
        const switchVal = this.evalExpr(stmt.expr);
        let matched = false;
        for (const c of stmt.cases) {
          const caseVal = this.evalExpr(c.value);
          if (this.switchMatch(switchVal, caseVal)) {
            matched = true;
            const signal = this.execStmts(c.body);
            if (signal) return signal;
            break;
          }
        }
        if (!matched && stmt.otherwise) {
          return this.execStmts(stmt.otherwise);
        }
        return null;
      }

      case "TryCatch": {
        try {
          const signal = this.execStmts(stmt.tryBody);
          if (signal) return signal;
        } catch (e) {
          if (stmt.catchVar && e instanceof Error) {
            this.env.set(
              stmt.catchVar,
              RTV.struct({
                message: RTV.string(e.message),
                identifier: RTV.string(""),
              })
            );
          }
          const signal = this.execStmts(stmt.catchBody);
          if (signal) return signal;
        }
        return null;
      }

      case "Function": {
        // Register as a nested function with access to current environment
        const nestedFn: FunctionDef = {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        };
        this.env.nestedFunctions.set(stmt.name, {
          fn: nestedFn,
          env: this.env,
        });
        return null;
      }

      case "Return":
        return new ReturnSignal([]);

      case "Break":
        return new BreakSignal();

      case "Continue":
        return new ContinueSignal();

      case "Global":
      case "Persistent":
        // TODO: implement global/persistent scope
        return null;

      case "ClassDef":
        // ClassDef in local file — already registered during setup
        return null;

      case "Import":
        // Imports are handled by the function index (collected during buildFunctionIndex)
        return null;
    }
  }

  private execStmts(stmts: Stmt[]): ControlSignal | null {
    for (const stmt of stmts) {
      const signal = this.execStmt(stmt);
      if (signal) return signal;
    }
    return null;
  }

  // ── Expression evaluation ────────────────────────────────────────────

  evalExpr(expr: Expr): unknown {
    return this.evalExprNargout(expr, 1);
  }

  private evalExprNargout(expr: Expr, nargout: number): unknown {
    switch (expr.type) {
      case "Number":
        return parseFloat(expr.value);

      case "Char": {
        let s = expr.value.slice(1, expr.value.length - 1);
        s = s.replaceAll("''", "'");
        return RTV.char(s);
      }

      case "String": {
        let s = expr.value.slice(1, expr.value.length - 1);
        s = s.replaceAll('""', '"');
        return RTV.string(s);
      }

      case "Ident": {
        const val = this.env.get(expr.name);
        if (val !== undefined) return val;
        try {
          return this.rt.getConstant(expr.name);
        } catch {
          // Not a constant
        }
        return this.callFunction(expr.name, [], nargout);
      }

      case "ImagUnit":
        return RTV.complex(0, 1);

      case "EndKeyword": {
        const ctx = this.endContextStack[this.endContextStack.length - 1];
        if (ctx) {
          const rv = ensureRuntimeValue(ctx.base);
          if (isRuntimeTensor(rv)) {
            if (ctx.numIndices === 1) return numel(rv.shape);
            return ctx.dimIndex < rv.shape.length ? rv.shape[ctx.dimIndex] : 1;
          }
          if (isRuntimeCell(rv)) {
            if (ctx.numIndices === 1) return numel(rv.shape);
            return ctx.dimIndex < rv.shape.length ? rv.shape[ctx.dimIndex] : 1;
          }
          if (isRuntimeChar(rv)) return rv.value.length;
          if (isRuntimeString(rv)) return (rv as string).length;
        }
        return END_SENTINEL;
      }

      case "Colon":
        return COLON_SENTINEL;

      case "Binary":
        return this.evalBinary(expr);

      case "Unary":
        return this.evalUnary(expr);

      case "Range":
        return this.evalRange(expr);

      case "FuncCall":
        return this.evalFuncCall(expr, nargout);

      case "Index":
        return this.evalIndex(expr);

      case "IndexCell":
        return this.evalIndexCell(expr);

      case "Member":
        return this.evalMember(expr, nargout);

      case "MemberDynamic": {
        const base = this.evalExpr(expr.base);
        const nameVal = this.evalExpr(expr.nameExpr);
        const name =
          typeof nameVal === "string"
            ? nameVal
            : isRuntimeChar(ensureRuntimeValue(nameVal))
              ? (ensureRuntimeValue(nameVal) as { value: string }).value
              : String(nameVal);
        return this.rt.getMember(base, name);
      }

      case "MethodCall": {
        // Check if this is a namespace-qualified call (pkg.func(args) or ClassName.staticMethod(args))
        const dottedBase = this.tryExtractDottedName(expr.base);
        if (dottedBase && !this.env.has(dottedBase.split(".")[0])) {
          const args = this.evalArgs(expr.args);
          // Try as namespace function: "dottedBase.name"
          const qualifiedName = `${dottedBase}.${expr.name}`;
          if (this.functionIndex.workspaceFunctions.has(qualifiedName)) {
            return this.callFunction(qualifiedName, args, nargout);
          }
          // Try as class constructor
          if (this.functionIndex.workspaceClasses.has(qualifiedName)) {
            return this.instantiateClass(qualifiedName, args, nargout);
          }
          // Try as static method on a class
          if (
            this.functionIndex.classStaticMethods
              .get(dottedBase)
              ?.has(expr.name)
          ) {
            const target: ResolvedTarget = {
              kind: "classMethod",
              className: dottedBase,
              methodName: expr.name,
              compileArgTypes: args.map(a =>
                getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
              ),
              stripInstance: false,
            };
            return this.interpretTarget(target, args, nargout);
          }
          // Try as class constructor call: ClassName(args) where ClassName = dottedBase
          if (this.functionIndex.workspaceClasses.has(dottedBase)) {
            // dottedBase is a class, expr.name is a method call on the class
            const target: ResolvedTarget = {
              kind: "classMethod",
              className: dottedBase,
              methodName: expr.name,
              compileArgTypes: args.map(a =>
                getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
              ),
              stripInstance: false,
            };
            return this.interpretTarget(target, args, nargout);
          }
        }
        const base = this.evalExpr(expr.base);
        const args = this.evalArgs(expr.args);
        return this.rt.methodDispatch(expr.name, nargout, [base, ...args]);
      }

      case "SuperMethodCall": {
        const args = this.evalArgs(expr.args);
        // Check if this is a super constructor call (obj@SuperClass(args))
        // vs a super method call (method@SuperClass(args))
        const objVal = this.env.get(expr.methodName);
        if (objVal !== undefined) {
          // Super constructor call: call super constructor with [obj, ...args]
          const superClassInfo = this.ctx.getClassInfo(expr.superClassName);
          if (superClassInfo && superClassInfo.constructorName) {
            const superResult = this.interpretConstructor(
              superClassInfo,
              [objVal, ...args],
              1
            );
            // Merge super fields into obj
            return this.rt.callSuperConstructor(objVal, superResult);
          }
          // No constructor — just initialize super properties
          const { propertyNames, propertyDefaults } =
            this.collectClassProperties(
              superClassInfo ??
                ({
                  propertyNames: [],
                  propertyDefaults: new Map(),
                  superClass: null,
                } as unknown as ClassInfo)
            );
          const defaults = new Map<string, RuntimeValue>();
          for (const [propName, defaultExpr] of propertyDefaults) {
            try {
              defaults.set(
                propName,
                ensureRuntimeValue(this.evalExpr(defaultExpr))
              );
            } catch {
              /* skip */
            }
          }
          const superInstance = RTV.classInstance(
            expr.superClassName,
            propertyNames,
            false,
            defaults
          );
          return this.rt.callSuperConstructor(objVal, superInstance);
        }
        // Super method call: method@SuperClass(args)
        return this.rt.callClassMethod(
          expr.superClassName,
          expr.methodName,
          nargout,
          args
        );
      }

      case "AnonFunc":
        return this.evalAnonFunc(expr);

      case "FuncHandle":
        return this.makeFuncHandle(expr.name);

      case "Tensor":
        return this.evalTensorLiteral(expr);

      case "Cell":
        return this.evalCellLiteral(expr);

      case "ClassInstantiation": {
        const args = this.evalArgs(expr.args);
        return this.instantiateClass(expr.className, args, nargout);
      }

      case "MetaClass":
        throw new RuntimeError("Interpreter does not yet support meta.class");
    }
  }

  /** Evaluate function arguments, flattening CSL (comma-separated list) expansions. */
  private evalArgs(argExprs: Expr[]): unknown[] {
    const args: unknown[] = [];
    for (const a of argExprs) {
      const val = this.evalExpr(a);
      // If the result is a raw JS array (CSL from cell{:} expansion), spread it
      if (Array.isArray(val)) {
        for (const elem of val) {
          args.push(elem);
        }
      } else {
        args.push(val);
      }
    }
    return args;
  }

  // ── Binary operators ─────────────────────────────────────────────────

  private evalBinary(expr: Extract<Expr, { type: "Binary" }>): unknown {
    if (expr.op === BinaryOperation.AndAnd) {
      const left = this.evalExpr(expr.left);
      if (!this.rt.toBool(left)) return RTV.logical(false);
      const right = this.evalExpr(expr.right);
      return RTV.logical(this.rt.toBool(right));
    }
    if (expr.op === BinaryOperation.OrOr) {
      const left = this.evalExpr(expr.left);
      if (this.rt.toBool(left)) return RTV.logical(true);
      const right = this.evalExpr(expr.right);
      return RTV.logical(this.rt.toBool(right));
    }

    const left = this.evalExpr(expr.left);
    const right = this.evalExpr(expr.right);

    const lv = ensureRuntimeValue(left);
    const rv = ensureRuntimeValue(right);
    if (isRuntimeClassInstance(lv) || isRuntimeClassInstance(rv)) {
      return this.rt.binop(expr.op, left, right);
    }

    return binop(expr.op, left, right);
  }

  // ── Unary operators ──────────────────────────────────────────────────

  private evalUnary(expr: Extract<Expr, { type: "Unary" }>): unknown {
    const operand = this.evalExpr(expr.operand);

    switch (expr.op) {
      case UnaryOperation.Plus:
        return uplus(operand);
      case UnaryOperation.Minus:
        return uminus(operand);
      case UnaryOperation.Not: {
        if (typeof operand === "number") return RTV.logical(operand === 0);
        return this.rt.not(operand);
      }
      case UnaryOperation.Transpose:
      case UnaryOperation.NonConjugateTranspose:
        return ctranspose(operand);
    }
  }

  // ── Range ────────────────────────────────────────────────────────────

  private evalRange(expr: Extract<Expr, { type: "Range" }>): unknown {
    const startVal = toNumber(ensureRuntimeValue(this.evalExpr(expr.start)));
    const endVal = toNumber(ensureRuntimeValue(this.evalExpr(expr.end)));
    const stepVal = expr.step
      ? toNumber(ensureRuntimeValue(this.evalExpr(expr.step)))
      : 1;
    return makeRangeTensor(startVal, stepVal, endVal);
  }

  // ── Function calls ───────────────────────────────────────────────────

  private evalFuncCall(
    expr: Extract<Expr, { type: "FuncCall" }>,
    nargout: number
  ): unknown {
    // Check if it's variable indexing first
    const varVal = this.env.get(expr.name);
    if (varVal !== undefined) {
      const rv = ensureRuntimeValue(varVal);
      if (isRuntimeFunction(rv)) {
        const args = this.evalArgs(expr.args);
        return this.rt.index(rv, args, nargout);
      }
      const args = this.evalIndicesWithEnd(varVal, expr.args);
      return this.rt.index(varVal, args, nargout);
    }

    const args = this.evalArgs(expr.args);
    return this.callFunction(expr.name, args, nargout);
  }

  private callFunction(
    name: string,
    args: unknown[],
    nargout: number
  ): unknown {
    // 0. Intrinsics
    if (name === "isa" && args.length === 2) {
      return this.rt.isa(args[0], args[1]);
    }
    if (name === "__inferred_type_str" && args.length === 1) {
      const rv = ensureRuntimeValue(args[0]);
      if (isRuntimeNumber(rv)) return RTV.string("Number");
      if (isRuntimeTensor(rv)) return RTV.string("Tensor");
      if (isRuntimeCell(rv)) return RTV.string("Cell");
      if (isRuntimeClassInstance(rv))
        return RTV.string(`ClassInstance(${rv.className})`);
      if (isRuntimeChar(rv)) return RTV.string("Char");
      if (isRuntimeString(rv)) return RTV.string("String");
      if (isRuntimeFunction(rv)) return RTV.string("Function");
      if (typeof rv === "boolean") return RTV.string("Boolean");
      return RTV.string("Unknown");
    }
    if (name === "nargin" && args.length === 0) {
      const v = this.env.get("$nargin");
      return v !== undefined ? v : 0;
    }
    if (name === "nargout" && args.length === 0) {
      const v = this.env.get("$nargout");
      return v !== undefined ? v : 0;
    }

    // 1. Check nested functions (share parent scope)
    const nested = this.env.getNestedFunction(name);
    if (nested) {
      return this.callNestedFunction(nested.fn, nested.env, args, nargout);
    }

    // 2. Resolve using function index
    const argTypes = args.map(a =>
      getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
    );
    const callSite: CallSite = {
      file: this.currentFile,
      ...(this.currentClassName ? { className: this.currentClassName } : {}),
      ...(this.currentMethodName ? { methodName: this.currentMethodName } : {}),
    };
    const target = resolveFunction(
      name,
      argTypes,
      callSite,
      this.functionIndex
    );
    if (target) {
      return this.interpretTarget(target, args, nargout);
    }

    throw new RuntimeError(`Undefined function or variable '${name}'`);
  }

  // ── Target interpretation ────────────────────────────────────────────

  private interpretTarget(
    target: ResolvedTarget,
    args: unknown[],
    nargout: number
  ): unknown {
    switch (target.kind) {
      case "builtin": {
        const builtin = this.rt.builtins[target.name];
        if (builtin) return builtin(nargout, args);
        throw new RuntimeError(`Unknown builtin: '${target.name}'`);
      }

      case "localFunction":
        return this.interpretLocalFunction(target, args, nargout);

      case "workspaceFunction":
        return this.interpretWorkspaceFunction(target, args, nargout);

      case "classMethod":
        return this.interpretClassMethod(target, args, nargout);

      case "workspaceClassConstructor":
        return this.instantiateClass(target.className, args, nargout);

      case "privateFunction":
        return this.interpretPrivateFunction(target, args, nargout);
    }
  }

  // ── Local function interpretation ────────────────────────────────────

  private interpretLocalFunction(
    target: Extract<ResolvedTarget, { kind: "localFunction" }>,
    args: unknown[],
    nargout: number
  ): unknown {
    const { source } = target;

    if (source.from === "main") {
      const fn = this.mainLocalFunctions.get(target.name);
      if (!fn)
        throw new RuntimeError(`Local function '${target.name}' not found`);
      return this.callUserFunction(fn, args, nargout);
    }

    if (source.from === "workspaceFile") {
      const fn = this.findFunctionInWorkspaceFile(source.wsName, target.name);
      if (!fn)
        throw new RuntimeError(
          `Local function '${target.name}' not found in workspace file '${source.wsName}'`
        );
      // Execute in the workspace file's context
      return this.withFileContext(
        this.getWorkspaceFileName(source.wsName),
        undefined,
        undefined,
        () => this.callUserFunction(fn, args, nargout)
      );
    }

    if (source.from === "classFile") {
      const fn = this.findFunctionInClassFile(source.className, target.name);
      if (!fn)
        throw new RuntimeError(
          `Local function '${target.name}' not found in class file '${source.className}'`
        );
      return this.withFileContext(
        this.getClassFileName(source.className),
        source.className,
        source.methodScope,
        () => this.callUserFunction(fn, args, nargout)
      );
    }

    if (source.from === "privateFile") {
      // The callerFile is the private file itself (contains /private/)
      // Look up the file's AST and find the subfunction
      const ast = this.ctx.getCachedAST(source.callerFile);
      for (const stmt of ast.body) {
        if (stmt.type === "Function" && stmt.name === target.name) {
          const fn: FunctionDef = {
            name: stmt.name,
            params: stmt.params,
            outputs: stmt.outputs,
            body: stmt.body,
          };
          return this.withFileContext(
            source.callerFile,
            undefined,
            undefined,
            () => this.callUserFunction(fn, args, nargout)
          );
        }
      }
      throw new RuntimeError(
        `Local function '${target.name}' not found in private file`
      );
    }

    throw new RuntimeError(`Unknown local function source`);
  }

  // ── Workspace function interpretation ────────────────────────────────

  private interpretWorkspaceFunction(
    target: Extract<ResolvedTarget, { kind: "workspaceFunction" }>,
    args: unknown[],
    nargout: number
  ): unknown {
    // The primary name may be dotted (e.g., "pkg.func")
    const dotIdx = target.name.lastIndexOf(".");
    const primaryName =
      dotIdx >= 0 ? target.name.slice(dotIdx + 1) : target.name;

    const fn = this.findFunctionInWorkspaceFile(target.name, primaryName);
    if (!fn) {
      // Could be a script file (no function declaration)
      const entry = this.ctx.registry.filesByFuncName.get(target.name);
      if (entry) {
        const ast = this.ctx.getCachedAST(entry.fileName);
        // Script: execute all statements
        return this.withFileContext(
          entry.fileName,
          undefined,
          undefined,
          () => {
            const savedEnv = this.env;
            this.env = new Environment();
            // Bind args as positional (not standard MATLAB, but handles edge cases)
            try {
              for (const stmt of ast.body) {
                if (stmt.type === "Function") continue;
                const signal = this.execStmt(stmt);
                if (signal instanceof ReturnSignal) break;
              }
              return this.ans;
            } finally {
              this.env = savedEnv;
            }
          }
        );
      }
      throw new RuntimeError(`Workspace function '${target.name}' not found`);
    }

    return this.withFileContext(
      this.getWorkspaceFileName(target.name),
      undefined,
      undefined,
      () => this.callUserFunction(fn, args, nargout)
    );
  }

  // ── Class method interpretation ──────────────────────────────────────

  private interpretClassMethod(
    target: Extract<ResolvedTarget, { kind: "classMethod" }>,
    args: unknown[],
    nargout: number
  ): unknown {
    const { className, methodName } = target;

    // Walk inheritance to find the defining class
    const definingClass = this.ctx.findDefiningClass(className, methodName);
    const classInfo = this.ctx.getClassInfo(definingClass);
    if (!classInfo)
      throw new RuntimeError(`Class '${definingClass}' not found`);

    // Check if it's the constructor
    if (methodName === classInfo.constructorName) {
      // Constructor call directly (e.g., from superclass)
      return this.interpretConstructor(classInfo, args, nargout);
    }

    // Find the method AST
    const isStatic = classInfo.staticMethodNames.has(methodName);
    const methodFn = this.findMethodInClass(classInfo, methodName);
    if (!methodFn) {
      // Check external method files
      const extFn = this.findExternalMethod(classInfo, methodName);
      if (extFn) {
        // For static methods that got instance passed, strip it
        const actualArgs =
          target.stripInstance && args.length > 0 ? args.slice(1) : args;
        return this.withFileContext(
          classInfo.externalMethodFiles.get(methodName)?.fileName ??
            classInfo.fileName,
          definingClass,
          methodName,
          () => this.callUserFunction(extFn, actualArgs, nargout)
        );
      }
      throw new RuntimeError(
        `No method '${methodName}' for class '${className}'`
      );
    }

    // For static methods that got instance passed, strip it
    const actualArgs =
      target.stripInstance && !isStatic && args.length > 0
        ? args.slice(1)
        : args;

    return this.withFileContext(
      classInfo.fileName,
      definingClass,
      methodName,
      () => this.callUserFunction(methodFn, actualArgs, nargout)
    );
  }

  // ── Private function interpretation ──────────────────────────────────

  private interpretPrivateFunction(
    target: Extract<ResolvedTarget, { kind: "privateFunction" }>,
    args: unknown[],
    nargout: number
  ): unknown {
    const entry = this.ctx.getPrivateFileEntry(target.callerFile, target.name);
    if (!entry)
      throw new RuntimeError(`Private function '${target.name}' not found`);

    const ast = this.ctx.getCachedAST(entry.fileName);
    for (const stmt of ast.body) {
      if (stmt.type === "Function" && stmt.name === target.name) {
        const fn: FunctionDef = {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        };
        return this.withFileContext(entry.fileName, undefined, undefined, () =>
          this.callUserFunction(fn, args, nargout)
        );
      }
    }
    throw new RuntimeError(`Private function '${target.name}' not found`);
  }

  // ── Class instantiation ──────────────────────────────────────────────

  private instantiateClass(
    className: string,
    args: unknown[],
    nargout: number
  ): unknown {
    const classInfo = this.ctx.getClassInfo(className);
    if (!classInfo) {
      // Fall back to runtime dispatch (might be a builtin class)
      return this.rt.callClassMethod(className, className, nargout, args);
    }

    // Collect all properties from inheritance chain
    const { propertyNames, propertyDefaults } =
      this.collectClassProperties(classInfo);

    // Evaluate property defaults
    const defaults = new Map<string, RuntimeValue>();
    for (const [propName, defaultExpr] of propertyDefaults) {
      try {
        defaults.set(propName, ensureRuntimeValue(this.evalExpr(defaultExpr)));
      } catch {
        // Default evaluation failed — use empty matrix
      }
    }

    // Determine if handle class
    const isHandle = this.isHandleClass(classInfo);

    // Create the instance
    const instance = RTV.classInstance(
      className,
      propertyNames,
      isHandle,
      defaults
    );

    // Call constructor if it exists
    if (classInfo.constructorName) {
      return this.interpretConstructor(classInfo, [instance, ...args], nargout);
    }

    return instance;
  }

  private interpretConstructor(
    classInfo: ClassInfo,
    args: unknown[],
    nargout: number
  ): unknown {
    const constructorName = classInfo.constructorName;
    if (!constructorName) return args[0]; // No constructor, return instance as-is

    // Find the constructor method
    for (const member of classInfo.ast.members) {
      if (member.type !== "Methods") continue;
      for (const methodStmt of member.body) {
        if (
          methodStmt.type === "Function" &&
          methodStmt.name === constructorName
        ) {
          // Transform: prepend output variable as first parameter
          const outputName =
            methodStmt.outputs.length > 0 ? methodStmt.outputs[0] : "obj";
          const fn: FunctionDef = {
            name: constructorName,
            params: [outputName, ...methodStmt.params],
            outputs: methodStmt.outputs,
            body: methodStmt.body,
          };
          return this.withFileContext(
            classInfo.fileName,
            classInfo.name,
            constructorName,
            () => this.callUserFunction(fn, args, nargout)
          );
        }
      }
    }

    // Constructor might be in an external method file
    const extFn = this.findExternalMethod(classInfo, constructorName);
    if (extFn) {
      // Transform: prepend output variable as first parameter
      const outputName = extFn.outputs.length > 0 ? extFn.outputs[0] : "obj";
      const fn: FunctionDef = {
        ...extFn,
        params: [outputName, ...extFn.params],
      };
      return this.withFileContext(
        classInfo.externalMethodFiles.get(constructorName)?.fileName ??
          classInfo.fileName,
        classInfo.name,
        constructorName,
        () => this.callUserFunction(fn, args, nargout)
      );
    }

    return args[0]; // No constructor found, return instance as-is
  }

  // ── Call user function (core execution) ──────────────────────────────

  private callUserFunction(
    fn: FunctionDef,
    args: unknown[],
    nargout: number
  ): unknown {
    const fnEnv = new Environment();

    // Bind parameters (handle varargin)
    const hasVarargin =
      fn.params.length > 0 && fn.params[fn.params.length - 1] === "varargin";
    const regularParams = hasVarargin ? fn.params.slice(0, -1) : fn.params;
    for (let i = 0; i < regularParams.length; i++) {
      if (i < args.length) {
        fnEnv.set(regularParams[i], ensureRuntimeValue(args[i]));
      }
    }
    if (hasVarargin) {
      const extraArgs = args
        .slice(regularParams.length)
        .map(a => ensureRuntimeValue(a));
      fnEnv.set("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
    }
    fnEnv.set("$nargin", args.length as unknown as RuntimeValue);
    fnEnv.set("$nargout", nargout as unknown as RuntimeValue);

    const savedEnv = this.env;
    this.env = fnEnv;

    try {
      this.execStmts(fn.body);

      // Collect outputs (handle varargout)
      const hasVarargout =
        fn.outputs.length > 0 &&
        fn.outputs[fn.outputs.length - 1] === "varargout";
      const regularOutputs = hasVarargout
        ? fn.outputs.slice(0, -1)
        : fn.outputs;

      const outputs: RuntimeValue[] = [];
      for (let i = 0; i < Math.min(regularOutputs.length, nargout); i++) {
        const val = this.env.get(regularOutputs[i]);
        outputs.push(val ?? RTV.num(0));
      }

      // Append varargout elements if needed
      if (hasVarargout && nargout > regularOutputs.length) {
        const varargoutVal = this.env.get("varargout");
        if (varargoutVal && isRuntimeCell(varargoutVal)) {
          const remaining = nargout - regularOutputs.length;
          for (
            let i = 0;
            i < Math.min(remaining, varargoutVal.data.length);
            i++
          ) {
            outputs.push(ensureRuntimeValue(varargoutVal.data[i]));
          }
        }
      }

      if (nargout <= 1) {
        return outputs[0];
      }
      return outputs;
    } finally {
      this.env = savedEnv;
    }
  }

  // ── Nested function call (shares parent scope) ───────────────────────

  private callNestedFunction(
    fn: FunctionDef,
    parentEnv: Environment,
    args: unknown[],
    nargout: number
  ): unknown {
    const fnEnv = new Environment(parentEnv);
    fnEnv.isNested = true;

    // Bind parameters to local scope (not shared)
    const hasVarargin =
      fn.params.length > 0 && fn.params[fn.params.length - 1] === "varargin";
    const regularParams = hasVarargin ? fn.params.slice(0, -1) : fn.params;
    for (let i = 0; i < regularParams.length; i++) {
      if (i < args.length) {
        fnEnv.setLocal(regularParams[i], ensureRuntimeValue(args[i]));
      }
    }
    if (hasVarargin) {
      const extraArgs = args
        .slice(regularParams.length)
        .map(a => ensureRuntimeValue(a));
      fnEnv.setLocal("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
    }
    fnEnv.setLocal("$nargin", args.length as unknown as RuntimeValue);
    fnEnv.setLocal("$nargout", nargout as unknown as RuntimeValue);

    const savedEnv = this.env;
    this.env = fnEnv;

    try {
      this.execStmts(fn.body);

      // Collect outputs (handle varargout)
      const hasVarargout =
        fn.outputs.length > 0 &&
        fn.outputs[fn.outputs.length - 1] === "varargout";
      const regularOutputs = hasVarargout
        ? fn.outputs.slice(0, -1)
        : fn.outputs;

      const outputs: RuntimeValue[] = [];
      for (let i = 0; i < Math.min(regularOutputs.length, nargout); i++) {
        const val = this.env.get(regularOutputs[i]);
        outputs.push(val ?? RTV.num(0));
      }

      if (hasVarargout && nargout > regularOutputs.length) {
        const varargoutVal = this.env.get("varargout");
        if (varargoutVal && isRuntimeCell(varargoutVal)) {
          const remaining = nargout - regularOutputs.length;
          for (
            let i = 0;
            i < Math.min(remaining, varargoutVal.data.length);
            i++
          ) {
            outputs.push(ensureRuntimeValue(varargoutVal.data[i]));
          }
        }
      }

      if (nargout <= 1) {
        return outputs[0];
      }
      return outputs;
    } finally {
      this.env = savedEnv;
    }
  }

  // ── Context management ───────────────────────────────────────────────

  /** Execute a callback with a different file/class context, then restore. */
  private withFileContext<T>(
    file: string,
    className: string | undefined,
    methodName: string | undefined,
    fn: () => T
  ): T {
    const savedFile = this.currentFile;
    const savedClassName = this.currentClassName;
    const savedMethodName = this.currentMethodName;
    this.currentFile = file;
    this.currentClassName = className;
    this.currentMethodName = methodName;
    try {
      return fn();
    } finally {
      this.currentFile = savedFile;
      this.currentClassName = savedClassName;
      this.currentMethodName = savedMethodName;
    }
  }

  // ── AST lookup helpers ───────────────────────────────────────────────

  private getWorkspaceFileName(funcName: string): string {
    const entry = this.ctx.registry.filesByFuncName.get(funcName);
    return entry?.fileName ?? funcName + ".m";
  }

  private getClassFileName(className: string): string {
    const info = this.ctx.getClassInfo(className);
    return info?.fileName ?? className + ".m";
  }

  private findFunctionInWorkspaceFile(
    wsName: string,
    funcName: string
  ): FunctionDef | null {
    const cacheKey = `ws:${wsName}:${funcName}`;
    const cached = this.functionDefCache.get(cacheKey);
    if (cached) return cached;

    const entry = this.ctx.registry.filesByFuncName.get(wsName);
    if (!entry) return null;

    const ast = this.ctx.getCachedAST(entry.fileName);
    for (const stmt of ast.body) {
      if (stmt.type === "Function" && stmt.name === funcName) {
        const fn: FunctionDef = {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        };
        this.functionDefCache.set(cacheKey, fn);
        return fn;
      }
    }
    return null;
  }

  private findFunctionInClassFile(
    className: string,
    funcName: string
  ): FunctionDef | null {
    const cacheKey = `cls:${className}:${funcName}`;
    const cached = this.functionDefCache.get(cacheKey);
    if (cached) return cached;

    const classInfo = this.ctx.getClassInfo(className);
    if (!classInfo) return null;

    // Check file-level helper functions (after the classdef block)
    const ast = this.ctx.getCachedAST(classInfo.fileName);
    for (const stmt of ast.body) {
      if (stmt.type === "Function" && stmt.name === funcName) {
        const fn: FunctionDef = {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        };
        this.functionDefCache.set(cacheKey, fn);
        return fn;
      }
    }

    // Check local helpers in external method files
    for (const [, mf] of classInfo.externalMethodFiles) {
      const methodAst = this.ctx.getCachedAST(mf.fileName);
      for (const stmt of methodAst.body) {
        if (stmt.type === "Function" && stmt.name === funcName) {
          const fn: FunctionDef = {
            name: stmt.name,
            params: stmt.params,
            outputs: stmt.outputs,
            body: stmt.body,
          };
          this.functionDefCache.set(cacheKey, fn);
          return fn;
        }
      }
    }

    return null;
  }

  private findMethodInClass(
    classInfo: ClassInfo,
    methodName: string
  ): FunctionDef | null {
    const cacheKey = `method:${classInfo.name}:${methodName}`;
    const cached = this.functionDefCache.get(cacheKey);
    if (cached) return cached;

    // Search in the classdef AST
    for (const member of classInfo.ast.members) {
      if (member.type !== "Methods") continue;
      for (const methodStmt of member.body) {
        if (methodStmt.type === "Function" && methodStmt.name === methodName) {
          const fn: FunctionDef = {
            name: methodStmt.name,
            params: methodStmt.params,
            outputs: methodStmt.outputs,
            body: methodStmt.body,
          };
          this.functionDefCache.set(cacheKey, fn);
          return fn;
        }
      }
    }

    // Walk up the inheritance chain
    if (classInfo.superClass) {
      const parentInfo = this.ctx.getClassInfo(classInfo.superClass);
      if (parentInfo) {
        return this.findMethodInClass(parentInfo, methodName);
      }
    }

    return null;
  }

  private findExternalMethod(
    classInfo: ClassInfo,
    methodName: string
  ): FunctionDef | null {
    const mf = classInfo.externalMethodFiles.get(methodName);
    if (!mf) {
      // Check parent class
      if (classInfo.superClass) {
        const parentInfo = this.ctx.getClassInfo(classInfo.superClass);
        if (parentInfo) return this.findExternalMethod(parentInfo, methodName);
      }
      return null;
    }

    const ast = this.ctx.getCachedAST(mf.fileName);
    for (const stmt of ast.body) {
      if (stmt.type === "Function" && stmt.name === methodName) {
        return {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        };
      }
    }
    return null;
  }

  // ── Class property helpers ───────────────────────────────────────────

  private collectClassProperties(classInfo: ClassInfo): {
    propertyNames: string[];
    propertyDefaults: Map<string, Expr>;
  } {
    const propertyNames: string[] = [...classInfo.propertyNames];
    const propertyDefaults = new Map(classInfo.propertyDefaults);

    let parentName = classInfo.superClass;
    while (parentName) {
      if (parentName === "handle") break;
      const parentInfo = this.ctx.getClassInfo(parentName);
      if (!parentInfo) break;
      for (const propName of parentInfo.propertyNames) {
        if (!propertyNames.includes(propName)) {
          propertyNames.push(propName);
          const defaultExpr = parentInfo.propertyDefaults.get(propName);
          if (defaultExpr && !propertyDefaults.has(propName)) {
            propertyDefaults.set(propName, defaultExpr);
          }
        }
      }
      parentName = parentInfo.superClass;
    }
    return { propertyNames, propertyDefaults };
  }

  private isHandleClass(classInfo: ClassInfo): boolean {
    let parentName = classInfo.superClass;
    while (parentName) {
      if (parentName === "handle") return true;
      const parentInfo = this.ctx.getClassInfo(parentName);
      if (!parentInfo) break;
      parentName = parentInfo.superClass;
    }
    return false;
  }

  // ── Member access ────────────────────────────────────────────────────

  private evalMember(
    expr: Extract<Expr, { type: "Member" }>,
    nargout: number
  ): unknown {
    // Check if this is a namespace-qualified name (pkg.func or ClassName.staticMethod)
    const dottedName = this.tryExtractDottedName(expr);
    if (dottedName) {
      // Check if it's a namespace function call
      const rootName = dottedName.split(".")[0];
      if (!this.env.has(rootName)) {
        // Try as workspace function or class constructor
        if (this.functionIndex.workspaceFunctions.has(dottedName)) {
          return this.callFunction(dottedName, [], nargout);
        }
        if (this.functionIndex.workspaceClasses.has(dottedName)) {
          return this.instantiateClass(dottedName, [], nargout);
        }
        // Try as static method: "ClassName.method"
        const lastDot = dottedName.lastIndexOf(".");
        if (lastDot > 0) {
          const prefix = dottedName.slice(0, lastDot);
          const methodName = dottedName.slice(lastDot + 1);
          if (
            this.functionIndex.classStaticMethods.get(prefix)?.has(methodName)
          ) {
            return this.callFunction(methodName, [], nargout);
          }
        }
      }
    }

    const base = this.evalExpr(expr.base);
    return this.rt.getMember(base, expr.name);
  }

  /** Try to extract a dotted name chain from a Member expression (e.g., a.b.c → "a.b.c"). */
  private tryExtractDottedName(expr: Expr): string | null {
    if (expr.type === "Ident") return expr.name;
    if (expr.type === "Member") {
      const baseChain = this.tryExtractDottedName(expr.base);
      if (baseChain) return `${baseChain}.${expr.name}`;
    }
    return null;
  }

  // ── Indexing ─────────────────────────────────────────────────────────

  private evalIndex(expr: Extract<Expr, { type: "Index" }>): unknown {
    const base = this.evalExpr(expr.base);
    const indices = this.evalIndicesWithEnd(base, expr.indices);
    return this.rt.index(base, indices);
  }

  private evalIndexCell(expr: Extract<Expr, { type: "IndexCell" }>): unknown {
    const base = this.evalExpr(expr.base);
    const indices = this.evalIndicesWithEnd(base, expr.indices);
    return this.rt.indexCell(base, indices);
  }

  private evalIndicesWithEnd(base: unknown, indexExprs: Expr[]): unknown[] {
    const numIndices = indexExprs.length;
    return indexExprs.map((idx, dimIndex) => {
      this.endContextStack.push({ base, dimIndex, numIndices });
      try {
        return this.evalExpr(idx);
      } finally {
        this.endContextStack.pop();
      }
    });
  }

  // ── Anonymous functions ──────────────────────────────────────────────

  private evalAnonFunc(
    expr: Extract<Expr, { type: "AnonFunc" }>
  ): RuntimeValue {
    // Anonymous functions capture variable values at definition time (snapshot)
    const capturedEnv = this.env.snapshot();
    const capturedFile = this.currentFile;
    const capturedClassName = this.currentClassName;
    const capturedMethodName = this.currentMethodName;
    const bodyExpr = expr.body;
    const paramNames = expr.params;

    const fn = RTV.func("anonymous", "user");
    fn.jsFn = (nargoutArg: unknown, ...rest: unknown[]) => {
      const fnEnv = new Environment(capturedEnv);
      const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
      for (let i = 0; i < paramNames.length; i++) {
        if (i < actualArgs.length) {
          fnEnv.set(paramNames[i], ensureRuntimeValue(actualArgs[i]));
        }
      }
      const narg = typeof nargoutArg === "number" ? nargoutArg : 1;
      const savedEnv = this.env;
      this.env = fnEnv;
      return this.withFileContext(
        capturedFile,
        capturedClassName,
        capturedMethodName,
        () => {
          try {
            return this.evalExprNargout(bodyExpr, narg);
          } finally {
            this.env = savedEnv;
          }
        }
      );
    };
    fn.jsFnExpectsNargout = true;
    fn.nargin = paramNames.length;

    return fn;
  }

  // ── Function handles ─────────────────────────────────────────────────

  private makeFuncHandle(name: string): RuntimeValue {
    const capturedFile = this.currentFile;
    const capturedClassName = this.currentClassName;
    const capturedMethodName = this.currentMethodName;
    const capturedEnv = this.env;

    const fn = RTV.func(name, "builtin");
    fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
      const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
      const narg = typeof nargout === "number" ? nargout : 1;

      // Check if the handle refers to a nested function captured at creation time
      const nested = capturedEnv.getNestedFunction(name);
      if (nested) {
        return this.callNestedFunction(nested.fn, nested.env, actualArgs, narg);
      }

      return this.withFileContext(
        capturedFile,
        capturedClassName,
        capturedMethodName,
        () => this.callFunction(name, actualArgs, narg)
      );
    };
    fn.jsFnExpectsNargout = true;
    return fn;
  }

  // ── Tensor/Cell literal construction ─────────────────────────────────

  private evalTensorLiteral(expr: Extract<Expr, { type: "Tensor" }>): unknown {
    if (expr.rows.length === 0) {
      return RTV.tensor(new FloatXArray(0), [0, 0]);
    }

    const rowValues: RuntimeValue[] = [];
    for (const row of expr.rows) {
      // Evaluate elements, flattening CSL expansions (e.g., c{:})
      const vals: RuntimeValue[] = [];
      for (const e of row) {
        const v = this.evalExpr(e);
        if (Array.isArray(v)) {
          for (const elem of v) vals.push(ensureRuntimeValue(elem));
        } else {
          vals.push(ensureRuntimeValue(v));
        }
      }
      if (vals.length === 1) {
        rowValues.push(vals[0]);
      } else {
        rowValues.push(horzcat(...vals) as RuntimeValue);
      }
    }

    if (rowValues.length === 1) {
      return rowValues[0];
    }
    return vertcat(...rowValues);
  }

  private evalCellLiteral(expr: Extract<Expr, { type: "Cell" }>): unknown {
    if (expr.rows.length === 0) {
      return RTV.cell([], [0, 0]);
    }

    if (expr.rows.length === 1) {
      const elements: RuntimeValue[] = [];
      for (const e of expr.rows[0]) {
        const v = this.evalExpr(e);
        if (Array.isArray(v)) {
          for (const elem of v) elements.push(ensureRuntimeValue(elem));
        } else {
          elements.push(ensureRuntimeValue(v));
        }
      }
      return RTV.cell(elements, [1, elements.length]);
    }

    const numRows = expr.rows.length;
    const numCols = expr.rows[0].length;
    const elements: RuntimeValue[] = [];
    for (let c = 0; c < numCols; c++) {
      for (let r = 0; r < numRows; r++) {
        elements.push(ensureRuntimeValue(this.evalExpr(expr.rows[r][c])));
      }
    }
    return RTV.cell(elements, [numRows, numCols]);
  }

  // ── LValue assignment ────────────────────────────────────────────────

  private assignLValue(lv: LValue, value: RuntimeValue): void {
    switch (lv.type) {
      case "Var":
        this.env.set(lv.name, value);
        break;

      case "Ignore":
        break;

      case "Index": {
        const base =
          lv.base.type === "Ident"
            ? (this.env.get(lv.base.name) ??
              RTV.tensor(new FloatXArray(0), [0, 0]))
            : this.evalExpr(lv.base);
        const result = this.rt.indexStore(
          base,
          lv.indices.map(idx => this.evalExpr(idx)),
          value
        );
        if (lv.base.type === "Ident") {
          this.env.set(lv.base.name, ensureRuntimeValue(result));
        }
        break;
      }

      case "IndexCell": {
        const base =
          lv.base.type === "Ident"
            ? (this.env.get(lv.base.name) ?? RTV.cell([], [0, 0]))
            : this.evalExpr(lv.base);
        const result = this.rt.indexCellStore(
          base,
          lv.indices.map(idx => this.evalExpr(idx)),
          value
        );
        if (lv.base.type === "Ident") {
          this.env.set(lv.base.name, ensureRuntimeValue(result));
        }
        break;
      }

      case "Member": {
        const base = this.evalLValueBase(lv.base, RTV.struct({}));
        const result = this.rt.setMemberReturn(base, lv.name, value);
        this.writeLValueBase(lv.base, ensureRuntimeValue(result));
        break;
      }

      case "MemberDynamic": {
        const base = this.evalLValueBase(lv.base, RTV.struct({}));
        const nameVal = this.evalExpr(lv.nameExpr);
        const result = this.rt.setMemberDynamicReturn(base, nameVal, value);
        this.writeLValueBase(lv.base, ensureRuntimeValue(result));
        break;
      }
    }
  }

  /** Write a value back through an LValue base expression chain. */
  private writeLValueBase(base: Expr, value: RuntimeValue): void {
    if (base.type === "Ident") {
      this.env.set(base.name, value);
    } else if (base.type === "Member") {
      // Recursively: update the parent's field, then write back the parent
      const parentBase = this.evalLValueBase(base.base, RTV.struct({}));
      const updatedParent = this.rt.setMemberReturn(
        parentBase,
        base.name,
        value
      );
      this.writeLValueBase(base.base, ensureRuntimeValue(updatedParent));
    }
  }

  /**
   * Evaluate an LValue base expression with auto-creation for nested member chains.
   * When the base is an Ident that doesn't exist, returns `defaultVal`.
   * When the base is itself a Member chain (e.g., `config.db` in `config.db.host = x`),
   * recursively auto-creates intermediate structs.
   */
  private evalLValueBase(base: Expr, defaultVal: RuntimeValue): unknown {
    if (base.type === "Ident") {
      return this.env.get(base.name) ?? defaultVal;
    }
    if (base.type === "Member") {
      // Recursively get/create the parent
      const parentBase = this.evalLValueBase(base.base, RTV.struct({}));
      const parentRv = ensureRuntimeValue(parentBase);
      // Try to get the field; if not present, return empty struct
      try {
        return this.rt.getMember(parentBase, base.name);
      } catch {
        // Field doesn't exist — create it and update parent
        const newStruct = RTV.struct({});
        const updatedParent = this.rt.setMemberReturn(
          parentRv,
          base.name,
          newStruct
        );
        // Write back the updated parent
        if (base.base.type === "Ident") {
          this.env.set(base.base.name, ensureRuntimeValue(updatedParent));
        }
        return newStruct;
      }
    }
    return this.evalExpr(base);
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  private switchMatch(switchVal: unknown, caseVal: unknown): boolean {
    const sv = ensureRuntimeValue(switchVal);
    const cv = ensureRuntimeValue(caseVal);

    if (isRuntimeCell(cv)) {
      for (const el of cv.data) {
        if (this.switchMatch(switchVal, el)) return true;
      }
      return false;
    }

    if (isRuntimeNumber(sv) && isRuntimeNumber(cv)) {
      return sv === cv;
    }

    if (
      (isRuntimeChar(sv) || isRuntimeString(sv)) &&
      (isRuntimeChar(cv) || isRuntimeString(cv))
    ) {
      const a = isRuntimeChar(sv) ? sv.value : sv;
      const b = isRuntimeChar(cv) ? cv.value : cv;
      return a === b;
    }

    return false;
  }

  private isOutputExpr(expr: Expr): boolean {
    if (expr.type !== "FuncCall") return false;
    const outputFunctions = [
      "disp",
      "display",
      "fprintf",
      "warning",
      "assert",
      "tic",
    ];
    return outputFunctions.includes(expr.name);
  }
}

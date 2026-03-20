/**
 * AST-walking interpreter for MATLAB code.
 *
 * Walks the parsed AST directly without lowering or codegen.
 * All dispatch decisions are made at runtime using actual values.
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
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
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
import type { WorkspaceFile } from "../workspace/index.js";
import { parseMFile } from "../parser/index.js";
import { toNumber } from "../runtime/convert.js";
import { numel } from "../runtime/utils.js";

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
  constructor(private parent?: Environment) {}

  get(name: string): RuntimeValue | undefined {
    return this.vars.get(name) ?? this.parent?.get(name);
  }

  set(name: string, value: RuntimeValue): void {
    this.vars.set(name, value);
  }

  has(name: string): boolean {
    return this.vars.has(name) || (this.parent?.has(name) ?? false);
  }

  /** Get all variable names defined in this scope (not parent). */
  localNames(): string[] {
    return [...this.vars.keys()];
  }

  /** Get all variables as a record (for ExecResult). */
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
  private localFunctions = new Map<string, FunctionDef>();
  private workspaceFunctionCache = new Map<string, FunctionDef>();
  private workspaceFiles: WorkspaceFile[];
  private workspaceASTCache = new Map<string, AbstractSyntaxTree>();
  public ans: RuntimeValue | undefined;
  /** Stack of [base, dimIndex, numIndices] for resolving `end` keyword in indexing. */
  private endContextStack: Array<{
    base: unknown;
    dimIndex: number;
    numIndices: number;
  }> = [];

  constructor(
    private rt: Runtime,
    workspaceFiles?: WorkspaceFile[],
    initialVariableValues?: Record<string, RuntimeValue>
  ) {
    this.env = new Environment();
    this.workspaceFiles = workspaceFiles ?? [];
    if (initialVariableValues) {
      for (const [name, value] of Object.entries(initialVariableValues)) {
        this.env.set(name, value);
      }
    }
  }

  /** Run a complete AST (main script). */
  run(ast: AbstractSyntaxTree): void {
    // First pass: collect local function definitions
    for (const stmt of ast.body) {
      if (stmt.type === "Function") {
        this.localFunctions.set(stmt.name, {
          name: stmt.name,
          params: stmt.params,
          outputs: stmt.outputs,
          body: stmt.body,
        });
      }
    }

    // Second pass: execute non-function statements
    const nonFuncStmts = ast.body.filter(s => s.type !== "Function");
    if (nonFuncStmts.length === 0 && this.localFunctions.size > 0) {
      // Function file: call the first function with 0 args
      const firstFn = this.localFunctions.values().next().value;
      if (firstFn) {
        this.callUserFunction(firstFn, [], 0);
      }
    } else {
      for (const stmt of nonFuncStmts) {
        const signal = this.execStmt(stmt);
        if (signal) break; // Return at script level ends execution
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
        const rv = ensureRuntimeValue(val);
        this.ans = rv;
        if (!stmt.suppressed && !this.isOutputExpr(stmt.expr)) {
          this.rt.displayResult(rv);
        }
        return null;
      }

      case "Assign": {
        const val = this.evalExpr(stmt.expr);
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
          // Display first output as ans
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
          // Iterate over columns
          const shape = rv.shape;
          if (shape.length === 2 && shape[0] === 1) {
            // Row vector: iterate elements
            for (let i = 0; i < rv.data.length; i++) {
              this.env.set(stmt.varName, rv.data[i] as unknown as RuntimeValue);
              const signal = this.execStmts(stmt.body);
              if (signal instanceof BreakSignal) break;
              if (signal instanceof ContinueSignal) continue;
              if (signal instanceof ReturnSignal) return signal;
            }
          } else {
            // Matrix: iterate columns
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
          // Iterate over cell elements
          for (let i = 0; i < rv.data.length; i++) {
            this.env.set(stmt.varName, rv.data[i]);
            const signal = this.execStmts(stmt.body);
            if (signal instanceof BreakSignal) break;
            if (signal instanceof ContinueSignal) continue;
            if (signal instanceof ReturnSignal) return signal;
          }
        } else if (isRuntimeNumber(rv)) {
          // Scalar: single iteration
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
            // Create MException-like struct
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

      case "Function":
        // Already collected in first pass
        return null;

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
        // TODO: implement class definitions
        throw new RuntimeError("Interpreter does not yet support classdef");

      case "Import":
        // TODO: implement imports
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
        // Strip surrounding quotes and unescape
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
        // Check variable first
        const val = this.env.get(expr.name);
        if (val !== undefined) return val;

        // Check if it's a constant (pi, inf, etc.)
        try {
          return this.rt.getConstant(expr.name);
        } catch {
          // Not a constant
        }

        // Check if it's a 0-arg function call (MATLAB ambiguity)
        return this.callFunction(expr.name, [], nargout);
      }

      case "ImagUnit":
        return RTV.complex(0, 1);

      case "EndKeyword": {
        // Resolve `end` to the actual dimension size if inside an indexing context
        const ctx = this.endContextStack[this.endContextStack.length - 1];
        if (ctx) {
          const rv = ensureRuntimeValue(ctx.base);
          if (isRuntimeTensor(rv)) {
            if (ctx.numIndices === 1) {
              // Linear indexing: end = numel
              return numel(rv.shape);
            }
            // Dimensional indexing: end = size along this dimension
            return ctx.dimIndex < rv.shape.length ? rv.shape[ctx.dimIndex] : 1;
          }
          if (isRuntimeCell(rv)) {
            if (ctx.numIndices === 1) {
              return numel(rv.shape);
            }
            return ctx.dimIndex < rv.shape.length ? rv.shape[ctx.dimIndex] : 1;
          }
          if (isRuntimeChar(rv)) {
            return rv.value.length;
          }
          if (isRuntimeString(rv)) {
            return (rv as string).length;
          }
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
        return this.evalMember(expr);

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
        const base = this.evalExpr(expr.base);
        const args = expr.args.map(a => this.evalExpr(a));
        return this.rt.methodDispatch(expr.name, nargout, [base, ...args]);
      }

      case "SuperMethodCall": {
        const args = expr.args.map(a => this.evalExpr(a));
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
        const args = expr.args.map(a => this.evalExpr(a));
        return this.rt.callClassMethod(
          expr.className,
          expr.className,
          nargout,
          args
        );
      }

      case "MetaClass":
        throw new RuntimeError("Interpreter does not yet support meta.class");
    }
  }

  // ── Binary operators ─────────────────────────────────────────────────

  private evalBinary(expr: Extract<Expr, { type: "Binary" }>): unknown {
    // Short-circuit for logical operators
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

    // Check for class operator overloading
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
    // Check if it's variable indexing first (before evaluating args)
    // so that `end` can resolve properly.
    const varVal = this.env.get(expr.name);
    if (varVal !== undefined) {
      const rv = ensureRuntimeValue(varVal);
      if (isRuntimeFunction(rv)) {
        const args = expr.args.map(a => this.evalExpr(a));
        return this.rt.index(rv, args, nargout);
      }
      // Variable indexing: evaluate args with end-context
      const args = this.evalIndicesWithEnd(varVal, expr.args);
      return this.rt.index(varVal, args, nargout);
    }

    const args = expr.args.map(a => this.evalExpr(a));
    return this.callFunction(expr.name, args, nargout);
  }

  private callFunction(
    name: string,
    args: unknown[],
    nargout: number
  ): unknown {
    // 0. Intrinsics that the interpreter handles directly
    if (name === "isa" && args.length === 2) {
      return this.rt.isa(args[0], args[1]);
    }
    if (name === "__inferred_type_str" && args.length === 1) {
      // In interpreter mode, there's no compile-time type — return the runtime type
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

    // 1. Check if any arg is a class instance → class method dispatch
    for (const arg of args) {
      const rv = ensureRuntimeValue(arg);
      if (isRuntimeClassInstance(rv)) {
        // Try to dispatch as a class method
        try {
          return this.rt.callClassMethod(rv.className, name, nargout, args);
        } catch {
          // Not a class method, fall through
        }
      }
    }

    // 2. Local functions (defined in the same script)
    const localFn = this.localFunctions.get(name);
    if (localFn) {
      return this.callUserFunction(localFn, args, nargout);
    }

    // 3. Workspace functions
    const wsFn = this.resolveWorkspaceFunction(name);
    if (wsFn) {
      return this.callUserFunction(wsFn, args, nargout);
    }

    // 4. Builtins
    if (this.rt.builtins[name]) {
      const result = this.rt.builtins[name](nargout, args);
      return result;
    }

    throw new RuntimeError(`Undefined function or variable '${name}'`);
  }

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
    // Store nargin/nargout so the function body can access them
    fnEnv.set("$nargin", args.length as unknown as RuntimeValue);
    fnEnv.set("$nargout", nargout as unknown as RuntimeValue);

    // Create a new interpreter scope for the function
    const savedEnv = this.env;
    this.env = fnEnv;

    try {
      const signal = this.execStmts(fn.body);

      // Collect output values
      const outputs: RuntimeValue[] = [];
      if (signal instanceof ReturnSignal) {
        // Return was explicit; collect output variables
        for (let i = 0; i < Math.min(fn.outputs.length, nargout); i++) {
          const val = this.env.get(fn.outputs[i]);
          outputs.push(val ?? RTV.num(0));
        }
      } else {
        // Normal completion; collect output variables
        for (let i = 0; i < Math.min(fn.outputs.length, nargout); i++) {
          const val = this.env.get(fn.outputs[i]);
          outputs.push(val ?? RTV.num(0));
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

  // ── Workspace function resolution ────────────────────────────────────

  private resolveWorkspaceFunction(name: string): FunctionDef | null {
    // Check cache
    const cached = this.workspaceFunctionCache.get(name);
    if (cached) return cached;

    // Search workspace files
    for (const wf of this.workspaceFiles) {
      if (wf.name === name + ".m" || wf.name.endsWith("/" + name + ".m")) {
        const ast = this.parseWorkspaceFile(wf);
        if (!ast) continue;
        // Find the function with the matching name
        for (const stmt of ast.body) {
          if (stmt.type === "Function" && stmt.name === name) {
            const fn: FunctionDef = {
              name: stmt.name,
              params: stmt.params,
              outputs: stmt.outputs,
              body: stmt.body,
            };
            this.workspaceFunctionCache.set(name, fn);
            return fn;
          }
        }
        // If the file is a script (no function declarations), the main body is the function
        // This handles .m files that are just scripts
      }
    }
    return null;
  }

  private parseWorkspaceFile(wf: WorkspaceFile): AbstractSyntaxTree | null {
    const key = wf.name;
    const cached = this.workspaceASTCache.get(key);
    if (cached) return cached;

    try {
      const ast = parseMFile(wf.source, wf.name);
      this.workspaceASTCache.set(key, ast);
      return ast;
    } catch {
      return null;
    }
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

  /** Evaluate index expressions with `end` resolved for the given base. */
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

  // ── Member access ────────────────────────────────────────────────────

  private evalMember(expr: Extract<Expr, { type: "Member" }>): unknown {
    const base = this.evalExpr(expr.base);
    return this.rt.getMember(base, expr.name);
  }

  // ── Anonymous functions ──────────────────────────────────────────────

  private evalAnonFunc(
    expr: Extract<Expr, { type: "AnonFunc" }>
  ): RuntimeValue {
    // Capture current environment for closure
    const capturedEnv = this.env;
    const bodyExpr = expr.body;
    const paramNames = expr.params;

    const fn = RTV.func("anonymous", "user");
    fn.jsFn = this.makeAnonJsFn(capturedEnv, paramNames, bodyExpr);
    fn.jsFnExpectsNargout = true;
    fn.nargin = paramNames.length;

    return fn;
  }

  private makeAnonJsFn(
    capturedEnv: Environment,
    paramNames: string[],
    bodyExpr: Expr
  ): (...args: unknown[]) => unknown {
    return (_nargout: unknown, ...rest: unknown[]) => {
      const fnEnv = new Environment(capturedEnv);
      const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
      for (let i = 0; i < paramNames.length; i++) {
        if (i < actualArgs.length) {
          fnEnv.set(paramNames[i], ensureRuntimeValue(actualArgs[i]));
        }
      }
      const savedEnv = this.env;
      this.env = fnEnv;
      try {
        return this.evalExpr(bodyExpr);
      } finally {
        this.env = savedEnv;
      }
    };
  }

  // ── Function handles ─────────────────────────────────────────────────

  private makeFuncHandle(name: string): RuntimeValue {
    const fn = RTV.func(name, "builtin");
    fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
      const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
      return this.callFunction(
        name,
        actualArgs,
        typeof nargout === "number" ? nargout : 1
      );
    };
    fn.jsFnExpectsNargout = true;
    return fn;
  }

  // ── Tensor/Cell literal construction ─────────────────────────────────

  private evalTensorLiteral(expr: Extract<Expr, { type: "Tensor" }>): unknown {
    if (expr.rows.length === 0) {
      // Empty matrix []
      return RTV.tensor(new FloatXArray(0), [0, 0]);
    }

    // Evaluate each row as a list of values, then horzcat within rows and vertcat across rows
    const rowValues: RuntimeValue[] = [];
    for (const row of expr.rows) {
      const vals = row.map(e => ensureRuntimeValue(this.evalExpr(e)));
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
      // Single row: {a, b, c} → [1, N]
      const elements: RuntimeValue[] = [];
      for (const e of expr.rows[0]) {
        elements.push(ensureRuntimeValue(this.evalExpr(e)));
      }
      return RTV.cell(elements, [1, elements.length]);
    }

    // Multiple rows: {a; b; c} → [N, 1] or {a, b; c, d} → [rows, cols]
    const numRows = expr.rows.length;
    const numCols = expr.rows[0].length;
    const elements: RuntimeValue[] = [];
    // Column-major order
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
        const base =
          lv.base.type === "Ident"
            ? (this.env.get(lv.base.name) ?? RTV.struct({}))
            : this.evalExpr(lv.base);
        const result = this.rt.setMemberReturn(base, lv.name, value);
        if (lv.base.type === "Ident") {
          this.env.set(lv.base.name, ensureRuntimeValue(result));
        }
        break;
      }

      case "MemberDynamic": {
        const base =
          lv.base.type === "Ident"
            ? (this.env.get(lv.base.name) ?? RTV.struct({}))
            : this.evalExpr(lv.base);
        const nameVal = this.evalExpr(lv.nameExpr);
        const result = this.rt.setMemberDynamicReturn(base, nameVal, value);
        if (lv.base.type === "Ident") {
          this.env.set(lv.base.name, ensureRuntimeValue(result));
        }
        break;
      }
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  private switchMatch(switchVal: unknown, caseVal: unknown): boolean {
    const sv = ensureRuntimeValue(switchVal);
    const cv = ensureRuntimeValue(caseVal);

    // Cell case: match any element
    if (isRuntimeCell(cv)) {
      for (const el of cv.data) {
        if (this.switchMatch(switchVal, el)) return true;
      }
      return false;
    }

    // Numeric comparison
    if (isRuntimeNumber(sv) && isRuntimeNumber(cv)) {
      return sv === cv;
    }

    // String/char comparison
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

  /** Check if an expression is an output function call (disp, fprintf, etc.)
   *  whose return value should not be displayed as "ans". */
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

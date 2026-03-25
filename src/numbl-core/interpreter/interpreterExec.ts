/**
 * Interpreter statement execution and expression evaluation methods.
 * Augments the Interpreter class via prototype assignment.
 */

import type { Stmt, Expr, LValue } from "../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../parser/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeFunction,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  FloatXArray,
} from "../runtime/types.js";
import { RTV, getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { binop, uplus } from "../runtime/runtimeOperators.js";
import { mPow } from "../builtins/arithmetic.js";
import { getBuiltinNargin } from "../builtins/registry.js";
import { getConstant } from "../builtins/constants.js";
import { buildLineTable, offsetToLineFast } from "../runtime/error.js";
import { COLON_SENTINEL, END_SENTINEL } from "../executor/types.js";
import { numel } from "../runtime/utils.js";
import {
  forIter,
  switchMatch as runtimeSwitchMatch,
  range as runtimeRange,
} from "../runtime/runtimeOperators.js";
import type { ResolvedTarget } from "../functionResolve.js";

import {
  BreakSignal,
  ContinueSignal,
  ReturnSignal,
  Environment,
  funcDefFromStmt,
  type ControlSignal,
} from "./types.js";

import type { Interpreter } from "./interpreter.js";
import { tryJitFor, tryJitWhile } from "./jit/jitLoop.js";

// ── Statement execution ──────────────────────────────────────────────────

export function execStmt(this: Interpreter, stmt: Stmt): ControlSignal | null {
  if (stmt.span) {
    this.rt.$file = stmt.span.file;
    // Compute line number from character offset using cached line table
    let table = this.lineTableCache.get(stmt.span.file);
    if (!table) {
      const src = this.fileSources.get(stmt.span.file) ?? "";
      table = buildLineTable(src);
      this.lineTableCache.set(stmt.span.file, table);
    }
    this.rt.$line = offsetToLineFast(table, stmt.span.start);
  }

  switch (stmt.type) {
    case "ExprStmt": {
      const val = this.evalExpr(stmt.expr);
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
      // Detect [c{idx}] = func() pattern for multi-output cell assign
      if (stmt.lvalues.length === 1 && stmt.lvalues[0].type === "IndexCell") {
        const lv = stmt.lvalues[0];
        const cellBase =
          lv.base.type === "Ident"
            ? (this.env.get(lv.base.name) ?? RTV.cell([], [0, 0]))
            : this.evalExpr(lv.base);
        const indices = lv.indices.map(idx => this.evalExpr(idx));
        // Determine nargout from index count
        const idxVal = ensureRuntimeValue(indices[0]);
        let expandedCount = 1;
        if (isRuntimeTensor(idxVal)) {
          expandedCount = idxVal.data.length;
        } else if (typeof idxVal === "number") {
          expandedCount = 1;
        }
        const val = this.evalExprNargout(stmt.expr, expandedCount);
        const values = Array.isArray(val) ? val : [val];
        const result = this.rt.multiOutputCellAssign(
          cellBase,
          indices[0],
          values.map(v => ensureRuntimeValue(v))
        );
        if (lv.base.type === "Ident") {
          this.env.set(lv.base.name, ensureRuntimeValue(result));
        }
        return null;
      }

      const nargout = stmt.lvalues.length;
      const val = this.evalExprNargout(stmt.expr, nargout);
      const values = Array.isArray(val) ? val : [val];
      for (let i = 0; i < stmt.lvalues.length; i++) {
        const lv = stmt.lvalues[i];
        if (lv.type === "Ignore") continue;
        const rv = this.rt.share(i < values.length ? values[i] : undefined);
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
      if (this.optimization >= 1 && tryJitWhile(this, stmt)) return null;
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
      if (this.optimization >= 1 && tryJitFor(this, stmt)) return null;
      const iterVal = this.evalExpr(stmt.expr);
      const rv = ensureRuntimeValue(iterVal);
      const iterItems = forIter(rv);
      for (let _i = 0; _i < iterItems.length; _i++) {
        this.env.set(stmt.varName, ensureRuntimeValue(iterItems[_i]));
        const signal = this.execStmts(stmt.body);
        if (signal instanceof BreakSignal) break;
        if (signal instanceof ContinueSignal) continue;
        if (signal instanceof ReturnSignal) return signal;
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
        if (stmt.catchVar) {
          this.env.set(stmt.catchVar, this.rt.wrapError(e));
        }
        const signal = this.execStmts(stmt.catchBody);
        if (signal) return signal;
      }
      return null;
    }

    case "Function": {
      this.env.nestedFunctions.set(stmt.name, {
        fn: funcDefFromStmt(stmt),
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

    case "Global": {
      for (const name of stmt.names) {
        this.env.globalNames.add(name);
      }
      return null;
    }

    case "Persistent": {
      const funcId = this.env.persistentFuncId;
      if (funcId) {
        for (const name of stmt.names) {
          this.env.persistentNames.add(name);
          const val = this.rt.getPersistent(funcId, name);
          this.env.setLocal(name, val);
        }
      }
      return null;
    }

    case "ClassDef":
      return null;

    case "Import":
      return null;
  }
}

export function execStmts(
  this: Interpreter,
  stmts: Stmt[]
): ControlSignal | null {
  for (const stmt of stmts) {
    const signal = this.execStmt(stmt);
    if (signal) return signal;
  }
  return null;
}

// ── Expression evaluation ────────────────────────────────────────────────

export function evalExpr(this: Interpreter, expr: Expr): unknown {
  return this.evalExprNargout(expr, 1);
}

export function evalExprNargout(
  this: Interpreter,
  expr: Expr,
  nargout: number
): unknown {
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
        // Class instances: call overloaded end(obj, k, n) if available
        if (isRuntimeClassInstance(rv)) {
          return this.rt.dispatch("end", 1, [
            rv,
            ctx.dimIndex + 1,
            ctx.numIndices,
          ]);
        }
        if (isRuntimeStructArray(rv)) {
          return rv.elements.length;
        }
        if (isRuntimeSparseMatrix(rv)) {
          if (ctx.numIndices === 1) return rv.m * rv.n;
          return ctx.dimIndex === 0 ? rv.m : ctx.dimIndex === 1 ? rv.n : 1;
        }
        // Scalars (number, boolean, complex) — default to 1
        return 1;
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
      const dottedBase = this.tryExtractDottedName(expr.base);
      if (dottedBase && !this.env.has(dottedBase.split(".")[0])) {
        const args = this.evalArgs(expr.args);
        const qualifiedName = `${dottedBase}.${expr.name}`;
        if (this.functionIndex.workspaceFunctions.has(qualifiedName)) {
          return this.callFunction(qualifiedName, args, nargout);
        }
        if (this.functionIndex.workspaceClasses.has(qualifiedName)) {
          return this.instantiateClass(qualifiedName, args, nargout);
        }
        if (
          this.functionIndex.classStaticMethods.get(dottedBase)?.has(expr.name)
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
        if (this.functionIndex.workspaceClasses.has(dottedBase)) {
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
      // For struct/class fields, try to get the field value for end-context resolution.
      // For class instances, read the field directly (not via getMember which could
      // trigger subsref or call a no-arg method as a side effect).
      let fieldVal: unknown = undefined;
      const baseRv = ensureRuntimeValue(base);
      if (isRuntimeClassInstance(baseRv)) {
        const fv = baseRv.fields.get(expr.name);
        if (fv !== undefined) fieldVal = fv;
      } else {
        try {
          fieldVal = this.rt.getMember(base, expr.name);
        } catch {
          // Not a struct field
        }
      }
      const args =
        fieldVal !== undefined
          ? this.evalIndicesWithEnd(fieldVal, expr.args)
          : this.evalArgs(expr.args);
      return this.rt.methodDispatch(expr.name, nargout, [base, ...args]);
    }

    case "SuperMethodCall": {
      const args = this.evalArgs(expr.args);
      const objVal = this.env.get(expr.methodName);
      if (objVal !== undefined) {
        const superClassInfo = this.ctx.getClassInfo(expr.superClassName);
        if (superClassInfo && superClassInfo.constructorName) {
          const superResult = this.interpretConstructor(
            superClassInfo,
            [objVal, ...args],
            1
          );
          return this.rt.callSuperConstructor(objVal, superResult);
        }
        // Built-in super class (e.g. classdef Foo < double):
        // pass the raw arg to callSuperConstructor which stores it as _builtinData
        const builtinSuperNames = new Set([
          "double",
          "single",
          "int8",
          "int16",
          "int32",
          "int64",
          "uint8",
          "uint16",
          "uint32",
          "uint64",
          "logical",
          "char",
        ]);
        if (builtinSuperNames.has(expr.superClassName)) {
          const data =
            args.length > 0
              ? ensureRuntimeValue(args[0])
              : RTV.tensor(new FloatXArray(0), [0, 0]);
          return this.rt.callSuperConstructor(objVal, data);
        }
        const { propertyNames, propertyDefaults } = this.collectClassProperties(
          superClassInfo ??
            ({
              propertyNames: [],
              propertyDefaults: new Map(),
              superClass: null,
            } as never)
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

export function evalArgs(this: Interpreter, argExprs: Expr[]): unknown[] {
  const args: unknown[] = [];
  for (const a of argExprs) {
    const val = this.evalExpr(a);
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

// ── Binary operators ─────────────────────────────────────────────────────

export function evalBinary(
  this: Interpreter,
  expr: Extract<Expr, { type: "Binary" }>
): unknown {
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

  if (
    (expr.op === BinaryOperation.Pow || expr.op === BinaryOperation.ElemPow) &&
    typeof left === "number" &&
    typeof right === "number" &&
    left < 0
  ) {
    const r = Math.pow(left, right);
    if (!isNaN(r)) return r;
    return mPow(ensureRuntimeValue(left), ensureRuntimeValue(right));
  }

  return binop(expr.op, left, right);
}

// ── Unary operators ──────────────────────────────────────────────────────

export function evalUnary(
  this: Interpreter,
  expr: Extract<Expr, { type: "Unary" }>
): unknown {
  const operand = this.evalExpr(expr.operand);
  switch (expr.op) {
    case UnaryOperation.Plus:
      return uplus(operand);
    case UnaryOperation.Minus:
      return this.rt.uminus(operand);
    case UnaryOperation.Not: {
      if (typeof operand === "number") return RTV.logical(operand === 0);
      return this.rt.not(operand);
    }
    case UnaryOperation.Transpose:
      // ' = conjugate transpose
      return this.rt.ctranspose(operand);
    case UnaryOperation.NonConjugateTranspose:
      // .' = non-conjugate transpose
      return this.rt.transpose(operand);
  }
}

// ── Range ────────────────────────────────────────────────────────────────

export function evalRange(
  this: Interpreter,
  expr: Extract<Expr, { type: "Range" }>
): unknown {
  const startVal = this.evalExpr(expr.start);
  const endVal = this.evalExpr(expr.end);
  const stepVal = expr.step ? this.evalExpr(expr.step) : 1;
  return runtimeRange(startVal, stepVal, endVal);
}

// ── Function calls ───────────────────────────────────────────────────────

export function evalFuncCall(
  this: Interpreter,
  expr: Extract<Expr, { type: "FuncCall" }>,
  nargout: number
): unknown {
  const varVal = this.env.get(expr.name);
  if (varVal !== undefined) {
    const rv = ensureRuntimeValue(varVal);
    if (isRuntimeFunction(rv)) {
      const args = this.evalArgs(expr.args);
      return this.rt.index(rv, args, nargout);
    }
    const args = this.evalIndicesWithEnd(varVal, expr.args);
    // Inside class methods, bypass overloaded subsref for same-class instances
    let skipSubsref: boolean | string = false;
    if (
      this.currentClassName &&
      isRuntimeClassInstance(rv) &&
      rv.className === this.currentClassName
    ) {
      skipSubsref = true;
    }
    return this.rt.index(varVal, args, nargout, skipSubsref);
  }
  const args = this.evalArgs(expr.args);
  // Constant called as zero-arg function? e.g. eps(), pi(), inf()
  if (args.length === 0) {
    const c = getConstant(expr.name);
    if (c !== undefined) return c;
  }
  return this.callFunction(expr.name, args, nargout);
}

// ── Member access ────────────────────────────────────────────────────────

export function evalMember(
  this: Interpreter,
  expr: Extract<Expr, { type: "Member" }>,
  nargout: number
): unknown {
  const dottedName = this.tryExtractDottedName(expr);
  if (dottedName) {
    const rootName = dottedName.split(".")[0];
    if (!this.env.has(rootName)) {
      if (this.functionIndex.workspaceFunctions.has(dottedName)) {
        return this.callFunction(dottedName, [], nargout);
      }
      if (this.functionIndex.workspaceClasses.has(dottedName)) {
        return this.instantiateClass(dottedName, [], nargout);
      }
      const lastDot = dottedName.lastIndexOf(".");
      if (lastDot > 0) {
        const prefix = dottedName.slice(0, lastDot);
        const methodName = dottedName.slice(lastDot + 1);
        if (
          this.functionIndex.classStaticMethods.get(prefix)?.has(methodName)
        ) {
          const target: ResolvedTarget = {
            kind: "classMethod",
            className: prefix,
            methodName,
            compileArgTypes: [],
            stripInstance: false,
          };
          return this.interpretTarget(target, [], nargout);
        }
      }
    }
  }
  const base = this.evalExpr(expr.base);
  return this.rt.getMember(base, expr.name);
}

export function tryExtractDottedName(
  this: Interpreter,
  expr: Expr
): string | null {
  if (expr.type === "Ident") return expr.name;
  if (expr.type === "Member") {
    const baseChain = this.tryExtractDottedName(expr.base);
    if (baseChain) return `${baseChain}.${expr.name}`;
  }
  return null;
}

// ── Indexing ─────────────────────────────────────────────────────────────

export function evalIndex(
  this: Interpreter,
  expr: Extract<Expr, { type: "Index" }>
): unknown {
  const base = this.evalExpr(expr.base);
  const indices = this.evalIndicesWithEnd(base, expr.indices);
  // Inside class methods, bypass overloaded subsref for same-class instances
  let skipSubsref: boolean | string = false;
  if (this.currentClassName) {
    const baseRv = ensureRuntimeValue(base);
    if (
      isRuntimeClassInstance(baseRv) &&
      baseRv.className === this.currentClassName
    ) {
      skipSubsref = true;
    }
  }
  return this.rt.index(base, indices, 1, skipSubsref);
}

export function evalIndexCell(
  this: Interpreter,
  expr: Extract<Expr, { type: "IndexCell" }>
): unknown {
  const base = this.evalExpr(expr.base);
  const indices = this.evalIndicesWithEnd(base, expr.indices);
  return this.rt.indexCell(base, indices);
}

export function evalIndicesWithEnd(
  this: Interpreter,
  base: unknown,
  indexExprs: Expr[]
): unknown[] {
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

// ── Anonymous functions ──────────────────────────────────────────────────

export function evalAnonFunc(
  this: Interpreter,
  expr: Extract<Expr, { type: "AnonFunc" }>
): RuntimeValue {
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

// ── Function handles ─────────────────────────────────────────────────────

export function makeFuncHandle(this: Interpreter, name: string): RuntimeValue {
  // Handle dotted names like @ClassName.method (static method handles)
  const dotIdx = name.indexOf(".");
  if (dotIdx > 0) {
    const className = name.slice(0, dotIdx);
    const methodName = name.slice(dotIdx + 1);
    const fn = RTV.func(name, "builtin");
    fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
      const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
      const narg = typeof nargout === "number" ? nargout : 1;
      const target: ResolvedTarget = {
        kind: "classMethod",
        className,
        methodName,
        compileArgTypes: actualArgs.map(a =>
          getItemTypeFromRuntimeValue(ensureRuntimeValue(a))
        ),
        stripInstance: false,
      };
      return this.interpretTarget(target, actualArgs, narg);
    };
    fn.jsFnExpectsNargout = true;
    return fn;
  }

  const capturedFile = this.currentFile;
  const capturedClassName = this.currentClassName;
  const capturedMethodName = this.currentMethodName;
  const capturedEnv = this.env;

  const fn = RTV.func(name, "builtin");
  fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
    const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
    const narg = typeof nargout === "number" ? nargout : 1;
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
  // Populate nargin for builtin handles (e.g. nargin(@sin) == 1)
  const narg = getBuiltinNargin(name);
  if (narg !== undefined) fn.nargin = narg;
  return fn;
}

// ── Tensor/Cell literal construction ─────────────────────────────────────

export function evalTensorLiteral(
  this: Interpreter,
  expr: Extract<Expr, { type: "Tensor" }>
): unknown {
  if (expr.rows.length === 0) {
    return RTV.tensor(new FloatXArray(0), [0, 0]);
  }
  const rowValues: unknown[] = [];
  for (const row of expr.rows) {
    const vals: unknown[] = [];
    for (const e of row) {
      const v = this.evalExpr(e);
      if (Array.isArray(v)) {
        for (const elem of v) vals.push(elem);
      } else {
        vals.push(v);
      }
    }
    if (vals.length === 1) {
      rowValues.push(vals[0]);
    } else {
      rowValues.push(this.rt.horzcat(vals));
    }
  }
  if (rowValues.length === 1) {
    return rowValues[0];
  }
  return this.rt.vertcat(rowValues);
}

export function evalCellLiteral(
  this: Interpreter,
  expr: Extract<Expr, { type: "Cell" }>
): unknown {
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

// ── LValue assignment ────────────────────────────────────────────────────

export function assignLValue(
  this: Interpreter,
  lv: LValue,
  value: RuntimeValue
): void {
  switch (lv.type) {
    case "Var":
      this.env.set(lv.name, value);
      break;

    case "Ignore":
      break;

    case "Index": {
      // Use undefined for uninitialized Ident so indexStore can detect it
      // (e.g. h(1) = classInstance when h is an uninitialized output var)
      const base =
        lv.base.type === "Ident"
          ? this.env.get(lv.base.name)
          : this.evalLValueBase(
              lv.base,
              RTV.tensor(new FloatXArray(0), [0, 0])
            );
      const indices = this.evalIndicesWithEnd(
        base ?? RTV.tensor(new FloatXArray(0), [0, 0]),
        lv.indices
      );
      // Inside class methods, bypass overloaded subsasgn for same-class instances
      let skipSubsasgn = false;
      if (this.currentClassName && base != null) {
        const baseRv = ensureRuntimeValue(base);
        if (
          isRuntimeClassInstance(baseRv) &&
          baseRv.className === this.currentClassName
        ) {
          skipSubsasgn = true;
        }
      }
      const result = this.rt.indexStore(base, indices, value, skipSubsasgn);
      if (lv.base.type === "Ident") {
        this.env.set(lv.base.name, ensureRuntimeValue(result));
      } else {
        this.writeLValueBase(lv.base, ensureRuntimeValue(result));
      }
      break;
    }

    case "IndexCell": {
      const base =
        lv.base.type === "Ident"
          ? (this.env.get(lv.base.name) ?? RTV.cell([], [0, 0]))
          : this.evalLValueBase(lv.base, RTV.cell([], [0, 0]));
      const indices = this.evalIndicesWithEnd(base, lv.indices);
      const result = this.rt.indexCellStore(base, indices, value);
      if (lv.base.type === "Ident") {
        this.env.set(lv.base.name, ensureRuntimeValue(result));
      } else {
        this.writeLValueBase(lv.base, ensureRuntimeValue(result));
      }
      break;
    }

    case "Member": {
      // Walk up the Member chain to find the first non-Member node and collect names
      const names: string[] = [lv.name];
      let cursor: Expr = lv.base;
      while (cursor.type === "Member") {
        names.unshift(cursor.name);
        cursor = cursor.base;
      }
      // cursor is now the root of the member chain (Ident, Index, etc.)
      const rootBase =
        cursor.type === "Ident"
          ? (this.env.get(cursor.name) ?? RTV.struct({}))
          : this.evalLValueBase(cursor, RTV.struct({}));
      const rootRv = ensureRuntimeValue(rootBase);
      if (isRuntimeClassInstance(rootRv)) {
        // Use memberChainAssign which routes through subsasgn if needed
        const result = this.rt.memberChainAssign(rootBase, names, value);
        if (cursor.type === "Ident") {
          this.env.set(cursor.name, ensureRuntimeValue(result));
        } else {
          this.writeLValueBase(cursor, ensureRuntimeValue(result));
        }
      } else {
        // Non-class: use direct field set with store-back chain
        const base = this.evalLValueBase(lv.base, RTV.struct({}));
        const result = this.rt.setMemberReturn(base, lv.name, value);
        this.writeLValueBase(lv.base, ensureRuntimeValue(result));
      }
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

export function writeLValueBase(
  this: Interpreter,
  base: Expr,
  value: RuntimeValue
): void {
  if (base.type === "Ident") {
    this.env.set(base.name, value);
  } else if (base.type === "Member") {
    const parentBase = this.evalLValueBase(base.base, RTV.struct({}));
    const updatedParent = this.rt.setMemberReturn(parentBase, base.name, value);
    this.writeLValueBase(base.base, ensureRuntimeValue(updatedParent));
  } else if (base.type === "Index") {
    const baseVal = this.evalLValueBase(
      base.base,
      RTV.tensor(new FloatXArray(0), [0, 0])
    );
    const indices = base.indices.map(idx => this.evalExpr(idx));
    // Use builtinIndexStore for compound assignment store-back —
    // this bypasses subsasgn for class instances (the compound decomposition
    // already handled the field set; the store-back should use builtin mechanics)
    const result = this.rt.builtinIndexStore(baseVal, indices, value);
    this.writeLValueBase(base.base, ensureRuntimeValue(result));
  } else if (base.type === "IndexCell") {
    const baseVal = this.evalLValueBase(base.base, RTV.cell([], [0, 0]));
    const indices = base.indices.map(idx => this.evalExpr(idx));
    const result = this.rt.indexCellStore(baseVal, indices, value);
    this.writeLValueBase(base.base, ensureRuntimeValue(result));
  }
}

export function evalLValueBase(
  this: Interpreter,
  base: Expr,
  defaultVal: RuntimeValue
): unknown {
  if (base.type === "Ident") {
    return this.env.get(base.name) ?? defaultVal;
  }
  if (base.type === "Member") {
    const parentBase = this.evalLValueBase(base.base, RTV.struct({}));
    const parentRv = ensureRuntimeValue(parentBase);
    try {
      return this.rt.getMember(parentBase, base.name);
    } catch {
      const newStruct = RTV.struct({});
      const updatedParent = this.rt.setMemberReturn(
        parentRv,
        base.name,
        newStruct
      );
      if (base.base.type === "Ident") {
        this.env.set(base.base.name, ensureRuntimeValue(updatedParent));
      }
      return newStruct;
    }
  }
  if (base.type === "Index") {
    const baseVal = this.evalLValueBase(base.base, RTV.struct({}));
    const indices = base.indices.map(idx => this.evalExpr(idx));
    try {
      let skipSubsref: boolean | string = false;
      if (this.currentClassName) {
        const bRv = ensureRuntimeValue(baseVal);
        if (
          isRuntimeClassInstance(bRv) &&
          bRv.className === this.currentClassName
        ) {
          skipSubsref = true;
        }
      }
      return this.rt.index(baseVal, indices, 1, skipSubsref);
    } catch {
      return defaultVal;
    }
  }
  if (base.type === "IndexCell") {
    const baseVal = this.evalLValueBase(base.base, RTV.cell([], [0, 0]));
    const indices = base.indices.map(idx => this.evalExpr(idx));
    try {
      return this.rt.indexCell(baseVal, indices);
    } catch {
      return defaultVal;
    }
  }
  return this.evalExpr(base);
}

// ── Helpers ──────────────────────────────────────────────────────────────

export function switchMatch(
  this: Interpreter,
  switchVal: unknown,
  caseVal: unknown
): boolean {
  return runtimeSwitchMatch(switchVal, caseVal);
}

export function isOutputExpr(this: Interpreter, expr: Expr): boolean {
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

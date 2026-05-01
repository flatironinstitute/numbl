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
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
} from "../runtime/types.js";
import { RTV, getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { binop, uplus } from "../runtime/runtimeOperators.js";
import { mPow } from "../helpers/arithmetic.js";
import { getIBuiltinNargin } from "./builtins/types.js";
import { getConstant } from "../helpers/constants.js";
import { buildLineTable, offsetToLineFast } from "../runtime/error.js";
import { COLON_SENTINEL, END_SENTINEL } from "../runtime/sentinels.js";
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
import { makeRootContext } from "../executors/registry.js";
import { zeroedFloatX, setAllocSource } from "../runtime/alloc.js";
import { disposeValue, deepCloneValue } from "../runtime/utils.js";

// ── Ownership classification ─────────────────────────────────────────────

/** True when `expr`'s evaluation result is **owned** by the caller —
 *  a fresh value with no other live reference. See
 *  `docs/developer_reference/runtime/ownership-and-dispose.md` §3.
 *
 *  Conservative: any expression not on this list is treated as
 *  *borrowed* and cloned at binding seams. Only list nodes whose
 *  evaluators *always* return a fresh top-level value.
 */
function isOwnedExpr(expr: Expr): boolean {
  switch (expr.type) {
    case "Tensor":
    case "Cell":
    case "Range":
    case "Binary":
    case "Unary":
    case "FuncCall":
    case "MethodCall":
      return true;
    default:
      return false;
  }
}

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
    // Forward the .m source location to the leak tracker so it can
    // tag any allocations the upcoming statement makes. No-op when
    // tracking is off (default), so the runtime path stays cheap.
    setAllocSource(this.rt.$file, this.rt.$line);
  }

  switch (stmt.type) {
    case "ExprStmt": {
      const val = this.evalExprNargout(stmt.expr, 0);
      const singleVal = Array.isArray(val) ? val[0] : val;
      // Calls to functions with no outputs produce no value: do not set
      // `ans` and do not display (matches MATLAB behavior).
      if (singleVal === undefined) {
        return null;
      }
      const rv = ensureRuntimeValue(singleVal);
      this.ans = rv;
      this.env.set("ans", rv);
      if (!stmt.suppressed && !this.isOutputExpr(stmt.expr)) {
        this.rt.displayResult(rv);
      }
      return null;
    }

    case "Assign": {
      const rawVal = this.evalExpr(stmt.expr);
      const val = Array.isArray(rawVal) ? rawVal[0] : rawVal;
      // Owned rhs (TensorLit / Binary / Unary / Range / FuncCall /
      // MethodCall) is moved into the binding without a clone — the
      // expression evaluator already produced a fresh value with no
      // other live reference. Borrowed rhs (Ident / Member / IndexCell /
      // etc.) is deep-cloned so the new binding owns its own buffer.
      // See ownership-and-dispose.md §4.1.
      const valRv = ensureRuntimeValue(val);
      const rv: RuntimeValue = isOwnedExpr(stmt.expr)
        ? valRv
        : (deepCloneValue(valRv) as RuntimeValue);
      const old = this.env.get(stmt.name);
      this.env.set(stmt.name, rv);
      // Plain Var Assign produces a uniquely-owned new binding, so the
      // previous value becomes garbage and can be recycled. Unsafe at
      // AssignLValue (container mutation shares unchanged fields with
      // the new container — see §5). Also unsafe when the binding's
      // current wrapper is captured by a closure snapshot — but
      // bindings created after all snapshots are independent (§6).
      if (
        old !== undefined &&
        old !== rv &&
        !this.env.isNameCaptured(stmt.name)
      ) {
        disposeValue(old);
      }
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
        const { args: indices, ownedArgs: ownedIdxArgs } =
          this.evalIndicesWithEndTracked(cellBase, lv.indices);
        // Determine nargout from index count
        let expandedCount = 1;
        const idx0 = indices[0];
        if (idx0 === COLON_SENTINEL) {
          // [c{:}] — expand to all elements of the cell base
          const baseRv = ensureRuntimeValue(cellBase);
          expandedCount = isRuntimeCell(baseRv) ? baseRv.data.length : 0;
        } else {
          const idxVal = ensureRuntimeValue(idx0);
          if (isRuntimeTensor(idxVal)) {
            expandedCount = idxVal.data.length;
          } else if (typeof idxVal === "number") {
            expandedCount = 1;
          }
        }
        // Pre-fetch the old entries about to be replaced. Each one is
        // unreachable from the new cell (multiOutputCellAssign rebuilds
        // cell-by-cell). Skip when the env-binding name is captured.
        const cellRootCapture =
          lv.base.type === "Ident"
            ? this.env.isNameCaptured(lv.base.name)
            : this.env.envCaptured;
        const oldEntries: RuntimeValue[] = [];
        if (!cellRootCapture) {
          const baseRv = ensureRuntimeValue(cellBase);
          if (isRuntimeCell(baseRv)) {
            const collectIdx = (k0: number) => {
              if (k0 >= 0 && k0 < baseRv.data.length) {
                oldEntries.push(baseRv.data[k0]);
              }
            };
            if (idx0 === COLON_SENTINEL) {
              for (let i = 0; i < baseRv.data.length; i++) collectIdx(i);
            } else {
              const idxVal = ensureRuntimeValue(idx0);
              if (isRuntimeTensor(idxVal)) {
                for (let i = 0; i < idxVal.data.length; i++) {
                  collectIdx(Math.round(idxVal.data[i]) - 1);
                }
              } else if (typeof idxVal === "number") {
                collectIdx(Math.round(idxVal) - 1);
              }
            }
          }
        }
        const val = this.evalExprNargout(stmt.expr, expandedCount);
        const values = Array.isArray(val) ? val : [val];
        const result = this.rt.multiOutputCellAssign(
          cellBase,
          idx0,
          values.map(v => ensureRuntimeValue(v))
        );
        const resultRv = ensureRuntimeValue(result);
        if (lv.base.type === "Ident") {
          this.env.set(lv.base.name, resultRv);
        }
        for (const e of oldEntries) {
          if (
            e !== null &&
            typeof e === "object" &&
            isRuntimeTensor(e as RuntimeValue)
          ) {
            disposeValue(e as RuntimeValue);
          }
        }
        // Owned index expressions (e.g. `1:N` in `out{1:N}`) are
        // consumed by multiOutputCellAssign — recycle.
        for (const a of ownedIdxArgs) {
          if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
        }
        return null;
      }

      const nargout = stmt.lvalues.length;
      const val = this.evalExprNargout(stmt.expr, nargout);
      const values = Array.isArray(val) ? val : [val];
      // Owned multi-output expressions (FuncCall / MethodCall) hand
      // each declared-output value over by ownership transfer at exit
      // (see ownership-and-dispose.md §4.4); the lvalues here become
      // the new owners with no clone needed. Borrowed expressions
      // would alias caller state, so deep-clone defensively.
      const isOwned = isOwnedExpr(stmt.expr);
      for (let i = 0; i < stmt.lvalues.length; i++) {
        const lv = stmt.lvalues[i];
        const raw = i < values.length ? values[i] : undefined;
        if (lv.type === "Ignore") {
          // Owned output dropped on the floor (`[~, i] = f(...)`); the
          // value never gets a holder.
          if (
            isOwned &&
            raw !== undefined &&
            raw !== null &&
            typeof raw === "object"
          ) {
            disposeValue(raw as RuntimeValue);
          }
          continue;
        }
        const rv = isOwned ? (raw as RuntimeValue) : this.rt.share(raw);
        this.assignLValue(lv, rv);
      }
      if (!stmt.suppressed) {
        for (let i = 0; i < stmt.lvalues.length; i++) {
          const lv = stmt.lvalues[i];
          if (lv.type === "Var" && i < values.length) {
            this.rt.displayAssign(lv.name, ensureRuntimeValue(values[i]));
          }
        }
      }
      return null;
    }

    case "AssignLValue": {
      const val = this.evalExpr(stmt.expr);
      const valRv = ensureRuntimeValue(Array.isArray(val) ? val[0] : val);
      // Same owned-vs-borrowed rule as Assign (§4.1): owned rhs is moved
      // into the lvalue's container without a clone; borrowed rhs is
      // deep-cloned so the target owns its own buffer (otherwise the
      // new container aliases the original env binding / member chain).
      const rv: RuntimeValue = isOwnedExpr(stmt.expr)
        ? valRv
        : (deepCloneValue(valRv) as RuntimeValue);
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
      const _whileStart = this.rt.profilingEnabled ? performance.now() : 0;
      let _whileIters = 0;
      while (true) {
        this.rt.checkCancel();
        const cond = this.evalExpr(stmt.cond);
        if (!this.rt.toBool(cond)) break;
        _whileIters++;
        const signal = this.execStmts(stmt.body);
        if (signal instanceof BreakSignal) break;
        if (signal instanceof ContinueSignal) continue;
        if (signal instanceof ReturnSignal) {
          recordHotLoop(this, stmt, "while", _whileIters, _whileStart);
          return signal;
        }
      }
      recordHotLoop(this, stmt, "while", _whileIters, _whileStart);
      return null;
    }

    case "For": {
      const _forStart = this.rt.profilingEnabled ? performance.now() : 0;
      const iterVal = this.evalExpr(stmt.expr);
      const rv = ensureRuntimeValue(iterVal);
      const iterItems = forIter(rv);
      let returnSignal: ReturnSignal | null = null;
      let iterCount = 0;
      for (let _i = 0; _i < iterItems.length; _i++) {
        this.rt.checkCancel();
        iterCount = _i + 1;
        // Iterating columns of a 2-D matrix produces a fresh tensor per
        // step (§3 forIter); the previous iteration's binding becomes
        // unreferenced when the new column is set. Recycle it here so
        // the loop doesn't accumulate one buffer per column.
        const prev = this.env.get(stmt.varName);
        this.env.set(stmt.varName, ensureRuntimeValue(iterItems[_i]));
        if (
          prev !== undefined &&
          prev !== null &&
          !this.env.envCaptured &&
          typeof prev === "object"
        ) {
          disposeValue(prev as RuntimeValue);
        }
        const signal = this.execStmts(stmt.body);
        if (signal instanceof BreakSignal) break;
        if (signal instanceof ContinueSignal) continue;
        if (signal instanceof ReturnSignal) {
          returnSignal = signal;
          break;
        }
      }
      // The iteration expression result (e.g. `1:N`) was held by the
      // `for` itself; once iteration completes it has no other holder.
      // It's a fresh value that never lived in env.vars, so any
      // closure snapshot taken in scope cannot reference it — the
      // envCaptured check that applies to binding-overwrite seams
      // does not apply here.
      if (isOwnedExpr(stmt.expr) && rv !== null && typeof rv === "object") {
        disposeValue(rv as RuntimeValue);
      }
      recordHotLoop(this, stmt, "for", iterCount, _forStart);
      if (returnSignal) return returnSignal;
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

    case "Directive": {
      if (stmt.directive === "assert_jit") {
        // Only enforce assert_jit at --opt 1 (JS-JIT). Other opt
        // modes (e3 etc.) cover narrower shapes than JS-JIT, so a
        // directive surviving to the interpreter is expected and
        // doesn't represent a regression. At --opt 0 it's a no-op.
        if (this.optimization !== "1") return null;
        const wantC = stmt.args.includes("c");
        throw new RuntimeError(
          `%!numbl:assert_jit${wantC ? " c" : ""}: expected the surrounding loop or function body to be JIT-compiled, but it was interpreted. Run with --opt 0 to silence.`
        );
      }
      // Unknown directives are silently ignored.
      return null;
    }

    case "Synth": {
      // Fallback: a transformer-built Synth node reached the
      // interpreter (no specialized executor matched, or the matching
      // executor bailed). Just run the original sub-stmts in order.
      for (const sub of stmt.subStmts) {
        const sig = this.execStmt(sub);
        if (sig) return sig;
      }
      return null;
    }
  }
}

export function execStmts(
  this: Interpreter,
  stmts: Stmt[]
): ControlSignal | null {
  // Apply registered AST transformers before walking. Cached per
  // input-list identity (WeakMap), so the cost is paid once per
  // unique stmt list. With no transformers registered (--opt 0/1),
  // returns the input unchanged with no cache lookup.
  const transformed = this.registry.transformStmts(stmts);
  // Allocate one DispatchContext for the whole sibling loop; reset
  // per-dispatch state between stmts. Hot-path code — a fresh ctx
  // per stmt would allocate a Map + Set per dispatch.
  const ctx = makeRootContext(this, this.registry);
  for (let i = 0; i < transformed.length; i++) {
    ctx.resetForNextDispatch();
    const result = this.registry.dispatch(transformed, i, ctx);
    if (result.signal) return result.signal;
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
      const argResult =
        fieldVal !== undefined
          ? this.evalIndicesWithEndTracked(fieldVal, expr.args)
          : this.evalArgsTracked(expr.args);
      const args = argResult.args;
      const ownedArgs = argResult.ownedArgs;
      const result = this.rt.methodDispatch(expr.name, nargout, [
        base,
        ...args,
      ]);
      if (ownedArgs.length > 0) {
        const resultRv = ensureRuntimeValue(
          Array.isArray(result) ? result[0] : result
        );
        for (const a of ownedArgs) {
          if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
        }
      }
      return result;
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
              : RTV.tensor(zeroedFloatX(0), [0, 0]);
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

/** Like `evalArgs`, but additionally records each *owned* expression
 *  result so the caller can dispose them after the function call
 *  returns. The callee's `callUserFunction` deep-clones each arg at
 *  entry, so post-call the originals are unreferenced and recyclable.
 *  Multi-output / array-splat results are conservatively treated as
 *  borrowed (not disposed). */
export function evalArgsTracked(
  this: Interpreter,
  argExprs: Expr[]
): { args: unknown[]; ownedArgs: RuntimeValue[] } {
  const args: unknown[] = [];
  const ownedArgs: RuntimeValue[] = [];
  for (const a of argExprs) {
    const val = this.evalExpr(a);
    if (Array.isArray(val)) {
      for (const elem of val) args.push(elem);
      continue;
    }
    args.push(val);
    if (isOwnedExpr(a) && val !== null && typeof val === "object") {
      ownedArgs.push(val as RuntimeValue);
    }
  }
  return { args, ownedArgs };
}

// ── Binary operators ─────────────────────────────────────────────────────

const binopProfileName: Record<string, string> = {
  [BinaryOperation.Add]: "plus",
  [BinaryOperation.Sub]: "minus",
  [BinaryOperation.Mul]: "mtimes",
  [BinaryOperation.ElemMul]: "times",
  [BinaryOperation.Div]: "mrdivide",
  [BinaryOperation.ElemDiv]: "rdivide",
  [BinaryOperation.LeftDiv]: "mldivide",
  [BinaryOperation.ElemLeftDiv]: "ldivide",
  [BinaryOperation.Pow]: "mpower",
  [BinaryOperation.ElemPow]: "power",
};

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

  // Profile non-scalar binary ops (tensor arithmetic)
  let result: unknown;
  if (
    this.rt.profilingEnabled &&
    (typeof left !== "number" || typeof right !== "number")
  ) {
    const opName = binopProfileName[expr.op] ?? expr.op;
    this.rt.profileEnter("builtin:interp:" + opName);
    result = binop(expr.op, left, right);
    this.rt.profileLeave();
  } else {
    result = binop(expr.op, left, right);
  }

  // Owned operand intermediates (`a*2`'s tensor in `a*2 + 1`) become
  // unreferenced once the binop produces its fresh result. Recycle them
  // here so chained binops don't leak per stage. Skip when the operand
  // is also the result (defensive — current binop builtins always
  // allocate fresh, but a future identity case shouldn't double-dispose).
  if (
    isOwnedExpr(expr.left) &&
    isRuntimeTensor(lv) &&
    lv !== result &&
    lv !== ensureRuntimeValue(result)
  ) {
    disposeValue(lv);
  }
  if (
    isOwnedExpr(expr.right) &&
    isRuntimeTensor(rv) &&
    rv !== result &&
    rv !== ensureRuntimeValue(result)
  ) {
    disposeValue(rv);
  }
  return result;
}

// ── Unary operators ──────────────────────────────────────────────────────

export function evalUnary(
  this: Interpreter,
  expr: Extract<Expr, { type: "Unary" }>
): unknown {
  const operand = this.evalExpr(expr.operand);
  let result: unknown;
  switch (expr.op) {
    case UnaryOperation.Plus:
      result = uplus(operand);
      break;
    case UnaryOperation.Minus:
      result = this.rt.uminus(operand);
      break;
    case UnaryOperation.Not:
      result =
        typeof operand === "number"
          ? RTV.logical(operand === 0)
          : this.rt.not(operand);
      break;
    case UnaryOperation.Transpose:
      // ' = conjugate transpose
      result = this.rt.ctranspose(operand);
      break;
    case UnaryOperation.NonConjugateTranspose:
      // .' = non-conjugate transpose
      result = this.rt.transpose(operand);
      break;
  }
  // Mirror evalBinary: an owned operand intermediate (e.g. `(a+b)'`)
  // becomes unreferenced once the unary op produces its fresh result.
  // Skip when operand IS the result — runtime `uplus` returns its
  // input verbatim, and a real-scalar transpose / not is a passthrough.
  if (
    isOwnedExpr(expr.operand) &&
    isRuntimeTensor(operand as RuntimeValue) &&
    operand !== result &&
    operand !== ensureRuntimeValue(result)
  ) {
    disposeValue(operand as RuntimeValue);
  }
  return result;
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
      const { args, ownedArgs } = this.evalArgsTracked(expr.args);
      const result = this.rt.index(rv, args, nargout);
      if (ownedArgs.length > 0) {
        const resultRv = ensureRuntimeValue(
          Array.isArray(result) ? result[0] : result
        );
        for (const a of ownedArgs) {
          if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
        }
      }
      return result;
    }
    const { args, ownedArgs } = this.evalIndicesWithEndTracked(
      varVal,
      expr.args
    );
    // Inside class methods, bypass overloaded subsref for same-class instances
    let skipSubsref: boolean | string = false;
    if (
      this.currentClassName &&
      isRuntimeClassInstance(rv) &&
      rv.className === this.currentClassName
    ) {
      skipSubsref = true;
    }
    const result = this.rt.index(varVal, args, nargout, skipSubsref);
    // Owned index expressions (e.g. `x(1:N)`'s range tensor) are
    // consumed by the index path which copies values out into the slice
    // result. Defend against identity-return: skip args that are the
    // result itself.
    if (ownedArgs.length > 0) {
      const resultRv = ensureRuntimeValue(
        Array.isArray(result) ? result[0] : result
      );
      for (const a of ownedArgs) {
        if (a !== resultRv && isRuntimeTensor(a)) {
          disposeValue(a);
        }
      }
    }
    return result;
  }
  const { args, ownedArgs } = this.evalArgsTracked(expr.args);
  // Constant called as zero-arg function? e.g. eps(), pi(), inf()
  if (args.length === 0) {
    const c = getConstant(expr.name);
    if (c !== undefined) return c;
  }
  const result = this.callFunction(expr.name, args, nargout);
  // Owned arg tensors (TensorLit / Binary / Unary / Range / FuncCall /
  // MethodCall results) are deep-cloned at the user-fn entry boundary
  // (callUserFunction) and consumed-and-copied by builtins, so the
  // outer result no longer references their buffers. Recycle them.
  // Skip an arg if it is the same object as the result — defends
  // against any remaining identity-return builtins.
  if (ownedArgs.length > 0) {
    const resultRv = ensureRuntimeValue(
      Array.isArray(result) ? result[0] : result
    );
    for (const a of ownedArgs) {
      if (a !== resultRv && isRuntimeTensor(a)) {
        disposeValue(a);
      }
    }
  }
  return result;
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
  const { args: indices, ownedArgs } = this.evalIndicesWithEndTracked(
    base,
    expr.indices
  );
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
  const result = this.rt.index(base, indices, 1, skipSubsref);
  if (ownedArgs.length > 0) {
    const resultRv = ensureRuntimeValue(
      Array.isArray(result) ? result[0] : result
    );
    for (const a of ownedArgs) {
      if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
    }
  }
  return result;
}

export function evalIndexCell(
  this: Interpreter,
  expr: Extract<Expr, { type: "IndexCell" }>
): unknown {
  const base = this.evalExpr(expr.base);
  const { args: indices, ownedArgs } = this.evalIndicesWithEndTracked(
    base,
    expr.indices
  );
  const result = this.rt.indexCell(base, indices);
  if (ownedArgs.length > 0) {
    const resultRv = ensureRuntimeValue(
      Array.isArray(result) ? result[0] : result
    );
    for (const a of ownedArgs) {
      if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
    }
  }
  return result;
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

/** Like evalIndicesWithEnd but additionally records each owned index
 *  expression's result so the caller can dispose it after the index
 *  call has consumed it. Mirrors `evalArgsTracked`. */
export function evalIndicesWithEndTracked(
  this: Interpreter,
  base: unknown,
  indexExprs: Expr[]
): { args: unknown[]; ownedArgs: RuntimeValue[] } {
  const numIndices = indexExprs.length;
  const ownedArgs: RuntimeValue[] = [];
  const args = indexExprs.map((idx, dimIndex) => {
    this.endContextStack.push({ base, dimIndex, numIndices });
    try {
      const v = this.evalExpr(idx);
      if (
        isOwnedExpr(idx) &&
        v !== null &&
        v !== undefined &&
        typeof v === "object"
      ) {
        ownedArgs.push(v as RuntimeValue);
      }
      return v;
    } finally {
      this.endContextStack.pop();
    }
  });
  return { args, ownedArgs };
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

  const hasVarargin =
    paramNames.length > 0 && paramNames[paramNames.length - 1] === "varargin";
  const regularParams = hasVarargin ? paramNames.slice(0, -1) : paramNames;

  const fn = RTV.func("anonymous", "user");
  fn.jsFn = (nargoutArg: unknown, ...rest: unknown[]) => {
    const fnEnv = new Environment(capturedEnv);
    const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
    for (let i = 0; i < regularParams.length; i++) {
      if (i < actualArgs.length) {
        fnEnv.set(regularParams[i], ensureRuntimeValue(actualArgs[i]));
      }
    }
    if (hasVarargin) {
      const extraArgs = actualArgs
        .slice(regularParams.length)
        .map(a => ensureRuntimeValue(a));
      fnEnv.set("varargin", RTV.cell(extraArgs, [1, extraArgs.length]));
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
          const result = this.evalExprNargout(bodyExpr, narg);
          // MATLAB: an anonymous function whose body is an expression (not
          // a multi-output function call) only produces a single output.
          // If more are requested, throw.  When the body is a call that
          // actually returned multiple outputs, evalExprNargout returns an
          // array of values, so we allow that.
          if (narg > 1 && !(Array.isArray(result) && result.length >= narg)) {
            throw new RuntimeError("Too many output arguments.");
          }
          return result;
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

  // If `name` resolves to a nested function defined here or in an ancestor,
  // the handle keeps that env alive via closure. Mark the chain so the
  // function-exit cleanup skips clearLocals — tearing the env down would
  // strand the handle if it escapes via an output / persistent / global.
  // Builtin/user-function handles don't need this marker; their dispatch
  // doesn't depend on `capturedEnv`'s vars.
  capturedEnv.markChainForNestedHandle(name);

  const fn = RTV.func(name, "builtin");
  fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
    const actualArgs = Array.isArray(rest[0]) ? (rest[0] as unknown[]) : rest;
    const narg = typeof nargout === "number" ? nargout : 1;
    const nested = capturedEnv.getNestedFunction(name);
    if (nested) {
      return this.withFileContext(
        capturedFile,
        capturedClassName,
        capturedMethodName,
        () => this.callNestedFunction(nested.fn, nested.env, actualArgs, narg)
      );
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
  const narg = getIBuiltinNargin(name);
  if (narg !== undefined) fn.nargin = narg;
  return fn;
}

// ── Tensor/Cell literal construction ─────────────────────────────────────

export function evalTensorLiteral(
  this: Interpreter,
  expr: Extract<Expr, { type: "Tensor" }>
): unknown {
  if (expr.rows.length === 0) {
    return RTV.tensor(zeroedFloatX(0), [0, 0]);
  }
  const rowValues: unknown[] = [];
  // Track which entries in rowValues are freshly-allocated horzcat
  // results (so they can be disposed once vertcat consumes them).
  const horzcatResults: number[] = [];
  // Per-element owned tensors (one entry per element across all rows
  // that came from an `isOwnedExpr` AST node). horzcat / single-element
  // pickup only references the buffer; the wrapper tensor is no longer
  // needed afterwards.
  const ownedRowElements: unknown[] = [];
  for (const row of expr.rows) {
    const vals: unknown[] = [];
    for (const e of row) {
      const v = this.evalExpr(e);
      if (Array.isArray(v)) {
        for (const elem of v) vals.push(elem);
      } else {
        vals.push(v);
        if (isOwnedExpr(e)) ownedRowElements.push(v);
      }
    }
    if (vals.length === 1) {
      rowValues.push(vals[0]);
    } else {
      horzcatResults.push(rowValues.length);
      rowValues.push(this.rt.horzcat(vals));
    }
  }
  let result: unknown;
  if (rowValues.length === 1) {
    result = rowValues[0];
  } else {
    result = this.rt.vertcat(rowValues);
  }
  // vertcat copies values out of each row into a fresh buffer; the
  // freshly-allocated horzcat-row tensors are no longer referenced.
  const resultRv = ensureRuntimeValue(result);
  for (const idx of horzcatResults) {
    const row = rowValues[idx];
    if (
      row !== resultRv &&
      row !== null &&
      typeof row === "object" &&
      isRuntimeTensor(row as RuntimeValue)
    ) {
      disposeValue(row as RuntimeValue);
    }
  }
  // Per-element owned tensors that were passed into horzcat / used as
  // a sole row value. horzcat/vertcat copy values out, so the original
  // tensor wrappers are now garbage.
  for (const v of ownedRowElements) {
    if (
      v !== resultRv &&
      v !== null &&
      typeof v === "object" &&
      isRuntimeTensor(v as RuntimeValue)
    ) {
      disposeValue(v as RuntimeValue);
    }
  }
  return result;
}

export function evalCellLiteral(
  this: Interpreter,
  expr: Extract<Expr, { type: "Cell" }>
): unknown {
  if (expr.rows.length === 0) {
    return RTV.cell([], [0, 0]);
  }
  // Each cell entry must be the cell's unique owner. Borrowed entries
  // (an Ident `{a}` is the env binding's wrapper) are deep-cloned so
  // the cell doesn't alias caller state. Owned entries (TensorLit /
  // Binary / FuncCall result, etc.) move into the cell directly.
  const takeEntry = (e: Expr, v: unknown): RuntimeValue => {
    const rv = ensureRuntimeValue(v);
    if (
      isOwnedExpr(e) ||
      typeof rv !== "object" ||
      rv === null ||
      isRuntimeFunction(rv)
    ) {
      return rv;
    }
    return deepCloneValue(rv) as RuntimeValue;
  };
  if (expr.rows.length === 1) {
    const elements: RuntimeValue[] = [];
    for (const e of expr.rows[0]) {
      const v = this.evalExpr(e);
      if (Array.isArray(v)) {
        // Multi-output expansion: each result is owned per §4.4.
        for (const elem of v) elements.push(ensureRuntimeValue(elem));
      } else {
        elements.push(takeEntry(e, v));
      }
    }
    return RTV.cell(elements, [1, elements.length]);
  }
  const numRows = expr.rows.length;
  const numCols = expr.rows[0].length;
  const elements: RuntimeValue[] = [];
  for (let c = 0; c < numCols; c++) {
    for (let r = 0; r < numRows; r++) {
      const e = expr.rows[r][c];
      elements.push(takeEntry(e, this.evalExpr(e)));
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
          : this.evalLValueBase(lv.base, RTV.tensor(zeroedFloatX(0), [0, 0]));
      const { args: indices, ownedArgs: ownedIdxArgs } =
        this.evalIndicesWithEndTracked(
          base ?? RTV.tensor(zeroedFloatX(0), [0, 0]),
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
      const baseIsTensor =
        base !== undefined &&
        base !== null &&
        isRuntimeTensor(base as RuntimeValue);
      const result = this.rt.indexStore(base, indices, value, skipSubsasgn);
      const resultRv = ensureRuntimeValue(result);
      if (lv.base.type === "Ident") {
        this.env.set(lv.base.name, resultRv);
      } else {
        this.writeLValueBase(lv.base, resultRv);
      }
      // When indexStore allocates a fresh tensor (e.g. growTensor2D when
      // the assignment extends past the current shape, or scalar→tensor
      // conversion), the OLD tensor's buffers are no longer reachable
      // from the binding and can be recycled. Restrict to tensor bases:
      // dict / cell / struct(-array) "rebuild" by sharing unchanged
      // entries with the new container, so disposing the old container
      // would corrupt the new one (see ownership-and-dispose.md §5).
      // Skip when the env is captured by a closure snapshot (§6).
      // The Ident-rooted base case: the binding's wrapper is captured
      // only when the *named* binding existed at snapshot time. For
      // compound bases (Member/Index), conservatively use envCaptured
      // since we don't know which name the dispose would affect.
      const baseCapture =
        lv.base.type === "Ident"
          ? this.env.isNameCaptured(lv.base.name)
          : this.env.envCaptured;
      if (
        base !== undefined &&
        base !== null &&
        base !== resultRv &&
        !baseCapture &&
        isRuntimeTensor(base as RuntimeValue)
      ) {
        disposeValue(base as RuntimeValue);
      }
      // Tensor-base indexStore reads values out of `value` (assignSlice
      // / assignStripe) into the base buffer; the rhs wrapper is no
      // longer referenced afterwards. The rhs is a fresh owned/cloned
      // wrapper (per AssignLValue), never in env.vars — independent of
      // any capture state. Also covers the auto-create case (base is
      // undefined / 0×0 empty): indexStore allocates the new tensor
      // and copies values out of rhs.
      const resultIsTensor = isRuntimeTensor(resultRv);
      if (
        (baseIsTensor || resultIsTensor) &&
        isRuntimeTensor(value) &&
        value !== resultRv
      ) {
        disposeValue(value);
      }
      // Owned index expressions (e.g. `x(1:N) = …`) — indexStore reads
      // their values to compute slot positions, then the original
      // tensor wrappers are unreferenced. Fresh values, never live in
      // env.vars, so the envCaptured check that guards binding-overwrite
      // does not apply.
      for (const a of ownedIdxArgs) {
        if (a !== resultRv && a !== value && isRuntimeTensor(a)) {
          disposeValue(a);
        }
      }
      break;
    }

    case "IndexCell": {
      const base =
        lv.base.type === "Ident"
          ? (this.env.get(lv.base.name) ?? RTV.cell([], [0, 0]))
          : this.evalLValueBase(lv.base, RTV.cell([], [0, 0]));
      const { args: indices, ownedArgs: ownedCellIdxArgs } =
        this.evalIndicesWithEndTracked(base, lv.indices);
      // Pre-fetch the old entry at the cell index. indexCellStore
      // rebuilds the cell, sharing the unchanged entries with the new
      // cell — but the entry at the targeted index is replaced. The
      // old entry is no longer referenced from the new cell. Skip
      // when the env is captured by a closure snapshot, or for
      // multi-index stores (we'd need to dispose multiple entries).
      const cellRootCapture =
        lv.base.type === "Ident"
          ? this.env.isNameCaptured(lv.base.name)
          : this.env.envCaptured;
      let oldCellEntry: RuntimeValue | undefined;
      if (
        !cellRootCapture &&
        indices.length === 1 &&
        isRuntimeCell(ensureRuntimeValue(base))
      ) {
        const cell = ensureRuntimeValue(
          base
        ) as import("../runtime/types.js").RuntimeCell;
        const idxVal = ensureRuntimeValue(indices[0]);
        if (typeof idxVal === "number") {
          const k = Math.round(idxVal) - 1;
          if (k >= 0 && k < cell.data.length) oldCellEntry = cell.data[k];
        }
      }
      const result = this.rt.indexCellStore(base, indices, value);
      const cellResultRv = ensureRuntimeValue(result);
      if (lv.base.type === "Ident") {
        this.env.set(lv.base.name, cellResultRv);
      } else {
        this.writeLValueBase(lv.base, cellResultRv);
      }
      if (
        oldCellEntry !== undefined &&
        oldCellEntry !== value &&
        oldCellEntry !== null &&
        typeof oldCellEntry === "object" &&
        isRuntimeTensor(oldCellEntry as RuntimeValue)
      ) {
        disposeValue(oldCellEntry as RuntimeValue);
      }
      for (const a of ownedCellIdxArgs) {
        if (a !== cellResultRv && isRuntimeTensor(a)) disposeValue(a);
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
      // Pre-fetch the old leaf so we can recycle it after the
      // assignment. setMemberReturn / memberChainAssign rebuild the
      // container, sharing every unchanged sibling field with the new
      // container — but the leaf field at the end of the chain is
      // replaced. The old leaf has no other live owner. Skip when env
      // is captured by a closure snapshot, or when the root is a
      // handle class (other handles may still observe it).
      // For Ident-rooted member chains, the root binding's capture
      // determines safety: if the binding existed at snapshot time,
      // the snapshot's wrapper graph still reaches the leaf field via
      // the unchanged-field-sharing chain (§5). Bindings created later
      // are independent — disposing their old leaves is safe.
      const memberRootCapture =
        cursor.type === "Ident"
          ? this.env.isNameCaptured(cursor.name)
          : this.env.envCaptured;
      let oldLeaf: RuntimeValue | undefined;
      if (
        !memberRootCapture &&
        !(isRuntimeClassInstance(rootRv) && rootRv.isHandleClass)
      ) {
        let cur: RuntimeValue | undefined = rootRv;
        for (const n of names) {
          if (cur === undefined) break;
          if (isRuntimeClassInstance(cur) || isRuntimeStruct(cur)) {
            cur = cur.fields.get(n);
          } else {
            cur = undefined;
            break;
          }
        }
        oldLeaf = cur;
      }
      let memberResult: RuntimeValue;
      if (isRuntimeClassInstance(rootRv)) {
        // Use memberChainAssign which routes through subsasgn if needed
        const result = this.rt.memberChainAssign(rootBase, names, value);
        memberResult = ensureRuntimeValue(result);
        if (cursor.type === "Ident") {
          this.env.set(cursor.name, memberResult);
        } else {
          this.writeLValueBase(cursor, memberResult);
        }
      } else {
        // Non-class: use direct field set with store-back chain
        const base = this.evalLValueBase(lv.base, RTV.struct({}));
        const result = this.rt.setMemberReturn(base, lv.name, value);
        memberResult = ensureRuntimeValue(result);
        this.writeLValueBase(lv.base, memberResult);
      }
      if (
        oldLeaf !== undefined &&
        oldLeaf !== value &&
        oldLeaf !== null &&
        typeof oldLeaf === "object" &&
        isRuntimeTensor(oldLeaf as RuntimeValue)
      ) {
        disposeValue(oldLeaf as RuntimeValue);
      }
      // If a property setter (or subsasgn override) intercepted the
      // assignment, the rhs was deep-cloned at the user-fn entry; the
      // outer caller's `value` is no longer referenced. Detect this by
      // walking the chain in the result and checking whether the leaf
      // wrapper is the same object we passed in. If different, our
      // value can be recycled.
      if (
        isRuntimeTensor(value) &&
        value !== memberResult &&
        !memberRootCapture
      ) {
        let leaf: RuntimeValue | undefined = memberResult;
        for (const n of names) {
          if (leaf === undefined) break;
          if (isRuntimeClassInstance(leaf) || isRuntimeStruct(leaf)) {
            leaf = leaf.fields.get(n);
          } else {
            leaf = undefined;
            break;
          }
        }
        if (leaf !== value) {
          disposeValue(value);
        }
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
      RTV.tensor(zeroedFloatX(0), [0, 0])
    );
    const { args: indices, ownedArgs: ownedIdx } =
      this.evalIndicesWithEndTracked(baseVal, base.indices);
    // Use builtinIndexStore for compound assignment store-back —
    // this bypasses subsasgn for class instances (the compound decomposition
    // already handled the field set; the store-back should use builtin mechanics)
    const result = this.rt.builtinIndexStore(baseVal, indices, value);
    const resultRv = ensureRuntimeValue(result);
    this.writeLValueBase(base.base, resultRv);
    for (const a of ownedIdx) {
      if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
    }
  } else if (base.type === "IndexCell") {
    const baseVal = this.evalLValueBase(base.base, RTV.cell([], [0, 0]));
    const { args: indices, ownedArgs: ownedIdx } =
      this.evalIndicesWithEndTracked(baseVal, base.indices);
    const result = this.rt.indexCellStore(baseVal, indices, value);
    const resultRv = ensureRuntimeValue(result);
    this.writeLValueBase(base.base, resultRv);
    for (const a of ownedIdx) {
      if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
    }
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
    const { args: indices, ownedArgs: ownedIdx } =
      this.evalIndicesWithEndTracked(baseVal, base.indices);
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
      const result = this.rt.index(baseVal, indices, 1, skipSubsref);
      if (ownedIdx.length > 0) {
        const resultRv = ensureRuntimeValue(
          Array.isArray(result) ? result[0] : result
        );
        for (const a of ownedIdx) {
          if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
        }
      }
      return result;
    } catch {
      for (const a of ownedIdx) {
        if (isRuntimeTensor(a)) disposeValue(a);
      }
      return defaultVal;
    }
  }
  if (base.type === "IndexCell") {
    const baseVal = this.evalLValueBase(base.base, RTV.cell([], [0, 0]));
    const { args: indices, ownedArgs: ownedIdx } =
      this.evalIndicesWithEndTracked(baseVal, base.indices);
    try {
      const result = this.rt.indexCell(baseVal, indices);
      if (ownedIdx.length > 0) {
        const resultRv = ensureRuntimeValue(
          Array.isArray(result) ? result[0] : result
        );
        for (const a of ownedIdx) {
          if (a !== resultRv && isRuntimeTensor(a)) disposeValue(a);
        }
      }
      return result;
    } catch {
      for (const a of ownedIdx) {
        if (isRuntimeTensor(a)) disposeValue(a);
      }
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
  const outputFunctions = [
    "disp",
    "display",
    "fprintf",
    "warning",
    "assert",
    "tic",
    "toc",
    "help",
  ];
  if (expr.type === "FuncCall") return outputFunctions.includes(expr.name);
  if (expr.type === "Ident") return outputFunctions.includes(expr.name);
  return false;
}

function recordHotLoop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" | "While" },
  kind: "for" | "while",
  iterations: number,
  startTime: number
): void {
  if (!interp.rt.profilingEnabled || iterations <= 1000 || !stmt.span) return;
  const durationMs = performance.now() - startTime;
  let table = interp.lineTableCache.get(stmt.span.file);
  if (!table) {
    const src = interp.fileSources.get(stmt.span.file) ?? "";
    table = buildLineTable(src);
    interp.lineTableCache.set(stmt.span.file, table);
  }
  const line = offsetToLineFast(table, stmt.span.start);
  const key = `${stmt.span.file}:${line}`;
  const prev = interp.rt.hotLoops.get(key);
  if (prev) {
    prev.callCount++;
    prev.totalTimeMs += durationMs;
    if (iterations > prev.iterations) prev.iterations = iterations;
  } else {
    interp.rt.hotLoops.set(key, {
      file: stmt.span.file,
      line,
      kind,
      iterations,
      callCount: 1,
      totalTimeMs: durationMs,
    });
  }
}

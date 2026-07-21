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
  isRuntimeClassInstanceArray,
  isRuntimeFunction,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  isRuntimeStringArray,
} from "../runtime/types.js";
import { RTV, getItemTypeFromRuntimeValue } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError, CancellationError } from "../runtime/error.js";
import { getLastJitDecline } from "../jitDeclineDiagnostics.js";
import { binop, uplus } from "../runtime/runtimeOperators.js";
import { enumEqualityOp } from "../runtime/runtimeEnum.js";
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
import { allocFloat64Array } from "../runtime/alloc.js";
import { cowCopy } from "../runtime/cow.js";
import { incref, isShared } from "../runtime/refcount.js";

// ── Statement execution ──────────────────────────────────────────────────

export function execStmt(this: Interpreter, stmt: Stmt): ControlSignal | null {
  return this.rt.withScope(() => execStmtInner.call(this, stmt));
}

function execStmtInner(this: Interpreter, stmt: Stmt): ControlSignal | null {
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
      const rv = val as RuntimeValue;
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
        const indices = this.evalIndicesWithEnd(cellBase, lv.indices);
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
        const val = this.evalExprNargout(stmt.expr, expandedCount);
        const values = Array.isArray(val) ? val : [val];
        const result = this.rt.multiOutputCellAssign(
          cellBase,
          idx0,
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
        const rv = (i < values.length ? values[i] : undefined) as RuntimeValue;
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
      const rv = val as RuntimeValue;
      this.assignLValue(stmt.lvalue, rv);
      if (!stmt.suppressed) {
        if (stmt.lvalue.type === "Var") {
          this.rt.displayAssign(stmt.lvalue.name, rv);
        }
      }
      return null;
    }

    case "If": {
      const cond = this.evalCondition(stmt.cond);
      if (this.rt.toBool(cond)) {
        return this.execBlockStmts(stmt.thenBody);
      }
      for (const elseif of stmt.elseifBlocks) {
        const elseifCond = this.evalCondition(elseif.cond);
        if (this.rt.toBool(elseifCond)) {
          return this.execBlockStmts(elseif.body);
        }
      }
      if (stmt.elseBody) {
        return this.execBlockStmts(stmt.elseBody);
      }
      return null;
    }

    case "While": {
      const _whileStart = this.rt.profilingEnabled ? performance.now() : 0;
      let _whileIters = 0;
      // `loopDepth` gates per-call JIT proposals — see comment on
      // the field in `Interpreter`. Bumped around the body only, not
      // the cond eval (the cond is evaluated once per iteration but
      // is conceptually loop-control, not a hot inner call).
      this.loopDepth++;
      try {
        while (true) {
          this.rt.checkCancel();
          const cond = this.evalCondition(stmt.cond);
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
      } finally {
        this.loopDepth--;
      }
      recordHotLoop(this, stmt, "while", _whileIters, _whileStart);
      return null;
    }

    case "For": {
      const _forStart = this.rt.profilingEnabled ? performance.now() : 0;
      // Lazy range iteration: a `for k = a:s:b` over a scalar-numeric
      // range is iterated WITHOUT materializing the range tensor or a JS
      // array of every index. The element count and per-element value
      // (`start + step*i`, with the last snapped to `end`) are computed
      // by the same formula as makeRangeTensor, so loop-variable values
      // are byte-identical — this only skips allocating a row vector
      // whose elements we'd consume one at a time anyway (and which a
      // `break`-heavy loop, e.g. adaptive quadrature's `for i=1:1e5`,
      // barely touches). Char ranges ('a':'z') return null from
      // asScalarNumber and keep the eager path, preserving char elems.
      let lazyRange: {
        count: number;
        start: number;
        step: number;
        end: number;
      } | null = null;
      let iterItems: unknown[] | null = null;
      if (stmt.expr.type === "Range") {
        const sv = this.evalExpr(stmt.expr.start);
        const stv = stmt.expr.step ? this.evalExpr(stmt.expr.step) : 1;
        const ev = this.evalExpr(stmt.expr.end);
        const s = asScalarNumber(sv);
        const st = asScalarNumber(stv);
        const e = asScalarNumber(ev);
        if (s !== null && st !== null && e !== null) {
          // makeRangeTensor's element-count formula, clamped to a finite
          // non-negative value (matches mtoc2_loop_count). step===0 and
          // non-finite counts yield an empty loop, same as the eager path.
          const rawN = st === 0 ? 0 : Math.floor((e - s) / st + 1 + 1e-10);
          const count = Number.isFinite(rawN) ? Math.max(0, rawN) : 0;
          lazyRange = { count, start: s, step: st, end: e };
        } else {
          iterItems = forIter(ensureRuntimeValue(runtimeRange(sv, stv, ev)));
        }
      } else {
        iterItems = forIter(ensureRuntimeValue(this.evalExpr(stmt.expr)));
      }

      const n = lazyRange ? lazyRange.count : iterItems!.length;
      this.loopDepth++;
      let _ran = 0;
      try {
        for (let _i = 0; _i < n; _i++) {
          this.rt.checkCancel();
          let elem: unknown;
          if (lazyRange) {
            let v = lazyRange.start + lazyRange.step * _i;
            // Snap the last element to exactly `end` (matches
            // makeRangeTensor / mtoc2_range_value), multi-element only.
            if (
              lazyRange.count > 1 &&
              _i === lazyRange.count - 1 &&
              Math.abs(v - lazyRange.end) < Math.abs(lazyRange.step) * 1e-10
            ) {
              v = lazyRange.end;
            }
            elem = v;
          } else {
            elem = iterItems![_i];
          }
          this.env.set(stmt.varName, ensureRuntimeValue(elem));
          const signal = this.execStmts(stmt.body);
          _ran = _i + 1;
          if (signal instanceof BreakSignal) break;
          if (signal instanceof ContinueSignal) continue;
          if (signal instanceof ReturnSignal) {
            recordHotLoop(this, stmt, "for", _i + 1, _forStart);
            return signal;
          }
        }
      } finally {
        this.loopDepth--;
      }
      recordHotLoop(
        this,
        stmt,
        "for",
        lazyRange ? _ran : iterItems!.length,
        _forStart
      );
      return null;
    }

    case "Switch": {
      const switchVal = this.evalExpr(stmt.expr);
      let matched = false;
      for (const c of stmt.cases) {
        const caseVal = this.evalExpr(c.value);
        if (this.switchMatch(switchVal, caseVal)) {
          matched = true;
          const signal = this.execBlockStmts(c.body);
          if (signal) return signal;
          break;
        }
      }
      if (!matched && stmt.otherwise) {
        return this.execBlockStmts(stmt.otherwise);
      }
      return null;
    }

    case "TryCatch": {
      try {
        const signal = this.execBlockStmts(stmt.tryBody);
        if (signal) return signal;
      } catch (e) {
        // A cancellation request (Ctrl-C / interrupt) is not catchable by
        // user try/catch — it must unwind the whole run, like MATLAB.
        if (e instanceof CancellationError) {
          throw e;
        }
        if (stmt.catchVar) {
          this.env.set(stmt.catchVar, this.rt.wrapError(e));
        }
        const signal = this.execBlockStmts(stmt.catchBody);
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
        // MATLAB: declaring a global that doesn't exist yet creates it,
        // initialized to []. This makes reads (e.g. isempty(x)) work
        // before any assignment instead of erroring as undefined.
        if (this.rt && !(name in this.rt.$g)) {
          const empty = RTV.tensor(allocFloat64Array(0), [0, 0]);
          incref(empty);
          this.rt.$g[name] = empty;
        }
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
      // `%!numbl:assert_jit` asserts that the enclosing loop / function /
      // script body is JIT-compiled. The lowerer treats the directive as
      // a no-op, so a JIT'd unit compiles it away and the interpreter
      // never reaches it. If we DO reach it here, the enclosing unit ran
      // in the interpreter — i.e. it was not JIT'd.
      //
      //   - plain `assert_jit`   requires JS-JIT at --opt 1 only.
      //   - `assert_jit c`       additionally requires C-JIT at --opt 2.
      //
      // --opt 0 is always a no-op. The --opt 2 case where a `c` unit
      // JS-JITs instead of C-JITs is forced here too: the JS-JIT
      // executors decline `c` units at --opt 2, routing them to the
      // interpreter when C-JIT also declines.
      if (stmt.directive === "assert_jit") {
        const wantC = stmt.args.includes("c");
        // Surface *why* the JIT declined. The guarded unit declined just
        // before falling through to the interpreter, so the most recent
        // recorded decline is (almost always) the relevant one.
        const decline = getLastJitDecline();
        const why = decline
          ? ` Most recent JIT decline (${decline.where}, ${decline.kind}): ${decline.message}`
          : ` (no JIT decline reason was recorded — the unit may have been ` +
            `declined before lowering, e.g. an unsupported input type.)`;
        if (this.optimization === "1") {
          throw new RuntimeError(
            `%!numbl:assert_jit: expected the enclosing loop/function/script ` +
              `to be JS-JIT-compiled at --opt 1, but it ran in the ` +
              `interpreter.${why} (Run with --opt 0 to silence.)`
          );
        }
        if (this.optimization === "2" && wantC) {
          throw new RuntimeError(
            `%!numbl:assert_jit c: expected the enclosing loop/function/` +
              `script to be C-JIT-compiled at --opt 2, but it ran in the ` +
              `interpreter.${why} (Run with --opt 0 to silence.)`
          );
        }
      }
      // No-ops: any directive at --opt 0; plain assert_jit at --opt 2;
      // unknown directives.
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

/** Execute a conditional-block body (`if` / `switch` / `try`), tracking
 *  `condBlockDepth` so a loop dispatched inside knows its sibling list is
 *  a nested block — the loop classifier then keeps every loop-assigned
 *  name live-out (the post-loop liveness scan can't see reads after the
 *  enclosing block). */
export function execBlockStmts(
  this: Interpreter,
  stmts: Stmt[]
): ControlSignal | null {
  this.condBlockDepth++;
  try {
    return this.execStmts(stmts);
  } finally {
    this.condBlockDepth--;
  }
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
      // Constant (pi, eps, ...)? Use the non-throwing lookup — the old
      // throw/catch built a RuntimeError (with a V8 stack capture) for
      // every non-variable identifier, which is a hot path: any bare
      // function-name reference, evaluated per loop iteration, paid for
      // a thrown-and-immediately-caught exception.
      const c = getConstant(expr.name);
      if (c !== undefined) return c;
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
        // A string scalar is a 1x1 array: end is 1 (s(end) is s itself).
        if (isRuntimeString(rv)) return 1;
        // Class instances: call overloaded end(obj, k, n) if available;
        // otherwise default to 1 (a scalar instance has extent 1 in every
        // dimension, matching MATLAB's builtin end).
        if (isRuntimeClassInstance(rv)) {
          if (this.rt.resolveClassMethod?.(rv.className, "end")) {
            return this.rt.dispatch("end", 1, [
              rv,
              ctx.dimIndex + 1,
              ctx.numIndices,
            ]);
          }
          return 1;
        }
        if (isRuntimeStructArray(rv)) {
          return rv.elements.length;
        }
        if (isRuntimeClassInstanceArray(rv)) {
          if (ctx.numIndices === 1) return rv.elements.length;
          return ctx.dimIndex < rv.shape.length ? rv.shape[ctx.dimIndex] : 1;
        }
        if (isRuntimeStringArray(rv)) {
          if (ctx.numIndices === 1) return rv.data.length;
          return ctx.dimIndex < 2 ? rv.shape[ctx.dimIndex] : 1;
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
              : RTV.tensor(allocFloat64Array(0), [0, 0]);
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

/** Compute the call-site variable names for `inputname`, aligned to the
 *  argument expressions. Entry i is the name of the variable passed as
 *  argument i+1, or '' if that argument is not a plain workspace variable.
 *
 *  Per MATLAB: an argument that uses cell `{}` or dot `.` indexing produces
 *  a comma-separated list, making the position of every following argument
 *  dynamic — so that argument and all subsequent ones report ''. Plain
 *  literals, expressions, and paren-indexing report '' only for themselves.
 *
 *  The returned array may be shorter than the flattened arg count (when an
 *  argument expanded to a CSL); callers treat out-of-range indices as ''.
 */
export function computeInputNames(
  argExprs: Expr[],
  callerEnv: Environment
): string[] {
  const names: string[] = [];
  let blanked = false;
  for (const a of argExprs) {
    if (blanked) {
      names.push("");
    } else if (a.type === "Ident" && callerEnv.has(a.name)) {
      names.push(a.name);
    } else if (
      a.type === "IndexCell" ||
      a.type === "Member" ||
      a.type === "MemberDynamic"
    ) {
      // cell / dot indexing: '' here and for everything after it
      names.push("");
      blanked = true;
    } else {
      names.push("");
    }
  }
  return names;
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
  // Enumeration ==/~= (member vs member / numeric / char). Handles both scalar
  // members and member arrays; returns null for other ops / non-enum operands.
  const enumResult = enumEqualityOp(expr.op, lv, rv);
  if (enumResult !== null) return enumResult;
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
  if (
    this.rt.profilingEnabled &&
    (typeof left !== "number" || typeof right !== "number")
  ) {
    const opName = binopProfileName[expr.op] ?? expr.op;
    this.rt.profileEnter("builtin:interp:" + opName);
    const result = binop(expr.op, left, right);
    this.rt.profileLeave();
    return result;
  }

  return binop(expr.op, left, right);
}

/** A value counts as scalar (numel === 1) for condition short-circuiting. */
function isScalarValue(v: unknown): boolean {
  if (typeof v === "number" || typeof v === "boolean") return true;
  const rv = ensureRuntimeValue(v);
  if (typeof rv === "number" || typeof rv === "boolean") return true;
  if (typeof rv === "string") return false; // treat char/string as non-scalar
  const shape = (rv as { shape?: number[] }).shape;
  if (shape) return numel(shape) === 1;
  return true; // complex_number and other scalar kinds
}

/**
 * Evaluate an `if`/`while`/`elseif` condition.
 *
 * MATLAB short-circuits the element-wise `&` and `|` operators when they
 * appear at the top of a conditional expression and the left operand is
 * scalar — e.g. `if nargin<4 | isempty(x)` must NOT evaluate `isempty(x)`
 * when `nargin<4` (otherwise `x` may be undefined). Outside this context
 * `&`/`|` evaluate both operands element-wise, so the special handling
 * lives here rather than in `evalBinary`.
 */
export function evalCondition(this: Interpreter, expr: Expr): unknown {
  if (
    expr.type === "Binary" &&
    (expr.op === BinaryOperation.BitOr || expr.op === BinaryOperation.BitAnd)
  ) {
    const left = this.evalCondition(expr.left);
    if (isScalarValue(left)) {
      const lb = this.rt.toBool(left);
      if (expr.op === BinaryOperation.BitOr) {
        if (lb) return RTV.logical(true);
      } else if (!lb) {
        return RTV.logical(false);
      }
      // Left scalar didn't decide it; the result follows the right side.
      return this.evalCondition(expr.right);
    }
    // Non-scalar left: no short-circuit, evaluate element-wise as usual.
    const right = this.evalExpr(expr.right);
    return binop(expr.op, left, right);
  }
  return this.evalExpr(expr);
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
  // Record call-site argument names so the callee's inputname() can read
  // them. Set after evalArgs (which may itself make calls) and consumed at
  // the start of callUserFunction.
  this.pendingInputNames =
    expr.args.length > 0 ? computeInputNames(expr.args, this.env) : undefined;
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
        // Enumeration member access: `ClassName.MemberName`.
        const enumMember = this.interpretEnumMember(prefix, methodName);
        if (enumMember !== null) return enumMember;
        if (
          this.functionIndex.classStaticMethods.get(prefix)?.has(methodName) ||
          // Implicit static `ClassName.empty` (handled by interpretClassMethod)
          (methodName === "empty" &&
            this.functionIndex.workspaceClasses.has(prefix))
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
  // Field access on a non-scalar struct array yields a comma-separated
  // list — represented as a JS array, like c{:} — so {s.f}, [s.f], and
  // f(s.f) expand per element. (Class-instance arrays keep their
  // materialized-horzcat behavior in rt.getMember.)
  const baseRv = ensureRuntimeValue(base);
  if (isRuntimeStructArray(baseRv) && baseRv.elements.length !== 1) {
    return baseRv.elements.map(el =>
      ensureRuntimeValue(this.rt.getMember(el, expr.name))
    );
  }
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

  const hasVarargin =
    paramNames.length > 0 && paramNames[paramNames.length - 1] === "varargin";
  const regularParams = hasVarargin ? paramNames.slice(0, -1) : paramNames;

  const fn = RTV.func("anonymous", "user");
  fn.jsFn = (nargoutArg: unknown, ...rest: unknown[]) => {
    const fnEnv = new Environment(capturedEnv);
    fnEnv.rt = this.rt;
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
          // Adopt the result into the caller's transient scope before
          // fnEnv.clearLocals decrefs every binding — without this an
          // anonymous-function output that's the only ref to a value
          // would drop to rc=0 (and _destroy fire) before the caller
          // can bind it.
          if (this.rt.currentScope) {
            if (Array.isArray(result)) {
              for (const v of result) this.rt.currentScope.adopt(v);
            } else {
              this.rt.currentScope.adopt(result as RuntimeValue);
            }
          }
          return result;
        } finally {
          fnEnv.clearLocals();
          this.env = savedEnv;
        }
      }
    );
  };
  fn.jsFnExpectsNargout = true;
  fn.nargin = paramNames.length;
  // The snapshot incref'd every captured value; balance that with a
  // matching clearLocals when the function wrapper is destroyed.
  // Without this, captured Float64Array buffers stay in the pool's
  // liveSet for the rest of the runtime's life.
  fn.releaseExtra = () => capturedEnv.clearLocals();
  // Expose the snapshot to the alias sweep so a tensor held both in the
  // snapshot and in the parent env triggers COW on parent-side mutation
  // — preserves MATLAB's by-value capture semantics.
  fn.capturedEnv = capturedEnv;
  // Retain the defining AST + file so the JIT can inline a capture-free
  // handle that later crosses a compile boundary (loop input / call arg).
  fn.handleAst = expr;
  fn.handleDefFile = capturedFile;
  return fn;
}

// ── Function handles ─────────────────────────────────────────────────────

export function makeFuncHandle(this: Interpreter, name: string): RuntimeValue {
  // Handle dotted names. @ClassName.method is a static-method handle; anything
  // else (e.g. @pkg.fn or @pkg.subpkg.fn) is a package-function handle that
  // dispatches through the normal function-resolution path.
  const dotIdx = name.indexOf(".");
  if (dotIdx > 0) {
    const className = name.slice(0, dotIdx);
    const methodName = name.slice(dotIdx + 1);
    if (this.ctx.getClassInfo(className)) {
      const fn = RTV.func(name, "builtin");
      fn.jsFn = (nargout: unknown, ...rest: unknown[]) => {
        const actualArgs = Array.isArray(rest[0])
          ? (rest[0] as unknown[])
          : rest;
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
    // Not a class — fall through to the package/workspace handle path below.
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
  const isNested = capturedEnv.markChainForNestedHandle(name);

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
  // Populate nargin for the handle: builtins (e.g. nargin(@sin) == 1), then
  // the declared input count of a user/nested/local function (so
  // `nargin(@f)` works for `@f` the same as it would for a call to `f`).
  let narg = getIBuiltinNargin(name);
  if (narg === undefined) {
    // Never let an introspection lookup break handle creation.
    try {
      narg = this.declaredNargin(name);
    } catch {
      narg = undefined;
    }
  }
  if (narg !== undefined) fn.nargin = narg;
  // For nested-function handles, the function-exit cleanup skipped
  // clearLocals (so the closure could keep its captured env alive).
  // When the handle dies, release those captures by clearing the env.
  if (isNested) {
    fn.releaseExtra = () => capturedEnv.clearLocals();
  } else {
    // A plain `@name` handle to a workspace/local/package function (no
    // captured env). Retain its AST + file so the JIT can inline it as
    // an in-scope `@name` constant when it crosses a compile boundary.
    // Nested-function handles are excluded: they depend on capturedEnv,
    // which the inlined form can't reconstruct.
    fn.handleAst = {
      type: "FuncHandle",
      name,
      span: { file: capturedFile, start: 0, end: 0 },
    };
    fn.handleDefFile = capturedFile;
  }
  return fn;
}

// ── Tensor/Cell literal construction ─────────────────────────────────────

export function evalTensorLiteral(
  this: Interpreter,
  expr: Extract<Expr, { type: "Tensor" }>
): unknown {
  if (expr.rows.length === 0) {
    return RTV.tensor(allocFloat64Array(0), [0, 0]);
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
  // Multi-row: evaluate each row, expanding comma-separated lists (JS
  // arrays, e.g. c{:} or structArray.field), then require rectangularity.
  const rowValues: RuntimeValue[][] = expr.rows.map(row => {
    const vals: RuntimeValue[] = [];
    for (const e of row) {
      const v = this.evalExpr(e);
      if (Array.isArray(v)) {
        for (const elem of v) vals.push(ensureRuntimeValue(elem));
      } else {
        vals.push(ensureRuntimeValue(v));
      }
    }
    return vals;
  });
  const numRows = rowValues.length;
  const numCols = rowValues[0].length;
  for (const rv of rowValues) {
    if (rv.length !== numCols) {
      throw new RuntimeError(
        "Cell array rows must have the same number of columns"
      );
    }
  }
  const elements: RuntimeValue[] = [];
  for (let c = 0; c < numCols; c++) {
    for (let r = 0; r < numRows; r++) {
      elements.push(rowValues[r][c]);
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
              RTV.tensor(allocFloat64Array(0), [0, 0])
            );
      const indices = this.evalIndicesWithEnd(
        base ?? RTV.tensor(allocFloat64Array(0), [0, 0]),
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
      RTV.tensor(allocFloat64Array(0), [0, 0])
    );
    const indices = this.evalIndicesWithEnd(baseVal, base.indices);
    // Use builtinIndexStore for compound assignment store-back —
    // this bypasses subsasgn for class instances (the compound decomposition
    // already handled the field set; the store-back should use builtin mechanics)
    const result = this.rt.builtinIndexStore(baseVal, indices, value);
    this.writeLValueBase(base.base, ensureRuntimeValue(result));
  } else if (base.type === "IndexCell") {
    const baseVal = this.evalLValueBase(base.base, RTV.cell([], [0, 0]));
    const indices = this.evalIndicesWithEnd(baseVal, base.indices);
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
    let parentBase = this.evalLValueBase(base.base, RTV.struct({}));
    let parentRv = ensureRuntimeValue(parentBase);
    // Refcount-driven COW: if the parent is shared with another
    // binding (env, struct field, cell element, …), copy it now and
    // rebind in its location. After this, the chain from env root to
    // `parent` is uniquely owned, so mutations through this slot won't
    // leak to the other holder.
    if (isShared(parentRv)) {
      const cowed = cowCopy(parentRv);
      this.writeLValueBase(base.base, cowed);
      parentBase = cowed;
      parentRv = cowed;
    }
    try {
      return this.rt.getMember(parentBase, base.name);
    } catch {
      // Field doesn't exist — auto-create using the caller-supplied
      // default. The default reflects what the next lvalue level needs
      // (empty struct for `.member`, empty tensor for `(...)` /
      // `{...}`), so `s.a(3) = 5` initializes `s.a` as an empty tensor
      // that the subsequent indexStore can grow to `[0 0 5]`, while
      // `s.a.b = 5` keeps the existing empty-struct path.
      const newValue = defaultVal;
      const updatedParent = this.rt.setMemberReturn(
        parentRv,
        base.name,
        newValue
      );
      if (base.base.type === "Ident") {
        this.env.set(base.base.name, ensureRuntimeValue(updatedParent));
      }
      return newValue;
    }
  }
  if (base.type === "Index") {
    let baseVal = this.evalLValueBase(base.base, RTV.struct({}));
    let baseRv = ensureRuntimeValue(baseVal);
    if (isShared(baseRv)) {
      const cowed = cowCopy(baseRv);
      this.writeLValueBase(base.base, cowed);
      baseVal = cowed;
      baseRv = cowed;
    }
    const indices = this.evalIndicesWithEnd(baseVal, base.indices);
    try {
      let skipSubsref: boolean | string = false;
      if (this.currentClassName) {
        if (
          isRuntimeClassInstance(baseRv) &&
          baseRv.className === this.currentClassName
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
    let baseVal = this.evalLValueBase(base.base, RTV.cell([], [0, 0]));
    let baseRv = ensureRuntimeValue(baseVal);
    if (isShared(baseRv)) {
      const cowed = cowCopy(baseRv);
      this.writeLValueBase(base.base, cowed);
      baseVal = cowed;
      baseRv = cowed;
    }
    const indices = this.evalIndicesWithEnd(baseVal, base.indices);
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

/** Extract a JS number from a scalar (number or 1×1 tensor) value;
 *  null when the value isn't a scalar. Used to detect a plain numeric
 *  range so a huge/infinite `for k = a:s:b` can iterate lazily. */
function asScalarNumber(v: unknown): number | null {
  const rv = ensureRuntimeValue(v);
  if (typeof rv === "number") return rv;
  if (isRuntimeTensor(rv) && rv.data.length === 1) return rv.data[0];
  return null;
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

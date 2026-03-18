/**
 * Statement code generation.
 *
 * Standalone functions that generate JavaScript for IR statements.
 * Each function takes the Codegen instance as the first parameter.
 */

import {
  type IRExpr,
  type IRStmt,
  itemTypeForExprKind,
} from "../lowering/index.js";
import { type ItemType, IType } from "../lowering/itemTypes.js";
import { snapshotTypes, restoreTypes } from "../lowering/lowerStmt.js";
import type { Codegen } from "./codegen.js";
import {
  genExpr,
  genIndexArg,
  isOutputFunction,
  genForRange,
} from "./codegenExpr.js";

// ── Externally-settable type reset (mirrors lowerStmt logic) ────────────

/**
 * After codegen processes a statement containing function calls,
 * reset types of externally-settable variables so subsequent codegen
 * doesn't use stale type assumptions for native math inlining.
 */
function resetExternallySettableTypesForCodegen(
  cg: Codegen,
  irExpr: IRExpr
): void {
  const ctx = cg.loweringCtx;
  if (
    ctx.workspaceAccessedVarNames.size === 0 &&
    ctx.callerAccessMap.size === 0
  ) {
    return;
  }

  if (ctx.workspaceAccessedVarNames.size > 0 && exprHasFuncCall(irExpr)) {
    for (const name of ctx.workspaceAccessedVarNames) {
      const v = ctx.lookup(name);
      if (v) cg.typeEnv.set(v.id, IType.Unknown);
    }
  }

  if (ctx.callerAccessMap.size > 0) {
    const callNames = new Set<string>();
    collectExprCallNames(irExpr, callNames);
    for (const callName of callNames) {
      const baseName = callName.includes(".")
        ? callName.slice(callName.lastIndexOf(".") + 1)
        : callName;
      const vars = ctx.callerAccessMap.get(baseName);
      if (vars) {
        for (const varName of vars) {
          const v = ctx.lookup(varName);
          if (v) cg.typeEnv.set(v.id, IType.Unknown);
        }
      }
    }
  }
}

function exprHasFuncCall(expr: IRExpr): boolean {
  const k = expr.kind;
  switch (k.type) {
    case "FuncCall":
    case "MethodCall":
    case "SuperConstructorCall":
    case "ClassInstantiation":
      return true;
    case "Binary":
      return exprHasFuncCall(k.left) || exprHasFuncCall(k.right);
    case "Unary":
      return exprHasFuncCall(k.operand);
    case "Range":
      return (
        exprHasFuncCall(k.start) ||
        (k.step !== null && exprHasFuncCall(k.step)) ||
        exprHasFuncCall(k.end)
      );
    case "Index":
    case "IndexCell":
      return exprHasFuncCall(k.base) || k.indices.some(i => exprHasFuncCall(i));
    case "Member":
      return exprHasFuncCall(k.base);
    case "MemberDynamic":
      return exprHasFuncCall(k.base) || exprHasFuncCall(k.nameExpr);
    case "AnonFunc":
      return exprHasFuncCall(k.body);
    case "Tensor":
    case "Cell":
      return k.rows.some(row => row.some(e => exprHasFuncCall(e)));
    default:
      return false;
  }
}

function collectExprCallNames(expr: IRExpr, out: Set<string>): void {
  const k = expr.kind;
  switch (k.type) {
    case "FuncCall":
      out.add(k.name);
      for (const a of k.args) collectExprCallNames(a, out);
      if (k.instanceBase) collectExprCallNames(k.instanceBase, out);
      break;
    case "MethodCall":
      out.add(k.name);
      collectExprCallNames(k.base, out);
      for (const a of k.args) collectExprCallNames(a, out);
      break;
    case "Binary":
      collectExprCallNames(k.left, out);
      collectExprCallNames(k.right, out);
      break;
    case "Unary":
      collectExprCallNames(k.operand, out);
      break;
    case "Range":
      collectExprCallNames(k.start, out);
      if (k.step) collectExprCallNames(k.step, out);
      collectExprCallNames(k.end, out);
      break;
    case "Index":
    case "IndexCell":
      collectExprCallNames(k.base, out);
      for (const i of k.indices) collectExprCallNames(i, out);
      break;
    case "Member":
      collectExprCallNames(k.base, out);
      break;
    case "MemberDynamic":
      collectExprCallNames(k.base, out);
      collectExprCallNames(k.nameExpr, out);
      break;
    case "SuperConstructorCall":
      for (const a of k.args) collectExprCallNames(a, out);
      break;
    case "AnonFunc":
      collectExprCallNames(k.body, out);
      break;
    case "Tensor":
    case "Cell":
      for (const row of k.rows)
        for (const e of row) collectExprCallNames(e, out);
      break;
    case "ClassInstantiation":
      for (const a of k.args) collectExprCallNames(a, out);
      break;
    default:
      break;
  }
}

// ── Public API ──────────────────────────────────────────────────────────

export function genStmts(cg: Codegen, stmts: IRStmt[]): void {
  for (const stmt of stmts) {
    genStmt(cg, stmt);
  }
}

export function genStmt(cg: Codegen, stmt: IRStmt): void {
  // Line tracking — always emit both file and line for every statement.
  // We cannot skip redundant emissions because function calls at runtime
  // change $rt.$file/$rt.$line, making the caller's values stale on return.
  if (stmt.span && cg.fileSources.size > 0 && !cg.noLineTracking) {
    const line = cg.getLineForOffset(stmt.span.file, stmt.span.start);
    if (line !== null) {
      cg.emit(`$rt.$file = ${JSON.stringify(stmt.span.file)};`);
      cg.emit(`$rt.$line = ${line};`);
    }
  }

  switch (stmt.type) {
    case "ExprStmt": {
      const val = genExpr(cg, stmt.expr);
      cg.emit(`$ret = ${val};`);
      if (!stmt.suppressed && !isOutputFunction(stmt.expr)) {
        cg.emit(`$rt.displayResult($ret);`);
      }
      resetExternallySettableTypesForCodegen(cg, stmt.expr);
      break;
    }

    case "Assign": {
      const val = genExpr(cg, stmt.expr);
      const vr = cg.varRef(stmt.variable.id.id);
      const valType =
        stmt.assignedType ?? itemTypeForExprKind(stmt.expr.kind, cg.typeEnv);
      const tc = cg.typeComment(valType);
      if (valType.kind === "Number") {
        cg.emit(`${vr} = ${val};${tc}`);
      } else {
        cg.emit(`${vr} = $rt.share(${val});${tc}`);
      }
      // Reset externally-settable types before updating the assigned var's type
      resetExternallySettableTypesForCodegen(cg, stmt.expr);
      // Update TypeEnv for subsequent statements reading this variable
      if (stmt.assignedType) {
        cg.typeEnv.set(stmt.variable.id, stmt.assignedType);
      }
      if (!stmt.suppressed) {
        const name = stmt.variable.name;
        cg.emit(`$rt.displayAssign(${JSON.stringify(name)}, ${vr});`);
      }
      break;
    }

    case "MultiAssign": {
      // Special case: [varargout{expr}] = func()
      if (
        stmt.lvalues.length === 1 &&
        stmt.lvalues[0]?.type === "IndexCell" &&
        stmt.lvalues[0].base.kind.type === "Var"
      ) {
        const lv = stmt.lvalues[0];
        const baseRef = cg.varRef(
          (lv.base.kind as Extract<typeof lv.base.kind, { type: "Var" }>)
            .variable.id.id
        );
        const idxExpr = genIndexArg(cg, lv.indices[0]);
        const idxVar = cg.freshTemp();
        cg.emit(`var ${idxVar} = ${idxExpr};`);
        const nVar = cg.freshTemp();
        cg.emit(
          `var ${nVar} = ${idxVar} && ${idxVar}.kind === "tensor" ? ${idxVar}.data.length : 1;`
        );
        const val = cg.withCodegenContext({ nargoutOverride: nVar }, () =>
          genExpr(cg, stmt.expr)
        );
        const tmp = cg.freshTemp();
        cg.emit(`var ${tmp} = ${val};`);
        cg.emit(`if (!Array.isArray(${tmp})) ${tmp} = [${tmp}];`);
        cg.emit(
          `${baseRef} = $rt.multiOutputCellAssign(${baseRef}, ${idxVar}, ${tmp}.slice(0, ${nVar}));`
        );
        break;
      }

      const val = genExpr(cg, stmt.expr);
      const tmp = cg.freshTemp();
      cg.emit(`var ${tmp} = ${val};`);
      cg.emit(`if (!Array.isArray(${tmp})) ${tmp} = [${tmp}];`);
      for (let i = 0; i < stmt.lvalues.length; i++) {
        const lv = stmt.lvalues[i];
        if (!lv) continue;
        const rhsItem = `$rt.share(${tmp}[${i}])`;
        cg.genLValueAssign(lv, rhsItem);
      }
      resetExternallySettableTypesForCodegen(cg, stmt.expr);
      // Update TypeEnv for subsequent statements reading these variables
      if (stmt.assignedTypes) {
        for (let i = 0; i < stmt.lvalues.length; i++) {
          const lv = stmt.lvalues[i];
          const ty = stmt.assignedTypes[i];
          if (lv?.type === "Var" && ty) {
            cg.typeEnv.set(lv.variable.id, ty);
          }
        }
      }
      if (!stmt.suppressed) {
        for (let i = 0; i < stmt.lvalues.length; i++) {
          const lv = stmt.lvalues[i];
          if (!lv || lv.type !== "Var") continue;
          const vid = lv.variable.id.id;
          const name = lv.variable.name;
          cg.emit(
            `$rt.displayAssign(${JSON.stringify(name)}, ${cg.varRef(vid)});`
          );
        }
      }
      break;
    }

    case "AssignLValue": {
      const rhs = genExpr(cg, stmt.expr);
      cg.genLValueAssign(stmt.lvalue, rhs);
      resetExternallySettableTypesForCodegen(cg, stmt.expr);
      // Apply type side effects computed during lowering
      applyTypeUpdates(cg, stmt.typeUpdates);
      break;
    }

    case "If": {
      // Use postFlowTypes from lowering to determine affected variables
      const varIds = new Set(stmt.postFlowTypes?.map(([id]) => id) ?? []);
      const preBranch = snapshotTypes(cg.typeEnv, varIds);

      const cond = genExpr(cg, stmt.cond);
      cg.emit(`if ($rt.toBool(${cond})) {`);
      cg.pushIndent();
      genStmts(cg, stmt.thenBody);
      cg.popIndent();
      restoreTypes(cg.typeEnv, preBranch);

      for (const block of stmt.elseifBlocks) {
        const c = genExpr(cg, block.cond);
        cg.emit(`} else if ($rt.toBool(${c})) {`);
        cg.pushIndent();
        genStmts(cg, block.body);
        cg.popIndent();
        restoreTypes(cg.typeEnv, preBranch);
      }

      if (stmt.elseBody) {
        cg.emit(`} else {`);
        cg.pushIndent();
        genStmts(cg, stmt.elseBody);
        cg.popIndent();
      }

      cg.emit(`}`);
      applyTypeUpdates(cg, stmt.postFlowTypes);
      break;
    }

    case "While": {
      // Apply loop-widened types before condition and body
      applyTypeUpdates(cg, stmt.postFlowTypes);
      const cond = genExpr(cg, stmt.cond);

      cg.emit(`while ($rt.toBool(${cond})) {`);
      cg.pushIndent();
      genStmts(cg, stmt.body);
      cg.popIndent();
      cg.emit(`}`);

      // Restore post-loop types (body generation mutates typeEnv)
      applyTypeUpdates(cg, stmt.postFlowTypes);
      break;
    }

    case "For": {
      // Set iter variable type for codegen inside the loop body
      if (stmt.iterVarType) {
        cg.typeEnv.set(stmt.variable.id, stmt.iterVarType);
      }
      // Apply loop-widened types before body
      applyTypeUpdates(cg, stmt.postFlowTypes);

      if (stmt.expr.kind.type === "Range") {
        genForRange(
          cg,
          stmt.variable.id.id,
          () => genStmts(cg, stmt.body),
          stmt.expr.kind
        );
      } else {
        const iter = genExpr(cg, stmt.expr);
        const iterVar = cg.freshTemp();
        const idxVar = cg.freshTemp();
        const lenVar = cg.freshTemp();
        cg.emit(`var ${iterVar} = $rt.forIter(${iter});`);
        cg.emit(`var ${lenVar} = ${iterVar}.length;`);
        cg.emit(
          `for (var ${idxVar} = 0; ${idxVar} < ${lenVar}; ${idxVar}++) {`
        );
        cg.pushIndent();
        cg.emit(`${cg.varRef(stmt.variable.id.id)} = ${iterVar}[${idxVar}];`);
        genStmts(cg, stmt.body);
        cg.popIndent();
        cg.emit(`}`);
      }

      // Restore post-loop types (body generation mutates typeEnv)
      applyTypeUpdates(cg, stmt.postFlowTypes);
      break;
    }

    case "Switch": {
      const varIds = new Set(stmt.postFlowTypes?.map(([id]) => id) ?? []);
      const preBranch = snapshotTypes(cg.typeEnv, varIds);

      const val = genExpr(cg, stmt.expr);
      const tmp = cg.freshTemp();
      cg.emit(`var ${tmp} = ${val};`);
      let first = true;
      for (const c of stmt.cases) {
        const cv = genExpr(cg, c.value);
        const kw = first ? "if" : "} else if";
        cg.emit(`${kw} ($rt.switchMatch(${tmp}, ${cv})) {`);
        cg.pushIndent();
        genStmts(cg, c.body);
        cg.popIndent();
        restoreTypes(cg.typeEnv, preBranch);
        first = false;
      }
      if (stmt.otherwise) {
        cg.emit(first ? `{` : `} else {`);
        cg.pushIndent();
        genStmts(cg, stmt.otherwise);
        cg.popIndent();
      }
      if (!first || stmt.otherwise) {
        cg.emit(`}`);
      }
      applyTypeUpdates(cg, stmt.postFlowTypes);
      break;
    }

    case "TryCatch": {
      const varIds = new Set(stmt.postFlowTypes?.map(([id]) => id) ?? []);
      const preBranch = snapshotTypes(cg.typeEnv, varIds);

      cg.emit(`try {`);
      cg.pushIndent();
      genStmts(cg, stmt.tryBody);
      cg.popIndent();
      restoreTypes(cg.typeEnv, preBranch);

      const catchParam = cg.freshTemp();
      cg.emit(`} catch (${catchParam}) {`);
      cg.pushIndent();
      if (stmt.catchVar) {
        cg.emit(
          `${cg.varRef(stmt.catchVar.id.id)} = $rt.wrapError(${catchParam});`
        );
      }
      genStmts(cg, stmt.catchBody);
      cg.popIndent();
      cg.emit(`}`);
      applyTypeUpdates(cg, stmt.postFlowTypes);
      break;
    }

    case "Break":
      cg.emit(`break;`);
      break;

    case "Continue":
      cg.emit(`continue;`);
      break;

    case "Return": {
      const fnCtx =
        cg.currentFunctionOutputs[cg.currentFunctionOutputs.length - 1];
      if (fnCtx) {
        cg.emitReturnCapture(fnCtx);
        cg.emit(`return ${fnCtx.resultVarName};`);
      } else {
        cg.emit(`return;`);
      }
      break;
    }

    case "Function": {
      // If this is a nested function (parent has pushed shared vars),
      // generate it inline so JavaScript closure provides variable sharing.
      // Use $fn_{originalName} as the JS name so callers can reference it.
      if (cg.sharedVarIdStack.length > 0) {
        const jsId = `$nested_function_${cg.sanitizeName(stmt.originalName)}`;
        cg.genFunctionDef(stmt, jsId);
      }
      break;
    }

    case "Global": {
      for (const v of stmt.vars) {
        cg.globalVarRefs.set(
          v.variable.id.id,
          `$rt.$g[${JSON.stringify(v.name)}]`
        );
      }
      break;
    }

    case "Persistent": {
      if (cg.currentFunctionJsId) {
        const funcId = JSON.stringify(cg.currentFunctionJsId);
        for (const v of stmt.vars) {
          const varRef = cg.varRef(v.variable.id.id);
          cg.emit(
            `${varRef} = $rt.getPersistent(${funcId}, ${JSON.stringify(v.name)});`
          );
        }
      }
      break;
    }

    case "ClassDef":
      // Not supported yet
      break;

    case "Import":
      // Declarative; imports are collected by buildFunctionIndex()
      break;
  }
}

// ── Flow-dependent type helpers ─────────────────────────────────────────

/** Apply pre-computed type updates to the codegen TypeEnv. */
function applyTypeUpdates(cg: Codegen, types?: [string, ItemType][]): void {
  if (types) {
    for (const [id, ty] of types) {
      cg.typeEnv.set({ id }, ty);
    }
  }
}

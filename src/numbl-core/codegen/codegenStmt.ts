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
import { walkExpr } from "../lowering/nodeUtils.js";
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
 * reset types of variables declared via `% external-access:` to Unknown
 * so subsequent codegen doesn't use stale type assumptions.
 */
function resetExternallySettableTypesForCodegen(
  cg: Codegen,
  irExpr: IRExpr
): void {
  const ctx = cg.loweringCtx;
  if (ctx.externalAccessVarNames.size === 0) return;
  let hasFuncCall = false;
  walkExpr(irExpr, e => {
    if (!hasFuncCall) {
      const t = e.kind.type;
      if (
        t === "FuncCall" ||
        t === "MethodCall" ||
        t === "SuperConstructorCall" ||
        t === "ClassInstantiation"
      ) {
        hasFuncCall = true;
      }
    }
  });
  if (!hasFuncCall) return;
  for (const name of ctx.externalAccessVarNames) {
    const v = ctx.lookup(name);
    if (v) cg.typeEnv.set(v.id, IType.Unknown);
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
      // who()/whos() as expression statements (no assignment) should print, not return
      if (
        stmt.expr.kind.type === "FuncCall" &&
        (stmt.expr.kind.name === "who" || stmt.expr.kind.name === "whos")
      ) {
        const fnName = stmt.expr.kind.name;
        const whoVarMap =
          cg.whoVarGetterStack.length > 0
            ? cg.whoVarGetterStack[cg.whoVarGetterStack.length - 1]
            : new Map<string, string>();
        const entries = [...whoVarMap.entries()]
          .map(([name, jsRef]) => `${JSON.stringify(name)}: () => ${jsRef}`)
          .join(", ");
        const whoArgs = stmt.expr.kind.args.map(a => genExpr(cg, a));
        cg.emit(`$rt.${fnName}(0, {${entries}}, [${whoArgs.join(", ")}]);`);
        resetExternallySettableTypesForCodegen(cg, stmt.expr);
        break;
      }
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

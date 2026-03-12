/**
 * Statement code generation.
 *
 * Standalone functions that generate JavaScript for IR statements.
 * Each function takes the Codegen instance as the first parameter.
 */

import { type IRStmt, itemTypeForExprKind } from "../lowering/index.js";
import type { Codegen } from "./codegen.js";
import {
  genExpr,
  genIndexArg,
  isOutputFunction,
  genForRange,
} from "./codegenExpr.js";

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
      break;
    }

    case "Assign": {
      const val = genExpr(cg, stmt.expr);
      const vr = cg.varRef(stmt.variable.id.id);
      const valType = itemTypeForExprKind(stmt.expr.kind);
      const tc = cg.typeComment(valType);
      if (valType.kind === "Number") {
        cg.emit(`${vr} = ${val};${tc}`);
      } else {
        cg.emit(`${vr} = $rt.share(${val});${tc}`);
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
      break;
    }

    case "If": {
      const cond = genExpr(cg, stmt.cond);
      cg.emit(`if ($rt.toBool(${cond})) {`);
      cg.pushIndent();
      genStmts(cg, stmt.thenBody);
      cg.popIndent();

      for (const block of stmt.elseifBlocks) {
        const c = genExpr(cg, block.cond);
        cg.emit(`} else if ($rt.toBool(${c})) {`);
        cg.pushIndent();
        genStmts(cg, block.body);
        cg.popIndent();
      }

      if (stmt.elseBody) {
        cg.emit(`} else {`);
        cg.pushIndent();
        genStmts(cg, stmt.elseBody);
        cg.popIndent();
      }

      cg.emit(`}`);
      break;
    }

    case "While": {
      const cond = genExpr(cg, stmt.cond);
      cg.emit(`while ($rt.toBool(${cond})) {`);
      cg.pushIndent();
      genStmts(cg, stmt.body);
      cg.popIndent();
      cg.emit(`}`);
      break;
    }

    case "For": {
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
      break;
    }

    case "Switch": {
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
      break;
    }

    case "TryCatch": {
      cg.emit(`try {`);
      cg.pushIndent();
      genStmts(cg, stmt.tryBody);
      cg.popIndent();
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

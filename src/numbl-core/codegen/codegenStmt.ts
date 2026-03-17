/**
 * Statement code generation.
 *
 * Standalone functions that generate JavaScript for IR statements.
 * Each function takes the Codegen instance as the first parameter.
 */

import {
  type IRExprKind,
  type IRExpr,
  type IRLValue,
  type IRStmt,
  itemTypeForExprKind,
} from "../lowering/index.js";
import { type ItemType, IType, isScalarType } from "../lowering/itemTypes.js";
import { type TypeEnv } from "../lowering/typeEnv.js";
import { collectAssignedVarIds } from "../lowering/lowerStmt.js";
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
      const valType =
        stmt.assignedType ?? itemTypeForExprKind(stmt.expr.kind, cg.typeEnv);
      const tc = cg.typeComment(valType);
      if (valType.kind === "Number") {
        cg.emit(`${vr} = ${val};${tc}`);
      } else {
        cg.emit(`${vr} = $rt.share(${val});${tc}`);
      }
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
      // Replay struct member type updates (mirrors updateStructTypeForMemberAssign)
      if (stmt.lvalue.type === "Member") {
        replayMemberTypeUpdate(cg, stmt.lvalue, stmt.expr);
      }
      // Replay indexed assignment scalar→Unknown widening
      if (
        (stmt.lvalue.type === "Index" || stmt.lvalue.type === "IndexCell") &&
        stmt.lvalue.base.kind.type === "Var"
      ) {
        const v = stmt.lvalue.base.kind.variable;
        const vTy = cg.typeEnv.get(v.id);
        if (vTy && isScalarType(vTy)) {
          cg.typeEnv.set(v.id, IType.Unknown);
        }
      }
      break;
    }

    case "If": {
      // Collect assigned vars for type join
      const ifAssigned = new Set<string>();
      collectAssignedVarIds(stmt.thenBody, ifAssigned);
      for (const block of stmt.elseifBlocks)
        collectAssignedVarIds(block.body, ifAssigned);
      if (stmt.elseBody) collectAssignedVarIds(stmt.elseBody, ifAssigned);

      const preBranch = snapshotTypes(cg.typeEnv, ifAssigned);

      const cond = genExpr(cg, stmt.cond);
      cg.emit(`if ($rt.toBool(${cond})) {`);
      cg.pushIndent();
      genStmts(cg, stmt.thenBody);
      cg.popIndent();
      const postThen = snapshotTypes(cg.typeEnv, ifAssigned);
      restoreTypes(cg.typeEnv, preBranch);

      const postElseifs: Map<string, ItemType | undefined>[] = [];
      for (const block of stmt.elseifBlocks) {
        const c = genExpr(cg, block.cond);
        cg.emit(`} else if ($rt.toBool(${c})) {`);
        cg.pushIndent();
        genStmts(cg, block.body);
        cg.popIndent();
        postElseifs.push(snapshotTypes(cg.typeEnv, ifAssigned));
        restoreTypes(cg.typeEnv, preBranch);
      }

      let postElse: Map<string, ItemType | undefined> | null = null;
      if (stmt.elseBody) {
        cg.emit(`} else {`);
        cg.pushIndent();
        genStmts(cg, stmt.elseBody);
        cg.popIndent();
        postElse = snapshotTypes(cg.typeEnv, ifAssigned);
        restoreTypes(cg.typeEnv, preBranch);
      }

      cg.emit(`}`);
      const allBranch = [postThen, ...postElseifs];
      if (postElse) allBranch.push(postElse);
      joinBranchTypes(
        cg.typeEnv,
        ifAssigned,
        preBranch,
        allBranch,
        !stmt.elseBody
      );
      break;
    }

    case "While": {
      const whileAssigned = new Set<string>();
      collectAssignedVarIds(stmt.body, whileAssigned);
      const preLoop = snapshotTypes(cg.typeEnv, whileAssigned);

      // Only widen variables whose body-assigned type differs from
      // pre-loop type (lowering fixpoint ensures assignedType is accurate).
      widenChangedLoopVars(cg, stmt.body, preLoop);
      const cond = genExpr(cg, stmt.cond);

      cg.emit(`while ($rt.toBool(${cond})) {`);
      cg.pushIndent();
      genStmts(cg, stmt.body);
      cg.popIndent();
      cg.emit(`}`);

      const postBody = snapshotTypes(cg.typeEnv, whileAssigned);
      restoreTypes(cg.typeEnv, preLoop);
      joinBranchTypes(cg.typeEnv, whileAssigned, preLoop, [postBody], true);
      break;
    }

    case "For": {
      const forAssigned = new Set<string>();
      collectAssignedVarIds(stmt.body, forAssigned);
      // Set iter variable type for codegen inside the loop body
      if (stmt.iterVarType) {
        cg.typeEnv.set(stmt.variable.id, stmt.iterVarType);
      }
      const preLoop = snapshotTypes(cg.typeEnv, forAssigned);

      // Only widen variables whose body-assigned type differs from
      // pre-loop type (lowering fixpoint ensures assignedType is accurate).
      widenChangedLoopVars(cg, stmt.body, preLoop);

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

      const postBody = snapshotTypes(cg.typeEnv, forAssigned);
      restoreTypes(cg.typeEnv, preLoop);
      joinBranchTypes(cg.typeEnv, forAssigned, preLoop, [postBody], true);
      break;
    }

    case "Switch": {
      const switchAssigned = new Set<string>();
      for (const c of stmt.cases) collectAssignedVarIds(c.body, switchAssigned);
      if (stmt.otherwise) collectAssignedVarIds(stmt.otherwise, switchAssigned);
      const preBranch = snapshotTypes(cg.typeEnv, switchAssigned);

      const val = genExpr(cg, stmt.expr);
      const tmp = cg.freshTemp();
      cg.emit(`var ${tmp} = ${val};`);
      let first = true;
      const postCases: Map<string, ItemType | undefined>[] = [];
      for (const c of stmt.cases) {
        const cv = genExpr(cg, c.value);
        const kw = first ? "if" : "} else if";
        cg.emit(`${kw} ($rt.switchMatch(${tmp}, ${cv})) {`);
        cg.pushIndent();
        genStmts(cg, c.body);
        cg.popIndent();
        postCases.push(snapshotTypes(cg.typeEnv, switchAssigned));
        restoreTypes(cg.typeEnv, preBranch);
        first = false;
      }
      let postOtherwise: Map<string, ItemType | undefined> | null = null;
      if (stmt.otherwise) {
        cg.emit(first ? `{` : `} else {`);
        cg.pushIndent();
        genStmts(cg, stmt.otherwise);
        cg.popIndent();
        postOtherwise = snapshotTypes(cg.typeEnv, switchAssigned);
        restoreTypes(cg.typeEnv, preBranch);
      }
      if (!first || stmt.otherwise) {
        cg.emit(`}`);
      }
      const allBranch = [...postCases];
      if (postOtherwise) allBranch.push(postOtherwise);
      joinBranchTypes(
        cg.typeEnv,
        switchAssigned,
        preBranch,
        allBranch,
        !stmt.otherwise
      );
      break;
    }

    case "TryCatch": {
      const tcAssigned = new Set<string>();
      collectAssignedVarIds(stmt.tryBody, tcAssigned);
      collectAssignedVarIds(stmt.catchBody, tcAssigned);
      const preBranch = snapshotTypes(cg.typeEnv, tcAssigned);

      cg.emit(`try {`);
      cg.pushIndent();
      genStmts(cg, stmt.tryBody);
      cg.popIndent();
      const postTry = snapshotTypes(cg.typeEnv, tcAssigned);
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
      const postCatch = snapshotTypes(cg.typeEnv, tcAssigned);
      restoreTypes(cg.typeEnv, preBranch);
      joinBranchTypes(
        cg.typeEnv,
        tcAssigned,
        preBranch,
        [postTry, postCatch],
        true
      );
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

/** Save current types for a set of variable IDs. */
function snapshotTypes(
  typeEnv: TypeEnv,
  varIds: Set<string>
): Map<string, ItemType | undefined> {
  const snap = new Map<string, ItemType | undefined>();
  for (const id of varIds) {
    snap.set(id, typeEnv.get({ id }));
  }
  return snap;
}

/** Restore types from a snapshot. */
function restoreTypes(
  typeEnv: TypeEnv,
  snap: Map<string, ItemType | undefined>
): void {
  for (const [id, ty] of snap) {
    if (ty !== undefined) {
      typeEnv.set({ id }, ty);
    }
  }
}

/** Join branch types: unify types across all branches at a merge point. */
function joinBranchTypes(
  typeEnv: TypeEnv,
  varIds: Set<string>,
  preBranchTypes: Map<string, ItemType | undefined>,
  branchTypes: Map<string, ItemType | undefined>[],
  includePreBranch: boolean
): void {
  for (const id of varIds) {
    let joined: ItemType | undefined = includePreBranch
      ? preBranchTypes.get(id)
      : undefined;
    for (const bt of branchTypes) {
      const ty = bt.get(id);
      if (ty !== undefined) {
        joined = joined !== undefined ? IType.unify(joined, ty) : ty;
      }
    }
    if (joined !== undefined) {
      typeEnv.set({ id }, joined);
    }
  }
}

/** Replay struct member type update during codegen (mirrors updateStructTypeForMemberAssign in lowering). */
function replayMemberTypeUpdate(
  cg: Codegen,
  lv: IRLValue & { type: "Member" },
  rhsExpr: IRExpr
): void {
  const chain: string[] = [lv.name];
  let cursor: IRExprKind = lv.base.kind;
  while (cursor.type === "Member") {
    chain.unshift(cursor.name);
    cursor = cursor.base.kind;
  }
  if (cursor.type !== "Var") return;
  const v = cursor.variable;
  const vTy = cg.typeEnv.get(v.id);
  if (vTy && vTy.kind !== "Struct" && vTy.kind !== "Unknown") return;

  const fieldType = itemTypeForExprKind(rhsExpr.kind, cg.typeEnv);
  let ty = IType.struct({ [chain[chain.length - 1]]: fieldType });
  for (let i = chain.length - 2; i >= 0; i--) {
    ty = IType.struct({ [chain[i]]: ty });
  }
  if (!vTy || vTy.kind === "Struct") {
    cg.typeEnv.unify(v.id, ty);
  }
}

/**
 * Collect the unified assigned types for each variable from the IR
 * `assignedType` annotations in a statement list.  For each variable
 * assigned anywhere in `stmts`, unifies all of its `assignedType` values
 * into a single type.  This tells us what type each variable may take
 * inside a loop body.
 */
function collectBodyAssignedTypes(
  stmts: IRStmt[],
  out: Map<string, ItemType>
): void {
  for (const s of stmts) {
    if (s.type === "Assign" && s.assignedType) {
      const id = s.variable.id.id;
      const prev = out.get(id);
      out.set(id, prev ? IType.unify(prev, s.assignedType) : s.assignedType);
    }
    if (s.type === "MultiAssign" && s.assignedTypes) {
      for (let i = 0; i < s.lvalues.length; i++) {
        const lv = s.lvalues[i];
        const ty = s.assignedTypes[i];
        if (lv?.type === "Var" && ty) {
          const id = lv.variable.id.id;
          const prev = out.get(id);
          out.set(id, prev ? IType.unify(prev, ty) : ty);
        }
      }
    }
    // Recurse into control flow
    if (s.type === "If") {
      collectBodyAssignedTypes(s.thenBody, out);
      for (const b of s.elseifBlocks) collectBodyAssignedTypes(b.body, out);
      if (s.elseBody) collectBodyAssignedTypes(s.elseBody, out);
    }
    if (s.type === "While" || s.type === "For")
      collectBodyAssignedTypes(s.body, out);
    if (s.type === "Switch") {
      for (const c of s.cases) collectBodyAssignedTypes(c.body, out);
      if (s.otherwise) collectBodyAssignedTypes(s.otherwise, out);
    }
    if (s.type === "TryCatch") {
      collectBodyAssignedTypes(s.tryBody, out);
      collectBodyAssignedTypes(s.catchBody, out);
    }
  }
}

/**
 * For each variable assigned in a loop body, check whether its body-assigned
 * type differs from its pre-loop type.  Only widen those variables (to the
 * unified type of pre-loop and body-assigned) — leave the rest at their
 * pre-loop types.
 */
function widenChangedLoopVars(
  cg: Codegen,
  body: IRStmt[],
  preLoop: Map<string, ItemType | undefined>
): void {
  const bodyTypes = new Map<string, ItemType>();
  collectBodyAssignedTypes(body, bodyTypes);
  for (const [id, bodyTy] of bodyTypes) {
    const pre = preLoop.get(id);
    if (pre && JSON.stringify(pre) !== JSON.stringify(bodyTy)) {
      cg.typeEnv.set({ id }, IType.unify(pre, bodyTy));
    }
  }
}

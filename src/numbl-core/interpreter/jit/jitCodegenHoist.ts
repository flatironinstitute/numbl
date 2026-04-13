/**
 * IR walkers for the JIT codegen hoist pass.
 *
 * These pure functions walk the JIT IR to collect information needed
 * for hoisting tensor aliases, struct field reads, and struct array
 * element reads to the top of the generated function.
 */

import type { JitExpr, JitStmt } from "./jitTypes.js";

// ── Hoisted alias types ────────────────────────────────────────────────

/**
 * Hoisted aliases for a tensor variable read/written in the loop body.
 * Maps the original variable name to local JS identifiers for its
 * .data, .data.length, .shape[0], and .shape[1].
 */
export interface HoistedAlias {
  data: string;
  len: string;
  d0: string;
  d1: string;
  maxDim: number;
  isWriteTarget: boolean;
  isParam: boolean;
}

// ── Key helpers ────────────────────────────────────────────────────────

export function structFieldKey(baseName: string, fieldName: string): string {
  return `${baseName}.${fieldName}`;
}

export function structArrayElementsKey(
  structVarName: string,
  structArrayFieldName: string
): string {
  return `${structVarName}.${structArrayFieldName}`;
}

// ── Tensor usage collection ────────────────────────────────────────────

export interface TensorUsage {
  maxReadDim: number;
  maxWriteDim: number;
  isReal: boolean;
}

/**
 * Walk the JIT IR collecting, for every variable that appears as the base
 * of an Index read, AssignIndex write, or AssignIndexRange write, the
 * maximum indexing arity and whether all uses are on a real tensor.
 */
export function collectTensorUsage(body: JitStmt[]): Map<string, TensorUsage> {
  const out = new Map<string, TensorUsage>();
  const bump = (
    name: string,
    isRead: boolean,
    dim: number,
    isReal: boolean
  ): void => {
    let u = out.get(name);
    if (!u) {
      u = { maxReadDim: 0, maxWriteDim: 0, isReal: true };
      out.set(name, u);
    }
    if (!isReal) u.isReal = false;
    if (isRead) {
      if (dim > u.maxReadDim) u.maxReadDim = dim;
    } else {
      if (dim > u.maxWriteDim) u.maxWriteDim = dim;
    }
  };

  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "Index":
        if (e.base.tag === "Var" && e.base.jitType.kind === "tensor") {
          const real = e.base.jitType.isComplex === false;
          bump(e.base.name, true, e.indices.length, real);
        }
        visitExpr(e.base);
        for (const idx of e.indices) visitExpr(idx);
        return;
      case "Binary":
        visitExpr(e.left);
        visitExpr(e.right);
        return;
      case "Unary":
        visitExpr(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      case "MemberRead":
        return;
      case "StructArrayMemberRead":
        visitExpr(e.indexExpr);
        return;
      default:
        return;
    }
  };

  const visitStmts = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "Assign":
        case "ExprStmt":
          visitExpr(s.expr);
          break;
        case "AssignIndex": {
          if (s.baseType.kind === "tensor") {
            bump(
              s.baseName,
              false,
              s.indices.length,
              s.baseType.isComplex === false
            );
          }
          for (const idx of s.indices) visitExpr(idx);
          visitExpr(s.value);
          break;
        }
        case "AssignIndexRange": {
          if (s.baseType.kind === "tensor") {
            bump(s.baseName, false, 1, s.baseType.isComplex === false);
          }
          if (s.srcType.kind === "tensor") {
            bump(s.srcBaseName, true, 1, s.srcType.isComplex === false);
          }
          visitExpr(s.dstStart);
          visitExpr(s.dstEnd);
          if (s.srcStart) visitExpr(s.srcStart);
          if (s.srcEnd) visitExpr(s.srcEnd);
          break;
        }
        case "MultiAssign":
          for (const a of s.args) visitExpr(a);
          break;
        case "If":
          visitExpr(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visitExpr(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visitExpr(s.start);
          if (s.step) visitExpr(s.step);
          visitExpr(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visitExpr(s.cond);
          visitStmts(s.body);
          break;
        default:
          break;
      }
    }
  };

  visitStmts(body);
  return out;
}

// ── Struct field read collection ───────────────────────────────────────

/**
 * Walk the JIT IR collecting all unique (baseName, fieldName) pairs
 * referenced by MemberRead nodes. The codegen hoists each pair as a
 * local alias at function entry.
 */
export function collectStructFieldReads(
  body: JitStmt[]
): Map<string, { baseName: string; fieldName: string }> {
  const out = new Map<string, { baseName: string; fieldName: string }>();

  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "MemberRead": {
        const key = structFieldKey(e.baseName, e.fieldName);
        if (!out.has(key)) {
          out.set(key, { baseName: e.baseName, fieldName: e.fieldName });
        }
        return;
      }
      case "Binary":
        visitExpr(e.left);
        visitExpr(e.right);
        return;
      case "Unary":
        visitExpr(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "Index":
        visitExpr(e.base);
        for (const idx of e.indices) visitExpr(idx);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      case "StructArrayMemberRead":
        visitExpr(e.indexExpr);
        return;
      default:
        return;
    }
  };

  const visitStmts = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "Assign":
        case "ExprStmt":
          visitExpr(s.expr);
          break;
        case "AssignIndex":
          for (const idx of s.indices) visitExpr(idx);
          visitExpr(s.value);
          break;
        case "AssignIndexRange":
          visitExpr(s.dstStart);
          visitExpr(s.dstEnd);
          if (s.srcStart) visitExpr(s.srcStart);
          if (s.srcEnd) visitExpr(s.srcEnd);
          break;
        case "MultiAssign":
          for (const a of s.args) visitExpr(a);
          break;
        case "If":
          visitExpr(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visitExpr(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visitExpr(s.start);
          if (s.step) visitExpr(s.step);
          visitExpr(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visitExpr(s.cond);
          visitStmts(s.body);
          break;
        default:
          break;
      }
    }
  };

  visitStmts(body);
  return out;
}

// ── Struct array element read collection ───────────────────────────────

/**
 * Walk the JIT IR collecting all unique (structVarName, structArrayFieldName)
 * pairs referenced by StructArrayMemberRead nodes.
 */
export function collectStructArrayElementReads(
  body: JitStmt[]
): Map<string, { structVarName: string; structArrayFieldName: string }> {
  const out = new Map<
    string,
    { structVarName: string; structArrayFieldName: string }
  >();

  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "StructArrayMemberRead": {
        const key = structArrayElementsKey(
          e.structVarName,
          e.structArrayFieldName
        );
        if (!out.has(key)) {
          out.set(key, {
            structVarName: e.structVarName,
            structArrayFieldName: e.structArrayFieldName,
          });
        }
        visitExpr(e.indexExpr);
        return;
      }
      case "Binary":
        visitExpr(e.left);
        visitExpr(e.right);
        return;
      case "Unary":
        visitExpr(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "Index":
        visitExpr(e.base);
        for (const idx of e.indices) visitExpr(idx);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      default:
        return;
    }
  };

  const visitStmts = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "Assign":
        case "ExprStmt":
          visitExpr(s.expr);
          break;
        case "AssignIndex":
          for (const idx of s.indices) visitExpr(idx);
          visitExpr(s.value);
          break;
        case "AssignIndexRange":
          visitExpr(s.dstStart);
          visitExpr(s.dstEnd);
          if (s.srcStart) visitExpr(s.srcStart);
          if (s.srcEnd) visitExpr(s.srcEnd);
          break;
        case "MultiAssign":
          for (const a of s.args) visitExpr(a);
          break;
        case "If":
          visitExpr(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visitExpr(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visitExpr(s.start);
          if (s.step) visitExpr(s.step);
          visitExpr(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visitExpr(s.cond);
          visitStmts(s.body);
          break;
        default:
          break;
      }
    }
  };

  visitStmts(body);
  return out;
}

/**
 * IR walkers for the JIT codegen hoist pass.
 *
 * These pure functions walk the JIT IR to collect information needed
 * for hoisting tensor aliases, struct field reads, and struct array
 * element reads to the top of the generated function.
 */

import type { JitExpr, JitStmt } from "../jitTypes.js";

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
      case "FuncHandleCall":
      case "UserDispatchCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      case "RangeSliceRead":
        bump(e.baseName, true, 1, true);
        visitExpr(e.start);
        if (e.end) visitExpr(e.end);
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
        case "AssignIndexCol": {
          // dst is written with 2-D arity so the hoist emits the $d0
          // alias (rows). src is read as a whole tensor so we only need
          // .data and .length (arity 1 is sufficient to get both).
          if (s.baseType.kind === "tensor") {
            bump(s.baseName, false, 2, s.baseType.isComplex === false);
          }
          if (s.srcType.kind === "tensor") {
            bump(s.srcBaseName, true, 1, s.srcType.isComplex === false);
          }
          visitExpr(s.colIndex);
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

/**
 * Walk the JIT IR collecting every base name that is the target of an
 * `AssignMember` stmt (stage 22 struct field write). Used by the
 * codegen to emit a one-time `$h.structUnshare_h(s)` at function entry
 * for any such name that is also a function parameter — this preserves
 * MATLAB value semantics when structs are passed by value and the
 * callee mutates a field.
 */
export function collectStructMemberWrites(body: JitStmt[]): Set<string> {
  const out = new Set<string>();
  const visitStmts = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "AssignMember":
          out.add(s.baseName);
          break;
        case "If":
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) visitStmts(eib.body);
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
        case "While":
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

/**
 * Walk the JIT IR collecting every name that is the target of any
 * plain `Assign` stmt. Used by the stage-12 struct-field-read hoist to
 * decide whether a struct-typed param is safe to hoist: if the body
 * ever reassigns the name (e.g. `s = [];` promoted to struct, or `s =
 * struct()` inside the loop), the pre-loop hoisted field aliases would
 * be stale and must be disabled.
 */
export function collectPlainAssignTargets(body: JitStmt[]): Set<string> {
  const out = new Set<string>();
  const visitStmts = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "Assign":
          out.add(s.name);
          break;
        case "If":
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) visitStmts(eib.body);
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
        case "While":
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
      case "FuncHandleCall":
      case "UserDispatchCall":
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
      case "RangeSliceRead":
        visitExpr(e.start);
        if (e.end) visitExpr(e.end);
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
        case "AssignIndexCol":
          visitExpr(s.colIndex);
          break;
        case "AssignMember":
          visitExpr(s.value);
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
      case "FuncHandleCall":
      case "UserDispatchCall":
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
        case "AssignIndexCol":
          visitExpr(s.colIndex);
          break;
        case "AssignMember":
          visitExpr(s.value);
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

/**
 * IR walking helpers — read-only traversal of expression trees and
 * the top-level expressions inside a statement. Used by the liveness
 * analyzer (and any future IR-walking pass) to enumerate sub-
 * expressions without each call site re-discriminating on `IRExpr`
 * kinds.
 */

import type { IRExpr, IRStmt } from "./ir.js";

/** Visit every sub-expression of `e`, including `e` itself. Walks
 *  pre-order: the outer node is visited first, then children. Pure
 *  read-only — `visit` should not mutate the IR. */
export function forEachSubExpr(e: IRExpr, visit: (sub: IRExpr) => void): void {
  visit(e);
  switch (e.kind) {
    case "NumLit":
    case "ImagLit":
    case "StringLit":
    case "Var":
      return;
    case "TensorBuild":
      for (const el of e.elements) forEachSubExpr(el, visit);
      return;
    case "TensorConcat":
      for (const row of e.cells) {
        for (const cell of row) forEachSubExpr(cell, visit);
      }
      return;
    case "CellLit":
      for (const el of e.elements) forEachSubExpr(el, visit);
      return;
    case "CellEmpty":
      for (const d of e.dims) forEachSubExpr(d, visit);
      return;
    case "CellIndexLoad":
      forEachSubExpr(e.base, visit);
      for (const i of e.indices) forEachSubExpr(i, visit);
      return;
    case "Binary":
      forEachSubExpr(e.left, visit);
      forEachSubExpr(e.right, visit);
      return;
    case "Unary":
      forEachSubExpr(e.operand, visit);
      return;
    case "Call":
      for (const a of e.args) forEachSubExpr(a, visit);
      return;
    case "HandleLit":
      for (const c of e.captures) forEachSubExpr(c.value, visit);
      return;
    case "HandleCaptureLoad":
      forEachSubExpr(e.base, visit);
      return;
    case "StructLit":
      for (const f of e.fields) forEachSubExpr(f.value, visit);
      return;
    case "MemberLoad":
      forEachSubExpr(e.base, visit);
      return;
    case "IndexLoad":
      forEachSubExpr(e.base, visit);
      for (const i of e.indices) forEachSubExpr(i, visit);
      return;
    case "IndexSlice":
      forEachSubExpr(e.base, visit);
      for (const slot of e.index) {
        if (slot.kind === "Range") {
          forEachSubExpr(slot.start, visit);
          forEachSubExpr(slot.step, visit);
          forEachSubExpr(slot.end, visit);
        } else if (slot.kind === "Scalar") {
          forEachSubExpr(slot.expr, visit);
        } else if (slot.kind === "IndexVec") {
          forEachSubExpr(slot.expr, visit);
        } else if (slot.kind === "LogicalMask") {
          forEachSubExpr(slot.expr, visit);
        }
      }
      return;
    case "EndRef":
      return;
    case "MakeRange":
      forEachSubExpr(e.start, visit);
      forEachSubExpr(e.step, visit);
      forEachSubExpr(e.end, visit);
      return;
  }
}

/** Visit every top-level expression in `s` (the expressions that
 *  live directly under the statement node — NOT recursing into
 *  nested control-flow bodies). The visitor can then call
 *  `forEachSubExpr` to walk into each. */
export function forEachTopLevelExpr(
  s: IRStmt,
  visit: (e: IRExpr) => void
): void {
  switch (s.kind) {
    case "ExprStmt":
      visit(s.expr);
      return;
    case "Assign":
      visit(s.expr);
      return;
    case "If":
      visit(s.cond);
      return;
    case "While":
      visit(s.cond);
      return;
    case "For":
      visit(s.start);
      visit(s.end);
      return;
    case "ReturnFromFunction":
    case "Break":
    case "Continue":
    case "TypeComment":
      return;
    case "MemberStore":
      // The `base` is always a `Var` already, but visiting it through
      // the same callback keeps owned-use detection uniform with
      // every other statement kind.
      visit(s.base);
      visit(s.rhs);
      return;
    case "MultiAssignCall":
      for (const a of s.args) visit(a);
      return;
    case "IndexStore":
      visit(s.base);
      for (const i of s.indices) visit(i);
      visit(s.rhs);
      return;
    case "CellIndexStore":
      visit(s.base);
      for (const i of s.indices) visit(i);
      visit(s.rhs);
      return;
    case "IndexSliceStore":
      visit(s.base);
      for (const slot of s.index) {
        if (slot.kind === "Range") {
          visit(slot.start);
          visit(slot.step);
          visit(slot.end);
        } else if (slot.kind === "Scalar") {
          visit(slot.expr);
        } else if (slot.kind === "IndexVec") {
          visit(slot.expr);
        } else if (slot.kind === "LogicalMask") {
          visit(slot.expr);
        }
      }
      visit(s.rhs);
      return;
  }
}

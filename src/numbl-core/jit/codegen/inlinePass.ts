/**
 * IR-to-IR rewrite for the C backend: substitutes a single-use
 * producer Assign's RHS into its consumer's RHS and drops the
 * producer.
 *
 * After ANF normalization every owned-producing tensor expression
 * lives in its own `_mtoc2_t<N>` temp. Without inlining, a chain
 * like
 *
 *     _t8  = rx .^ 2
 *     _t9  = ry .^ 2
 *     _t10 = _t8 + _t9
 *     r2   = _t10 + 0.01
 *
 * emits four separate fused loops with three intermediate allocs.
 * After inlining, the chain collapses to
 *
 *     r2 = (rx .^ 2 + ry .^ 2 + 0.01)
 *
 * which the elementwise-fused emitter (`emitTensorFused.ts`) renders
 * as a single loop.
 *
 * Gates (all must hold to inline producer P into consumer C):
 *
 *   1. P is a fusable Assign (`isFusableAssign`): real multi-element
 *      tensor target, pure-elementwise RHS, operand shapes matching
 *      the target.
 *   2. P's LHS cName is read exactly once in the whole function
 *      (nested bodies included). Function outputs get a +1 bump so
 *      they are never dropped. Counting function-wide, not per body,
 *      is what makes inlining inside loop / if bodies safe: a
 *      body-local count misses reads after the loop.
 *   3. C is the first statement after P in the same body that reads
 *      P's LHS, and it is an Assign. Statements in between must not
 *      write P's LHS or any Var P's RHS reads, and must not be
 *      control flow. C itself may write such a Var: an Assign
 *      evaluates its whole RHS before storing.
 *   4. The read in C is in a slot position (Binary / Unary / Call
 *      operand), and C is still fusable after substitution. The
 *      latter is an ownership rule: a non-fused consumer would get a
 *      nested owned operand that nothing frees.
 *
 * Use counts are stable under substitution (a single-use producer's
 * reads move into its consumer; none is duplicated or dropped), so
 * one forward sweep per body suffices: a consumer rewritten at index
 * j is itself considered as a producer when the sweep reaches j,
 * which collapses chains of any depth.
 *
 * The pass never mutates its input. The Lowerer's specialization
 * cache is shared by the JS and C backends, so changed functions and
 * statements are fresh objects; untouched ones are shared.
 */

import type {
  Assign,
  IRExpr,
  IRFunc,
  IRStmt,
  IRProgram,
} from "../lowering/ir.js";
import { forEachSubExpr, forEachTopLevelExpr } from "../lowering/walk.js";
import { isFusableAssign } from "./emitTensorFused.js";

/** Returns a new program with every function inlined. */
export function inlinePass(prog: IRProgram): IRProgram {
  const functions = new Map<string, IRFunc>();
  for (const [k, fn] of prog.functions) functions.set(k, inlineFunc(fn));
  return { functions };
}

/** Returns `fn` itself when nothing was inlined. */
export function inlineFunc(fn: IRFunc): IRFunc {
  const counts = computeUseCounts(fn.body, fn.cOutputs);
  const body = inlineInBody(fn.body, counts);
  return body === fn.body ? fn : { ...fn, body };
}

/** Inline within one body, children first. Returns `stmts` itself
 *  when nothing changed. */
function inlineInBody(
  stmts: IRStmt[],
  counts: ReadonlyMap<string, number>
): IRStmt[] {
  let changed = false;
  const cur: IRStmt[] = stmts.map(s => {
    const r = inlineNested(s, counts);
    if (r !== s) changed = true;
    return r;
  });

  const removed = new Array<boolean>(cur.length).fill(false);
  for (let i = 0; i < cur.length; i++) {
    const prod = cur[i];
    if (prod.kind !== "Assign") continue;
    if (counts.get(prod.cName) !== 1) continue;
    if (!isFusableAssign(prod)) continue;

    const prodFreeVars = collectFreeVarCNames(prod.expr);
    for (let j = i + 1; j < cur.length; j++) {
      const s = cur[j];
      if (isControlFlow(s)) break;
      if (stmtReadsVar(s, prod.cName)) {
        if (
          s.kind === "Assign" &&
          !appearsInNonSlotPosition(s.expr, prod.cName)
        ) {
          const consumer: Assign = {
            ...s,
            expr: substituteVar(s.expr, prod.cName, prod.expr),
          };
          if (isFusableAssign(consumer)) {
            cur[j] = consumer;
            removed[i] = true;
            changed = true;
          }
        }
        break;
      }
      if (stmtWrites(s, prod.cName, prodFreeVars)) break;
    }
  }

  if (!changed) return stmts;
  return cur.filter((_, i) => !removed[i]);
}

/** Recurse into an If / While / For body. Returns `s` itself when
 *  nothing changed. */
function inlineNested(s: IRStmt, counts: ReadonlyMap<string, number>): IRStmt {
  if (s.kind === "If") {
    const thenBody = inlineInBody(s.thenBody, counts);
    const elseBody = inlineInBody(s.elseBody, counts);
    if (thenBody === s.thenBody && elseBody === s.elseBody) return s;
    return { ...s, thenBody, elseBody };
  }
  if (s.kind === "While" || s.kind === "For") {
    const body = inlineInBody(s.body, counts);
    return body === s.body ? s : { ...s, body };
  }
  return s;
}

/** Var reads per cName across the whole function body, nested bodies
 *  included. Outputs get a +1 bump so they never reach count 1. */
function computeUseCounts(
  stmts: ReadonlyArray<IRStmt>,
  outputs: ReadonlyArray<string>
): Map<string, number> {
  const counts = new Map<string, number>();
  const bump = (cName: string) =>
    counts.set(cName, (counts.get(cName) ?? 0) + 1);
  for (const cName of outputs) bump(cName);
  for (const s of stmts) countVarRefsInStmt(s, bump);
  return counts;
}

function countVarRefsInStmt(s: IRStmt, bump: (cName: string) => void): void {
  forEachTopLevelExpr(s, e =>
    forEachSubExpr(e, sub => {
      if (sub.kind === "Var") bump(sub.cName);
    })
  );
  if (s.kind === "If") {
    for (const sub of s.thenBody) countVarRefsInStmt(sub, bump);
    for (const sub of s.elseBody) countVarRefsInStmt(sub, bump);
  } else if (s.kind === "While" || s.kind === "For") {
    for (const sub of s.body) countVarRefsInStmt(sub, bump);
  }
}

function isControlFlow(s: IRStmt): boolean {
  return (
    s.kind === "If" ||
    s.kind === "While" ||
    s.kind === "For" ||
    s.kind === "Break" ||
    s.kind === "Continue" ||
    s.kind === "ReturnFromFunction"
  );
}

function collectFreeVarCNames(e: IRExpr): Set<string> {
  const out = new Set<string>();
  forEachSubExpr(e, sub => {
    if (sub.kind === "Var") out.add(sub.cName);
  });
  return out;
}

/** True iff `s` writes `cName` or any name in `freeVars`: Assign LHS,
 *  base of an indexed / member / cell store, or a bound output of a
 *  MultiAssignCall. */
function stmtWrites(
  s: IRStmt,
  cName: string,
  freeVars: ReadonlySet<string>
): boolean {
  const hits = (n: string) => n === cName || freeVars.has(n);
  switch (s.kind) {
    case "Assign":
      return hits(s.cName);
    case "IndexStore":
    case "IndexSliceStore":
    case "MemberStore":
    case "CellIndexStore":
      return hits(s.base.cName);
    case "MultiAssignCall":
      return s.outputs.some(o => o.binding !== null && hits(o.binding.cName));
    default:
      return false;
  }
}

/** True iff `s` reads `cName` as a Var in its top-level expressions
 *  (store bases included). */
function stmtReadsVar(s: IRStmt, cName: string): boolean {
  let found = false;
  forEachTopLevelExpr(s, e => {
    forEachSubExpr(e, sub => {
      if (sub.kind === "Var" && sub.cName === cName) found = true;
    });
  });
  return found;
}

/** True iff `cName` appears in `e` in a position the substitution
 *  won't reach (and therefore the producer's value would still be
 *  referenced after inlining). Slot positions where substitution
 *  DOES reach: Binary operands, Unary operand, Call args. Non-slot:
 *  IndexLoad / IndexSlice / MemberLoad base, TensorBuild /
 *  TensorConcat cells, HandleLit captures, StructLit fields. */
function appearsInNonSlotPosition(e: IRExpr, cName: string): boolean {
  switch (e.kind) {
    case "Var":
    case "NumLit":
    case "ImagLit":
    case "StringLit":
    case "EndRef":
      return false;
    case "Binary":
      return (
        appearsInNonSlotPosition(e.left, cName) ||
        appearsInNonSlotPosition(e.right, cName)
      );
    case "Unary":
      return appearsInNonSlotPosition(e.operand, cName);
    case "Call":
      for (const a of e.args) {
        if (appearsInNonSlotPosition(a, cName)) return true;
      }
      return false;
    case "IndexLoad":
      if (e.base.kind === "Var" && e.base.cName === cName) return true;
      if (appearsInNonSlotPosition(e.base, cName)) return true;
      for (const i of e.indices) {
        if (appearsInNonSlotPosition(i, cName)) return true;
      }
      return false;
    case "IndexSlice":
      if (e.base.kind === "Var" && e.base.cName === cName) return true;
      if (appearsInNonSlotPosition(e.base, cName)) return true;
      for (const slot of e.index) {
        if (slot.kind === "Colon") continue;
        const slotExprs: IRExpr[] =
          slot.kind === "Range"
            ? [slot.start, slot.step, slot.end]
            : [slot.expr];
        for (const sub of slotExprs) {
          if (sub.kind === "Var" && sub.cName === cName) return true;
          if (appearsInNonSlotPosition(sub, cName)) return true;
        }
      }
      return false;
    case "MemberLoad":
      if (e.base.kind === "Var" && e.base.cName === cName) return true;
      return appearsInNonSlotPosition(e.base, cName);
    case "TensorBuild": {
      for (const el of e.elements) {
        if (el.kind === "Var" && el.cName === cName) return true;
        if (appearsInNonSlotPosition(el, cName)) return true;
      }
      return false;
    }
    case "TensorConcat": {
      for (const row of e.cells) {
        for (const cell of row) {
          if (cell.kind === "Var" && cell.cName === cName) return true;
          if (appearsInNonSlotPosition(cell, cName)) return true;
        }
      }
      return false;
    }
    case "CellLit": {
      for (const el of e.elements) {
        if (el.kind === "Var" && el.cName === cName) return true;
        if (appearsInNonSlotPosition(el, cName)) return true;
      }
      return false;
    }
    case "CellEmpty": {
      for (const d of e.dims) {
        if (d.kind === "Var" && d.cName === cName) return true;
        if (appearsInNonSlotPosition(d, cName)) return true;
      }
      return false;
    }
    case "CellIndexLoad":
      if (e.base.kind === "Var" && e.base.cName === cName) return true;
      if (appearsInNonSlotPosition(e.base, cName)) return true;
      for (const i of e.indices) {
        if (appearsInNonSlotPosition(i, cName)) return true;
      }
      return false;
    case "StructLit": {
      for (const f of e.fields) {
        if (f.value.kind === "Var" && f.value.cName === cName) return true;
        if (appearsInNonSlotPosition(f.value, cName)) return true;
      }
      return false;
    }
    case "HandleLit": {
      for (const c of e.captures) {
        if (c.value.kind === "Var" && c.value.cName === cName) return true;
        if (appearsInNonSlotPosition(c.value, cName)) return true;
      }
      return false;
    }
    case "HandleCaptureLoad":
      if (e.base.kind === "Var" && e.base.cName === cName) return true;
      return appearsInNonSlotPosition(e.base, cName);
    case "MakeRange":
      return (
        appearsInNonSlotPosition(e.start, cName) ||
        appearsInNonSlotPosition(e.step, cName) ||
        appearsInNonSlotPosition(e.end, cName)
      );
  }
}

/** Pure tree rewrite: replace every `Var(cName === target)` in `e`
 *  with `replacement`. Does NOT descend into non-slot positions
 *  (IndexLoad/IndexSlice/MemberLoad base, TensorBuild/TensorConcat
 *  cells, StructLit fields, HandleLit captures) — those positions
 *  are already filtered by `appearsInNonSlotPosition` before this
 *  function is called. Preserves identity for unchanged subtrees so
 *  the substitution is cheap when nothing matches. */
function substituteVar(e: IRExpr, target: string, replacement: IRExpr): IRExpr {
  switch (e.kind) {
    case "Var":
      return e.cName === target ? replacement : e;
    case "NumLit":
    case "ImagLit":
    case "StringLit":
    case "EndRef":
    case "TensorBuild":
    case "TensorConcat":
    case "CellLit":
    case "CellEmpty":
    case "CellIndexLoad":
    case "MakeRange":
    case "MemberLoad":
    case "StructLit":
    case "HandleLit":
    case "HandleCaptureLoad":
    case "IndexSlice":
      return e;
    case "Binary": {
      const left = substituteVar(e.left, target, replacement);
      const right = substituteVar(e.right, target, replacement);
      if (left === e.left && right === e.right) return e;
      return { ...e, left, right };
    }
    case "Unary": {
      const operand = substituteVar(e.operand, target, replacement);
      if (operand === e.operand) return e;
      return { ...e, operand };
    }
    case "Call": {
      let changed = false;
      const newArgs = e.args.map(a => {
        const sub = substituteVar(a, target, replacement);
        if (sub !== a) changed = true;
        return sub;
      });
      if (!changed) return e;
      return { ...e, args: newArgs };
    }
    case "IndexLoad": {
      // Don't touch the base; substitute in indices.
      let changed = false;
      const newIndices = e.indices.map(i => {
        const sub = substituteVar(i, target, replacement);
        if (sub !== i) changed = true;
        return sub;
      });
      if (!changed) return e;
      return { ...e, indices: newIndices };
    }
  }
}

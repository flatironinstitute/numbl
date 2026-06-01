/**
 * `--inline-temps` IR-to-IR rewrite: substitutes a single-use
 * producer Assign's RHS into its consumer's RHS and removes the
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
 * After inlining (this pass), the chain collapses to
 *
 *     r2 = (rx .^ 2 + ry .^ 2 + 0.01)
 *
 * which the elementwise-fused emitter (`emitTensorFused.ts`) renders
 * as a single iter loop. No intermediate tensors, no extra passes
 * through memory.
 *
 * Gates (must all hold to inline producer P into consumer C):
 *
 *   1. P is an Assign producing a real-double multi-element tensor.
 *   2. P's RHS is pure elementwise — only NumLit / Var / Binary /
 *      Unary / Call to a builtin that has `perSlotC`. No
 *      TensorBuild / TensorConcat / IndexLoad / IndexSlice /
 *      MemberLoad / HandleLit / etc.
 *   3. P's LHS cName is used exactly once in the body. Function
 *      output cNames get a `+1` protective bump so the returned
 *      value is never elided.
 *   4. The single use is in another Assign's RHS, in a slot
 *      position (subexpression of Binary / Unary / Call args). NOT
 *      as the base of an IndexLoad / IndexSlice / MemberLoad, NOT
 *      inside a TensorBuild / TensorConcat / StructLit / HandleLit.
 *   5. C is also fusable (same predicate as `isFusableAssign`).
 *   6. P and C are at the same body level (no inlining across
 *      If / While / For — nested bodies are inlined separately).
 *   7. No statement between P and C writes to P's LHS or to any
 *      free Var in P's RHS, and no statement reads P's LHS except
 *      C. Control flow (If / While / For / Break / Continue /
 *      ReturnFromFunction) ends the safe window.
 *
 * Algorithm: per body, iterate `inlineOnePass` to fixed point.
 * Each pass scans producers in source order, finds the first
 * inlinable pair, substitutes, and returns the new body. Chained
 * inlining (`a → b → c`) needs multiple iterations because each
 * substitution can flip a previously-multi-use cName to single-use;
 * we cap at 32 iterations defensively.
 *
 * Preserves: post-inlining the IR is still well-typed (each node's
 * `.ty` is unchanged — the substitution copies the producer's RHS
 * subtree verbatim); the ANF invariant for owned producers still
 * holds because we only inline producers whose RHS is NOT itself
 * an owned producer (no TensorBuild / Call-to-direct-owned-builtin
 * / etc.); and the `isFusableAssign` predicate that gates the
 * emit-path routing still accepts the consumer after substitution
 * (shape checks compose under inlining — see the design notes in
 * `emitTensorFused.ts`).
 */

import type { Assign, IRExpr, IRStmt, IRProgram } from "../lowering/ir.js";
import { forEachSubExpr, forEachTopLevelExpr } from "../lowering/walk.js";
import { isFusableAssign } from "./emitTensorFused.js";

/** Top-level entry. Mutates `prog.topLevelStmts` and each function's
 *  `body` in place when inlining fires. */
export function inlinePass(prog: IRProgram): void {
  prog.topLevelStmts = inlineInBody(prog.topLevelStmts, new Set());
  for (const fn of prog.functions.values()) {
    const protectedNames = new Set<string>(fn.cOutputs);
    fn.body = inlineInBody(fn.body, protectedNames);
  }
}

/** Iterate `inlineOnePass` to fixed point over a single body.
 *  Recurses into nested control-flow bodies first so each child
 *  stabilizes before its parent re-counts uses. */
function inlineInBody(
  stmts: IRStmt[],
  protectedNames: ReadonlySet<string>
): IRStmt[] {
  recurseInlineNested(stmts);

  let cur = stmts;
  for (let iter = 0; iter < 32; iter++) {
    const next = inlineOnePass(cur, protectedNames);
    if (next === cur) return cur;
    cur = next;
  }
  return cur;
}

/** Recurse into If / While / For body fields and run inlining on
 *  each. Each nested body gets an empty `protectedNames` set —
 *  outputs are at the function-body level. */
function recurseInlineNested(stmts: IRStmt[]): void {
  for (const s of stmts) {
    if (s.kind === "If") {
      s.thenBody = inlineInBody(s.thenBody, new Set());
      s.elseBody = inlineInBody(s.elseBody, new Set());
    } else if (s.kind === "While" || s.kind === "For") {
      s.body = inlineInBody(s.body, new Set());
    }
  }
}

/** One linear forward sweep. Performs AT MOST ONE substitution per
 *  call: scans for the first inlinable (producer, consumer) pair
 *  and rewrites them. The fixed-point loop iterates this until
 *  quiescent. Single-substitution-per-pass keeps the bookkeeping
 *  trivial; cost is multiple passes for deep chains. */
function inlineOnePass(
  stmts: IRStmt[],
  protectedNames: ReadonlySet<string>
): IRStmt[] {
  const useCounts = computeUseCounts(stmts, protectedNames);

  for (let i = 0; i < stmts.length; i++) {
    const producer = stmts[i];
    if (!isInlinableProducer(producer)) continue;
    const prod = producer as Assign;
    if (protectedNames.has(prod.cName)) continue;
    if (useCounts.get(prod.cName) !== 1) continue;

    const prodFreeVars = collectFreeVarCNames(prod.expr);

    // Forward scan for the consumer. Bails on intervening writes or
    // control flow.
    let consumer: Assign | null = null;
    let bail = false;
    for (let j = i + 1; j < stmts.length; j++) {
      const s = stmts[j];
      if (isControlFlow(s)) {
        bail = true;
        break;
      }
      if (stmtWrites(s, prod.cName, prodFreeVars)) {
        bail = true;
        break;
      }
      if (stmtReadsAsMultiElemVar(s, prod.cName)) {
        if (s.kind !== "Assign") {
          bail = true;
          break;
        }
        consumer = s;
        break;
      }
    }
    if (bail || consumer === null) continue;

    if (!isFusableAssign(consumer)) continue;
    if (appearsInNonSlotPosition(consumer.expr, prod.cName)) continue;

    // Substitute. The pure tree rewrite preserves types (each Var's
    // `.ty` and the substituted subtree's `.ty` are identical).
    const newExpr = substituteVar(consumer.expr, prod.cName, prod.expr);
    const newConsumer: IRStmt = { ...consumer, expr: newExpr };

    const out: IRStmt[] = [];
    for (const s of stmts) {
      if (s === producer) continue;
      out.push(s === consumer ? newConsumer : s);
    }
    return out;
  }

  return stmts;
}

/** Count Var occurrences per cName across the entire body (including
 *  nested control-flow bodies — a Var read inside an `if` still
 *  counts as a use of a top-level Assign's LHS). Function output
 *  cNames in `protectedNames` get a +1 bump so they never reach
 *  count-1 and get inlined out. */
function computeUseCounts(
  stmts: ReadonlyArray<IRStmt>,
  protectedNames: ReadonlySet<string>
): Map<string, number> {
  const counts = new Map<string, number>();
  const bump = (cName: string) =>
    counts.set(cName, (counts.get(cName) ?? 0) + 1);
  for (const cName of protectedNames) bump(cName);
  for (const s of stmts) countVarRefsInStmt(s, bump);
  return counts;
}

/** Bump every Var read in `s`'s top-level expressions, recursing
 *  through nested If/While/For bodies. Reuses the shared walkers in
 *  `walk.ts` so any new IR-expr kind (e.g. a new IndexSlice slot
 *  variant) is picked up automatically. */
function countVarRefsInStmt(s: IRStmt, bump: (cName: string) => void): void {
  const visitExpr = (e: IRExpr): void => {
    forEachSubExpr(e, sub => {
      if (sub.kind === "Var") bump(sub.cName);
    });
  };
  if (s.kind === "If") {
    visitExpr(s.cond);
    for (const sub of s.thenBody) countVarRefsInStmt(sub, bump);
    for (const sub of s.elseBody) countVarRefsInStmt(sub, bump);
    return;
  }
  if (s.kind === "While") {
    visitExpr(s.cond);
    for (const sub of s.body) countVarRefsInStmt(sub, bump);
    return;
  }
  if (s.kind === "For") {
    visitExpr(s.start);
    visitExpr(s.end);
    for (const sub of s.body) countVarRefsInStmt(sub, bump);
    return;
  }
  forEachTopLevelExpr(s, visitExpr);
}

/** Gate 1+2: producer is a multi-element real-double Assign whose
 *  RHS is pure-elementwise AND every multi-element Var in the RHS
 *  matches the producer's static shape (i.e. P is itself fusable).
 *  The shape check matters: a broadcast Assign like
 *  `rx = col - row` has a pure-elementwise RHS but its operand
 *  shapes don't match its target shape — substituting it into a
 *  consumer would introduce mixed-shape Vars and force the consumer
 *  off the fused path, leaving nested helper calls whose
 *  intermediates leak. `isFusableAssign` already wraps both
 *  predicates. */
function isInlinableProducer(s: IRStmt): boolean {
  if (s.kind !== "Assign") return false;
  return isFusableAssign(s);
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

/** Free Var cNames in `e`. Used to detect intervening writes that
 *  would invalidate inlining. */
function collectFreeVarCNames(e: IRExpr): Set<string> {
  const out = new Set<string>();
  forEachSubExpr(e, sub => {
    if (sub.kind === "Var") out.add(sub.cName);
  });
  return out;
}

/** True iff `s` writes to `cName` or to any name in `freeVars`.
 *  "Writes" means LHS of an Assign, base of an IndexStore /
 *  IndexSliceStore / MemberStore, or an output slot of a
 *  MultiAssignCall. */
function stmtWrites(
  s: IRStmt,
  cName: string,
  freeVars: ReadonlySet<string>
): boolean {
  if (s.kind === "Assign") {
    return s.cName === cName || freeVars.has(s.cName);
  }
  if (s.kind === "IndexStore" || s.kind === "IndexSliceStore") {
    return s.base.cName === cName || freeVars.has(s.base.cName);
  }
  if (s.kind === "MemberStore") {
    return s.base.cName === cName || freeVars.has(s.base.cName);
  }
  if (s.kind === "MultiAssignCall") {
    for (const o of s.outputs) {
      if (o.binding === null) continue;
      if (o.binding.cName === cName || freeVars.has(o.binding.cName)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

/** True iff `s` reads `cName` as a Var anywhere in its top-level
 *  expressions, OR uses `cName` as the base of an IndexStore /
 *  IndexSliceStore / MemberStore (a write to `cName`, conservatively
 *  treated as "consumes" so the forward scan terminates without
 *  inlining). The forward scan is bounded above by `isControlFlow`
 *  so If/While/For/Break/Continue/ReturnFromFunction never reach
 *  here. */
function stmtReadsAsMultiElemVar(s: IRStmt, cName: string): boolean {
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

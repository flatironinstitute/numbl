/**
 * Conservative definite-assignment check for the C backend.
 *
 * The C-JIT predeclares every function local at the top of the emitted
 * function with a default value (`double x = 0.0;`, an empty owned struct,
 * …). C has no "unassigned" sentinel, so a variable that is assigned ONLY
 * on a path that doesn't run — e.g. inside a `for` loop with an empty range,
 * or one arm of an `if` with no matching `else` — leaks its default value
 * when read afterwards. The interpreter and JS-JIT instead leave the binding
 * absent: the interpreter raises "Undefined function or variable", and the
 * JS-JIT returns `undefined` which fails to marshal back and bails to the
 * interpreter. So at `--opt 2` a use-before-assign silently returns 0 while
 * `--opt 0/1` (correctly) raise.
 *
 * Rather than model MATLAB's binding-absent semantics in C, we DECLINE: if a
 * variable that is assigned somewhere in the function may be read before it
 * is definitely assigned, throw `UnsupportedConstruct`. The C-JIT executor
 * catches that and falls back to the JS-JIT / interpreter, which handle the
 * case correctly. Declining is always safe — it can only cost a fallback,
 * never correctness — so the analysis is deliberately conservative:
 *
 *   - On the READ side it may under-approximate (a missed read kind just
 *     means we don't catch that particular use-before-assign — still safe).
 *   - On the ASSIGNMENT side it must not under-approximate, or we'd flag a
 *     valid program; the `assert_jit c` integration tests guard against
 *     over-declining real code.
 */

import type { IRStmt, IRFunc, Var } from "./ir.js";
import { UnsupportedConstruct } from "./errors.js";

/** Collect the source names of every variable READ within a node.
 *  Generic structural walk: any `{ kind: "Var" }` contributes its name.
 *  `ty` / `span` subtrees are skipped (they carry no reads), and handle
 *  literals are not descended into (their params/captures rebind names,
 *  so descending would produce false positives). */
function collectReads(node: unknown, out: Set<string>): void {
  if (node === null || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const el of node) collectReads(el, out);
    return;
  }
  const kind = (node as { kind?: unknown }).kind;
  if (typeof kind === "string") {
    if (kind === "Var") {
      out.add((node as Var).name);
      return;
    }
    // Do not descend into a handle's body: its parameters and captured
    // names are bound at the handle, not in this scope.
    if (kind === "HandleLit" || kind === "HandleCaptureLoad") return;
  }
  for (const key of Object.keys(node as Record<string, unknown>)) {
    if (key === "ty" || key === "span") continue;
    collectReads((node as Record<string, unknown>)[key], out);
  }
}

/** Every source name assigned anywhere in `stmts` (any assignment form).
 *  These are the locals whose reads must be definitely assigned. */
function collectAssignedNames(stmts: IRStmt[], out: Set<string>): void {
  for (const s of stmts) {
    switch (s.kind) {
      case "Assign":
        out.add(s.name);
        break;
      case "MultiAssignCall":
        for (const o of s.outputs) if (o.binding) out.add(o.binding.name);
        break;
      case "IndexStore":
      case "IndexSliceStore":
      case "CellIndexStore":
      case "MemberStore":
        out.add(s.base.name);
        break;
      case "For":
        out.add(s.varName);
        collectAssignedNames(s.body, out);
        break;
      case "While":
        collectAssignedNames(s.body, out);
        break;
      case "If":
        collectAssignedNames(s.thenBody, out);
        collectAssignedNames(s.elseBody, out);
        break;
      default:
        break;
    }
  }
}

function setIntersect(a: Set<string>, b: Set<string>): Set<string> {
  const out = new Set<string>();
  for (const x of a) if (b.has(x)) out.add(x);
  return out;
}

/** True when a `for` loop provably runs at least one iteration — i.e. its
 *  start/end are numeric literals and the iteration count is ≥ 1. Only then
 *  are the loop body's (and the loop variable's) assignments definite after
 *  the loop. A variable-bound range (`1:n`) is NOT provable: it may be empty,
 *  so reads of body-assigned vars after the loop must still be declined (the
 *  interpreter reproduces MATLAB's undefined-variable error when n < 1). */
function forProvablyRunsOnce(s: Extract<IRStmt, { kind: "For" }>): boolean {
  if (s.start.kind !== "NumLit" || s.end.kind !== "NumLit") return false;
  const start = s.start.value;
  const end = s.end.value;
  const step = s.step;
  if (!Number.isFinite(start) || !Number.isFinite(end) || step === 0) {
    return false;
  }
  // Iteration count for `start:step:end` (MATLAB colon semantics).
  const count = Math.floor((end - start) / step) + 1;
  return count >= 1;
}

/**
 * Walk `stmts` threading the set of definitely-assigned names. Returns the
 * set as of the end of the (fall-through) sequence. Throws
 * `UnsupportedConstruct` on the first read of a `local` name that is not yet
 * definitely assigned.
 */
function walk(
  stmts: IRStmt[],
  assigned: Set<string>,
  locals: Set<string>
): Set<string> {
  const checkReads = (node: unknown, span: Var["span"]): void => {
    const reads = new Set<string>();
    collectReads(node, reads);
    for (const name of reads) {
      if (locals.has(name) && !assigned.has(name)) {
        throw new UnsupportedConstruct(
          `'${name}' may be used before it is assigned (assigned only on a ` +
            `path that need not run); declining C-JIT so the interpreter can ` +
            `reproduce MATLAB's undefined-variable semantics`,
          span
        );
      }
    }
  };

  for (const s of stmts) {
    switch (s.kind) {
      case "Assign":
        checkReads(s.expr, s.span);
        assigned.add(s.name);
        break;
      case "ExprStmt":
        checkReads(s.expr, s.span);
        break;
      case "If": {
        checkReads(s.cond, s.span);
        const thenA = walk(s.thenBody, new Set(assigned), locals);
        const elseA = walk(s.elseBody, new Set(assigned), locals);
        // Only names assigned on BOTH arms are definite afterwards. (An
        // arm that exits via return/break makes this conservative — the
        // intersection drops names the surviving arm did assign — which
        // can only over-decline, never mis-accept.)
        const merged = setIntersect(thenA, elseA);
        for (const x of merged) assigned.add(x);
        break;
      }
      case "While": {
        checkReads(s.cond, s.span);
        // The body may run zero times, so its assignments are not definite
        // afterwards; walk it (with a copy) only to check reads inside.
        walk(s.body, new Set(assigned), locals);
        break;
      }
      case "For": {
        checkReads(s.start, s.span);
        checkReads(s.end, s.span);
        const bodyAssigned = new Set(assigned);
        bodyAssigned.add(s.varName); // loop var is bound inside the body
        const afterBody = walk(s.body, bodyAssigned, locals);
        // If the loop provably runs at least once, the body's fall-through
        // assignments (and the loop variable) are definite afterwards.
        // Otherwise the loop may run zero times, so they are not.
        if (forProvablyRunsOnce(s)) {
          for (const x of afterBody) assigned.add(x);
        }
        break;
      }
      case "MultiAssignCall":
        checkReads(s.args, s.span);
        for (const o of s.outputs) if (o.binding) assigned.add(o.binding.name);
        break;
      case "IndexStore":
        checkReads(s.indices, s.span);
        checkReads(s.rhs, s.span);
        assigned.add(s.base.name); // store can create/grow the base
        break;
      case "IndexSliceStore":
        checkReads(s.index, s.span);
        checkReads(s.rhs, s.span);
        assigned.add(s.base.name);
        break;
      case "CellIndexStore":
        checkReads(s.indices, s.span);
        checkReads(s.rhs, s.span);
        assigned.add(s.base.name);
        break;
      case "MemberStore":
        checkReads(s.rhs, s.span);
        assigned.add(s.base.name);
        break;
      case "ReturnFromFunction":
      case "Break":
      case "Continue":
      case "TypeComment":
        break;
      default:
        break;
    }
  }
  return assigned;
}

/**
 * Throw `UnsupportedConstruct` if `fn` may read a local variable before it
 * is definitely assigned. Parameters are assigned on entry.
 *
 * `checkOutputs` additionally rejects a function whose declared output may
 * be returned before assignment (MATLAB's "Output argument not assigned").
 * Enable it only for REAL user functions: a synthetic loop/top-level wrapper
 * lists the loop variable (and other loop-body locals) as pseudo-outputs
 * that are legitimately unassigned when the loop runs zero times, so the
 * check would over-decline every loop.
 */
export function assertDefiniteAssignment(
  fn: IRFunc,
  opts: { checkOutputs: boolean } = { checkOutputs: false }
): void {
  const locals = new Set<string>();
  collectAssignedNames(fn.body, locals);
  // Parameters are always bound; never flag reads of them.
  for (const p of fn.params) locals.delete(p);
  const assigned = new Set<string>(fn.params);
  walk(fn.body, assigned, locals);
  if (!opts.checkOutputs) return;
  // An output that is not definitely assigned on the fall-through path would
  // be returned as C's predeclared default (0 / empty) instead of raising
  // "Output argument not assigned" like the interpreter. Decline so the
  // fallback handles it.
  for (const out of fn.outputs) {
    if (!assigned.has(out)) {
      throw new UnsupportedConstruct(
        `output '${out}' may be returned before it is assigned (assigned ` +
          `only on a path that need not run); declining C-JIT so the ` +
          `interpreter can reproduce MATLAB's undefined-variable semantics`,
        fn.span
      );
    }
  }
}

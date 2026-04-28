/**
 * Shared lowering pipeline.
 *
 * The dispatcher calls `tryLower` once per stmt-dispatch, before any
 * executor is asked to propose. The result — a `LoweredStmt` — is
 * passed to every executor's `propose()` as the first argument.
 *
 * Lowering produces an IR; it does NOT make codegen-feasibility
 * decisions. "Can this be JS-JIT'd?" lives in the JS-JIT executor's
 * propose. "Can this be a C kernel?" lives in the C kernel executor's
 * propose. Lowering's only "no" is structural: type-unknown inputs,
 * lowerFunction declined, etc. — failures that prevent the lowering
 * pipeline from producing an IR at all.
 *
 * For shapes that haven't been added yet, lowering returns the raw
 * stmt wrapped in `{ kind: "stmt", stmt }`. AST-driven executors
 * (e.g., the e2 family) filter on `kind === "stmt"` and continue to
 * inspect the AST as before.
 *
 * Lowerings are cached by (head stmt, classification cacheKey). The
 * cheap classify pass runs every dispatch; the expensive IR-lowering
 * pass runs at most once per cacheKey.
 */

import type { Stmt } from "../parser/types.js";
import type { DispatchContext } from "./context.js";
import {
  classifyTopLevel,
  lowerTopLevel,
  type TopLevelClassification,
  type TopLevelLowered,
} from "./jsJit/jitTopLevel.js";
import {
  classifyLoop,
  lowerLoop,
  type LoopClassification,
  type LoopLowered,
} from "./jsJit/jitLoop.js";
import { analyzeTopLevel } from "../jit/jitLoopAnalysis.js";
import {
  JIT_IO_BUILTINS,
  irHasBailRisk,
  irHasIO,
} from "../jit/jitBailSafety.js";

/** What `propose()` receives. The dispatcher always produces some
 *  variant — there's no `null` LoweredStmt. Executors filter on
 *  `kind`. */
export type LoweredStmt = TopLevelLoweredStmt | LoopLoweredStmt | RawStmt;

/** Top-level shape: the entire script body lowered to JS-JIT IR.
 *  Carries the lowered IR plus pre-computed feasibility flags so
 *  propose() can do cheap inspection without re-walking the IR. */
export interface TopLevelLoweredStmt {
  readonly kind: "top-level";
  readonly classification: TopLevelClassification;
  readonly lowered: TopLevelLowered;
  readonly flags: TopLevelFlags;
}

/** Pre-computed feasibility flags for top-level codegen executors. */
export interface TopLevelFlags {
  /** Source body contains a `return` statement. JIT cannot model
   *  early-return from the synthetic top-level fn. */
  readonly hasReturn: boolean;
  /** Source body contains an unsuppressed assign/multiassign. In
   *  display-mode, the JIT has no emit for auto-display, so it must
   *  bail when these are present. */
  readonly hasUnsuppressedAssign: boolean;
  /** Lowered IR contains an I/O builtin (disp, fprintf, ...). */
  readonly hasIO: boolean;
  /** Lowered IR contains a possibly-bailing operation. Combined with
   *  hasIO, signals a body that mustn't be retried after a partial
   *  run (already-emitted output would duplicate). */
  readonly hasBailRisk: boolean;
}

/** Loop shape: a For/While stmt lowered to JS-JIT IR. */
export interface LoopLoweredStmt {
  readonly kind: "loop";
  readonly classification: LoopClassification;
  readonly lowered: LoopLowered;
  readonly flags: LoopFlags;
}

/** Pre-computed feasibility flags for loop codegen executors. */
export interface LoopFlags {
  /** Loop body contains `return` — JIT can't model early-return. */
  readonly hasReturn: boolean;
  /** Lowered IR contains an I/O builtin. */
  readonly hasIO: boolean;
  /** Lowered IR contains a possibly-bailing operation. */
  readonly hasBailRisk: boolean;
}

/** Fallback wrapper for stmts that have no specialized lowering shape
 *  yet. AST-driven executors filter on this kind and inspect `stmt`
 *  directly. */
export interface RawStmt {
  readonly kind: "stmt";
  readonly stmt: Stmt;
}

const BAILED = Symbol("LOWERING_BAILED");
type Bailed = typeof BAILED;

/** Per-stmt lowering cache. Entries are scoped to the head stmt
 *  (WeakMap key) so they're reclaimed when the AST is dropped. The
 *  inner Map is keyed by the classification's cacheKey. */
export class LoweringCache {
  private readonly slots = new WeakMap<
    Stmt,
    Map<string, LoweredStmt | Bailed>
  >();

  get(stmt: Stmt, cacheKey: string): LoweredStmt | Bailed | undefined {
    return this.slots.get(stmt)?.get(cacheKey);
  }

  set(stmt: Stmt, cacheKey: string, value: LoweredStmt): void {
    let perStmt = this.slots.get(stmt);
    if (!perStmt) {
      perStmt = new Map();
      this.slots.set(stmt, perStmt);
    }
    perStmt.set(cacheKey, value);
  }

  markBailed(stmt: Stmt, cacheKey: string): void {
    let perStmt = this.slots.get(stmt);
    if (!perStmt) {
      perStmt = new Map();
      this.slots.set(stmt, perStmt);
    }
    perStmt.set(cacheKey, BAILED);
  }

  isBailed(value: unknown): value is Bailed {
    return value === BAILED;
  }
}

/**
 * Try to lower the stmt at `siblings[i]` based on current runtime
 * info. Always returns a LoweredStmt — at minimum, the raw stmt
 * wrapped in `{ kind: "stmt", stmt }`.
 *
 * Uses `cache` to memoize specialized shapes by (head stmt,
 * classification cacheKey). The cheap classify runs every dispatch;
 * the expensive IR-lowering runs at most once per cacheKey.
 */
export function tryLower(
  siblings: readonly Stmt[],
  i: number,
  ctx: DispatchContext,
  cache: LoweringCache
): LoweredStmt {
  const head = siblings[i];

  // Top-level shape: only at the script's root, on the first stmt.
  if (ctx.scope === "top-level" && i === 0) {
    const classification = classifyTopLevel(ctx.interp, siblings as Stmt[]);
    if (classification) {
      const hit = cache.get(head, classification.cacheKey);
      if (hit !== undefined) {
        if (cache.isBailed(hit)) {
          // Lowering previously failed — fall through to raw stmt.
          return { kind: "stmt", stmt: head };
        }
        return hit;
      }
      const lowered = lowerTopLevel(ctx.interp, classification);
      if (!lowered) {
        cache.markBailed(head, classification.cacheKey);
        return { kind: "stmt", stmt: head };
      }
      const flags = computeTopLevelFlags(classification, lowered);
      const entry: TopLevelLoweredStmt = {
        kind: "top-level",
        classification,
        lowered,
        flags,
      };
      cache.set(head, classification.cacheKey, entry);
      return entry;
    }
  }

  // Loop shape: For/While stmts.
  if (head.type === "For" || head.type === "While") {
    const loopStmt = head as Stmt & { type: "For" | "While" };
    const classification = classifyLoop(ctx.interp, loopStmt, siblings, i);
    if (classification) {
      const hit = cache.get(head, classification.cacheKey);
      if (hit !== undefined) {
        if (cache.isBailed(hit)) {
          return { kind: "stmt", stmt: head };
        }
        return hit;
      }
      const lowered = lowerLoop(ctx.interp, classification);
      if (!lowered) {
        cache.markBailed(head, classification.cacheKey);
        return { kind: "stmt", stmt: head };
      }
      const flags = computeLoopFlags(classification, lowered);
      const entry: LoopLoweredStmt = {
        kind: "loop",
        classification,
        lowered,
        flags,
      };
      cache.set(head, classification.cacheKey, entry);
      return entry;
    }
  }

  return { kind: "stmt", stmt: head };
}

function computeTopLevelFlags(
  classification: TopLevelClassification,
  lowered: TopLevelLowered
): TopLevelFlags {
  // Re-run the AST analysis to get hasReturn (cheap; same walk
  // analyzeTopLevel does for inputs/outputs). Done here rather than
  // returning it from classifyTopLevel because feasibility flags are
  // a lowering-output concern, not a classification-output concern.
  const analysis = analyzeTopLevel(classification.stmts as Stmt[]);
  const hasReturn = analysis.hasReturn;

  let hasUnsuppressedAssign = false;
  for (const s of classification.stmts) {
    if (
      (s.type === "Assign" ||
        s.type === "AssignLValue" ||
        s.type === "MultiAssign") &&
      !s.suppressed
    ) {
      hasUnsuppressedAssign = true;
      break;
    }
    if (s.type === "ExprStmt" && !s.suppressed) {
      const e = s.expr;
      const isVoidCall =
        e.type === "FuncCall" &&
        (JIT_IO_BUILTINS.has(e.name) || e.name === "tic");
      if (!isVoidCall) {
        hasUnsuppressedAssign = true;
        break;
      }
    }
  }

  const result = lowered.result;
  const hasIO = irHasIO(result.body, result.generatedIRBodies);
  const hasBailRisk = irHasBailRisk(result.body, result.generatedIRBodies);

  return { hasReturn, hasUnsuppressedAssign, hasIO, hasBailRisk };
}

function computeLoopFlags(
  classification: LoopClassification,
  lowered: LoopLowered
): LoopFlags {
  const result = lowered.result;
  const hasIO = irHasIO(result.body, result.generatedIRBodies);
  const hasBailRisk = irHasBailRisk(result.body, result.generatedIRBodies);
  return { hasReturn: classification.hasReturn, hasIO, hasBailRisk };
}

/**
 * For/while loop body — phased pipeline.
 *
 *   1. `classifyLoop`   — analyze loop, gather typed env inputs,
 *      live-out filter on outputs (using post-siblings), type
 *      widening, cacheKey synthesis.
 *
 *   2. `lowerLoop`      — wrap as a synthetic `FunctionDef` and
 *      lower via `lowerSyntheticFn`.
 *
 *   3. `generateLoopJS` — `generateSyntheticFnJS` + diagnostics.
 *
 *   4. `runLoopCompiled` — `runSyntheticFnAgainstEnv`.
 *
 * Shared with top-level via `shared.ts`. What's loop-specific is
 * the analysis (For/While discrimination), the live-out filter
 * against post-siblings, the cacheKey location (per loop AST
 * span), and the `$loop_for` / `$loop_while` synthetic name.
 */
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt } from "../../parser/types.js";
import type { JitType } from "../../jitTypes.js";
import { jitTypeKey } from "../../jitTypes.js";
import type { LoweringResult } from "./lower/jitLower.js";
import {
  analyzeForLoop,
  analyzeWhileLoop,
  collectReadsFromSiblings,
} from "./lower/blockAnalysis.js";
import {
  gatherTypedEnvInputs,
  generateSyntheticFnJS,
  lowerSyntheticFn,
  runSyntheticFnAgainstEnv,
  widenAgainst,
  type SyntheticFnRunResult,
} from "./shared.js";

export type LoopKind = "for" | "while";

export interface LoopClassification {
  readonly stmt: Stmt & { type: "For" | "While" };
  readonly kind: LoopKind;
  readonly inputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly outputs: readonly string[];
  readonly hasReturn: boolean;
  readonly cacheKey: string;
}

export interface LoopLowered {
  readonly classification: LoopClassification;
  readonly result: LoweringResult;
}

export interface LoopCompiled {
  readonly fn: (...args: unknown[]) => unknown;
  readonly source: string;
}

export type LoopRunResult = SyntheticFnRunResult;

/** `postSiblings` / `postIdx` describe the stmt list this loop sits
 *  in; used to compute the live-out filter on outputs. */
export function classifyLoop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" | "While" },
  postSiblings: readonly Stmt[],
  postIdx: number,
  prevInputTypes: readonly JitType[] | undefined
): LoopClassification | null {
  const kind: LoopKind = stmt.type === "For" ? "for" : "while";
  const analysis =
    stmt.type === "For" ? analyzeForLoop(stmt) : analyzeWhileLoop(stmt);

  // Candidate inputs: referenced names ∪ assigned-in-body names.
  // Promoting outputs-that-exist-in-env to inputs preserves their
  // pre-loop value for write-only-in-body locals.
  const seen = new Set<string>();
  const inputCandidates: string[] = [];
  for (const name of [...analysis.inputs, ...analysis.outputs]) {
    if (seen.has(name)) continue;
    seen.add(name);
    inputCandidates.push(name);
  }

  const gathered = gatherTypedEnvInputs(interp, inputCandidates);
  if (!gathered) return null;
  const { inputs, inputTypes } = gathered;

  // Outputs: dedupe, then filter to live-out set. Loop-internal
  // temporaries that aren't read post-loop end up unwritten so V8
  // can keep them in registers.
  let outputs = [...new Set(analysis.outputs)];
  const liveOut = new Set<string>(inputs);
  if (stmt.type === "For") liveOut.add(stmt.varName);
  if (postSiblings.length > postIdx + 1) {
    collectReadsFromSiblings(postSiblings as Stmt[], postIdx + 1, liveOut);
  }
  outputs = outputs.filter(o => liveOut.has(o));

  // Progressive type widening: matters when called from an
  // interpreted outer loop where input types can shift. The
  // dispatcher records the unified result in the lowering cache.
  widenAgainst(inputTypes, prevInputTypes);

  const loc = stmt.span
    ? `${stmt.span.file}:${stmt.span.start}`
    : `loop:${kind}`;
  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const cacheKey = `${loc}|${typeKey}`;

  return {
    stmt,
    kind,
    inputs,
    inputTypes,
    outputs,
    hasReturn: analysis.hasReturn,
    cacheKey,
  };
}

export function lowerLoop(
  interp: Interpreter,
  classification: LoopClassification
): LoopLowered | null {
  const { stmt, kind, inputs, inputTypes, outputs } = classification;
  const result = lowerSyntheticFn(
    interp,
    `$loop_${kind}`,
    inputs,
    inputTypes,
    outputs,
    [stmt]
  );
  if (!result) return null;
  return { classification, result };
}

export function generateLoopJS(
  interp: Interpreter,
  lowered: LoopLowered
): LoopCompiled | null {
  const { classification, result } = lowered;
  const { kind, inputs, inputTypes, outputs } = classification;

  const generated = generateSyntheticFnJS(
    interp,
    result,
    inputs,
    outputs.length
  );
  if (!generated) return null;

  const paramComments = inputs
    .map((p, i) => `${p}: ${jitTypeKey(inputTypes[i])}`)
    .join(", ");
  const name = `$loop_${kind}`;
  const source = `// JIT loop (${kind}): (${paramComments})\nfunction ${name}(${inputs.join(", ")}) {\n${generated.jsBody}\n}`;

  const line = interp.rt.$line ?? 0;
  interp.onJitCompile?.(`loop:${kind}@${line}(${paramComments})`, source);

  return { fn: generated.fn, source };
}

export function runLoopCompiled(
  interp: Interpreter,
  compiled: LoopCompiled,
  classification: LoopClassification
): LoopRunResult {
  return runSyntheticFnAgainstEnv(
    interp,
    compiled.fn,
    classification.inputs,
    classification.outputs
  );
}

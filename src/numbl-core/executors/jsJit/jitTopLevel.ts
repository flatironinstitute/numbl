/**
 * Top-level script body — phased pipeline.
 *
 *   1. `classifyTopLevel`  — analyze script body, gather typed env
 *      inputs, type widening, cacheKey synthesis.
 *
 *   2. `lowerTopLevel`     — wrap as a synthetic `FunctionDef` and
 *      lower via `lowerSyntheticFn`.
 *
 *   3. `generateTopLevelJS` — `generateSyntheticFnJS` + diagnostics.
 *
 *   4. `runTopLevelCompiled` — `runSyntheticFnAgainstEnv`.
 *
 * The shared mechanics live in `shared.ts`; this module spells out
 * what's actually top-level-specific (whole-script body, no
 * post-loop liveness filter, `$top` synthetic name).
 */
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt } from "../../parser/types.js";
import type { JitType } from "../../jitTypes.js";
import { jitTypeKey } from "../../jitTypes.js";
import type { LoweringResult } from "./lower/jitLower.js";
import { analyzeTopLevel } from "./lower/blockAnalysis.js";
import {
  buildJitSourceComment,
  gatherTypedEnvInputs,
  generateSyntheticFnJS,
  lowerSyntheticFn,
  runSyntheticFnAgainstEnv,
  widenAgainst,
  type SyntheticFnRunResult,
} from "./shared.js";

const SYNTHETIC_NAME = "$top";

export interface TopLevelClassification {
  readonly stmts: readonly Stmt[];
  readonly inputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly outputs: readonly string[];
  readonly currentFile: string;
  readonly hasReturn: boolean;
  readonly cacheKey: string;
}

export interface TopLevelLowered {
  readonly classification: TopLevelClassification;
  readonly result: LoweringResult;
}

export interface TopLevelCompiled {
  readonly fn: (...args: unknown[]) => unknown;
  readonly source: string;
}

export type TopLevelRunResult = SyntheticFnRunResult;

export function classifyTopLevel(
  interp: Interpreter,
  stmts: readonly Stmt[],
  prevInputTypes: readonly JitType[] | undefined
): TopLevelClassification | null {
  if (stmts.length === 0) return null;

  const analysis = analyzeTopLevel(stmts as Stmt[]);

  // Candidate input order: referenced names first, then assigned
  // names that also exist in env (pre-script values to preserve).
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

  // Every assigned name is live-out — top-level has no post-loop
  // liveness analog, so no liveness filter here.
  const outputs = [...new Set(analysis.outputs)];

  // Progressive type widening against the previously-recorded
  // signature for this stmt position. The dispatcher records the
  // result in the lowering cache after this returns.
  widenAgainst(inputTypes, prevInputTypes);

  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const cacheKey = `$top:${interp.currentFile}|${typeKey}`;

  return {
    stmts,
    inputs,
    inputTypes,
    outputs,
    currentFile: interp.currentFile,
    hasReturn: analysis.hasReturn,
    cacheKey,
  };
}

export function lowerTopLevel(
  interp: Interpreter,
  classification: TopLevelClassification
): TopLevelLowered | null {
  const { stmts, inputs, inputTypes, outputs } = classification;
  const result = lowerSyntheticFn(
    interp,
    SYNTHETIC_NAME,
    inputs,
    inputTypes,
    outputs,
    [...stmts]
  );
  if (!result) return null;
  return { classification, result };
}

export function generateTopLevelJS(
  interp: Interpreter,
  lowered: TopLevelLowered
): TopLevelCompiled | null {
  const { classification, result } = lowered;
  const { inputs, inputTypes, outputs, currentFile } = classification;

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
  const stmts = classification.stmts;
  const sourceComment =
    stmts.length > 0 && stmts[0].span
      ? buildJitSourceComment(
          interp,
          stmts[0].span.file,
          stmts[0].span.start,
          stmts[stmts.length - 1].span.end
        ) + "\n"
      : "";
  const source = `${sourceComment}// JIT top-level: (${paramComments})\nfunction ${SYNTHETIC_NAME}(${inputs.join(", ")}) {\n${generated.jsBody}\n}`;

  interp.onJitCompile?.(`top-level@${currentFile}(${paramComments})`, source);

  return { fn: generated.fn, source };
}

export function runTopLevelCompiled(
  interp: Interpreter,
  compiled: TopLevelCompiled,
  classification: TopLevelClassification
): TopLevelRunResult {
  return runSyntheticFnAgainstEnv(
    interp,
    compiled.fn,
    classification.inputs,
    classification.outputs
  );
}

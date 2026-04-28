/**
 * Top-level script body — phased pipeline.
 *
 * Splits the work into four phases that map onto the new architecture
 * where the dispatcher does the lowering before any executor proposes:
 *
 *   1. `classifyTopLevel`  — cheap classification + cacheKey
 *      synthesis. Type-infers env inputs, gathers outputs, applies
 *      progressive type widening, builds the cache key. Returns null
 *      only on structural blockers (empty body, type-unknown inputs)
 *      that prevent the lowering pipeline from running. Codegen-
 *      feasibility checks (display mode, hasReturn, IO+bail-risk)
 *      live in the executor's propose, not here.
 *
 *   2. `lowerTopLevel`     — IR lowering. Wraps stmts as a synthetic
 *      `FunctionDef`, calls `lowerFunction`. Returns null when
 *      lowering itself declines.
 *
 *   3. `generateTopLevelJS` — JS codegen + `new Function`. Per-
 *      executor (the JS-JIT executor calls it; a hypothetical C
 *      executor for whole-script would have its own pass against
 *      the same IR).
 *
 *   4. `runTopLevelCompiled` — gather inputs from env, invoke
 *      compiled fn, write back outputs. Translates
 *      JitBailToInterpreter into a transient bail (cache preserved)
 *      and JitFuncHandleBailError into a permanent bail (cache
 *      invalidated).
 */
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt } from "../../parser/types.js";
import type { FunctionDef } from "../../interpreter/types.js";
import type { JitType } from "../../jit/jitTypes.js";
import { jitTypeKey, unifyJitTypes } from "../../jit/jitTypes.js";
import { lowerFunction, type LoweringResult } from "../../jit/jitLower.js";
import { generateJS } from "./js/jitCodegen.js";
import {
  jitHelpers,
  JitFuncHandleBailError,
  JitBailToInterpreter,
} from "./js/jitHelpers.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import { analyzeTopLevel } from "../../jit/jitLoopAnalysis.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../../runtime/types.js";

const KNOWN_CONSTANTS = new Set([
  "pi",
  "inf",
  "Inf",
  "nan",
  "NaN",
  "eps",
  "true",
  "false",
  "end",
  "i",
  "j",
]);

/** Cheap pre-lowering data: the inputs/outputs/types that drive the
 *  cacheKey, the head stmt for cache scoping, and the original stmt
 *  list. Produced by `classifyTopLevel`. */
export interface TopLevelClassification {
  readonly stmts: readonly Stmt[];
  readonly inputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly outputs: readonly string[];
  readonly currentFile: string;
  /** Stable string key derived from (currentFile, input-type signature).
   *  The dispatcher's lowering cache and any per-executor compile cache
   *  both key on this. */
  readonly cacheKey: string;
}

/** The lowered IR plus the classification it came from. Produced by
 *  `lowerTopLevel`; consumed by per-executor codegen passes
 *  (`generateTopLevelJS` for JS, …). */
export interface TopLevelLowered {
  readonly classification: TopLevelClassification;
  readonly result: LoweringResult;
}

/** A compiled JS artifact ready to run. Produced by
 *  `generateTopLevelJS`; consumed by `runTopLevelCompiled`. */
export interface TopLevelCompiled {
  readonly fn: (...args: unknown[]) => unknown;
  readonly source: string;
}

/**
 * Phase 1 — cheap classification. Returns null when the body is
 * unjittable on grounds we can decide without lowering (display mode +
 * unsuppressed assigns, hasReturn, unknown-typed inputs, empty body).
 *
 * Mutates `interp.loopLastInputTypes` for progressive type widening:
 * subsequent calls unify against the most recent type signature so
 * repeated runs of the same script with shifting types converge to a
 * single specialization.
 */
export function classifyTopLevel(
  interp: Interpreter,
  stmts: readonly Stmt[]
): TopLevelClassification | null {
  if (stmts.length === 0) return null;

  // Codegen-feasibility checks (display mode + unsuppressed stmts,
  // hasReturn, IO+bail-risk) live in the executor's propose. This
  // pass only blocks on structural lowering pre-conditions: empty
  // body, type-unknown inputs.

  const analysis = analyzeTopLevel(stmts as Stmt[]);

  // Candidate input order: referenced names first, then assigned names
  // that also exist in env (pre-script values we must preserve).
  const inputCandidates: string[] = [];
  const seen = new Set<string>();
  for (const name of analysis.inputs) {
    if (seen.has(name)) continue;
    seen.add(name);
    inputCandidates.push(name);
  }
  for (const name of analysis.outputs) {
    if (seen.has(name)) continue;
    seen.add(name);
    inputCandidates.push(name);
  }

  const inputs: string[] = [];
  const inputTypes: JitType[] = [];

  for (const name of inputCandidates) {
    if (KNOWN_CONSTANTS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) continue; // not a variable in scope (likely a fn name)
    let t = inferJitType(val);
    if (t.kind === "unknown") return null;
    if (t.kind === "number" && t.exact !== undefined) {
      const pruned: JitType = { kind: "number" };
      if (t.sign !== undefined) pruned.sign = t.sign;
      if (t.isInteger) pruned.isInteger = true;
      t = pruned;
    }
    inputs.push(name);
    inputTypes.push(t);
  }

  // Every assigned name is live-out — no liveness filter here.
  const outputs = [...new Set(analysis.outputs)];

  // Progressive type widening: rarely matters in practice (top-level
  // body normally runs once) but keeps semantics consistent if the
  // same interp ever re-runs the same script.
  const loc = `$top:${interp.currentFile}`;
  const prevTypes = interp.loopLastInputTypes.get(loc);
  if (prevTypes && prevTypes.length === inputTypes.length) {
    for (let i = 0; i < inputTypes.length; i++) {
      inputTypes[i] = unifyJitTypes(inputTypes[i], prevTypes[i]);
    }
  }
  interp.loopLastInputTypes.set(loc, inputTypes.slice());

  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const cacheKey = `${loc}|${typeKey}`;

  return {
    stmts,
    inputs,
    inputTypes,
    outputs,
    currentFile: interp.currentFile,
    cacheKey,
  };
}

/**
 * Phase 2 — IR lowering. Produces the shared lowered IR consumed by
 * all whole-script codegen passes. Returns null when `lowerFunction`
 * itself declines (the body has constructs the JS-JIT IR doesn't
 * model, e.g. unsupported control flow). Codegen-feasibility checks
 * (IO+bail-risk, etc.) are the executor's responsibility.
 */
export function lowerTopLevel(
  interp: Interpreter,
  classification: TopLevelClassification
): TopLevelLowered | null {
  const { stmts, inputs, inputTypes, outputs } = classification;

  const syntheticFn: FunctionDef = {
    name: `$top`,
    params: [...inputs],
    outputs: [...outputs],
    body: [...stmts],
  };

  const result = lowerFunction(
    syntheticFn,
    [...inputTypes],
    outputs.length,
    interp
  );
  if (!result) return null;

  return { classification, result };
}

/**
 * Phase 3 — JS codegen. Generates a JS function from the lowered IR
 * and wires it through the runtime helpers. Returns null only if
 * `new Function` itself throws (malformed JS — should never happen for
 * IR that lowering accepted; defensive).
 */
export function generateTopLevelJS(
  interp: Interpreter,
  lowered: TopLevelLowered
): TopLevelCompiled | null {
  const { classification, result } = lowered;
  const { inputs, inputTypes, outputs, currentFile } = classification;

  const mainBody = generateJS(
    result.body,
    [...inputs],
    result.outputNames,
    outputs.length,
    result.localVars,
    currentFile,
    interp.experimental,
    interp.par
  );

  const parts: string[] = [];
  for (const [, code] of result.generatedFns) {
    parts.push(code.replace(/^/gm, "  "));
  }
  parts.push(mainBody);
  const jsBody = parts.join("\n");

  let compiledFn: (...args: unknown[]) => unknown;
  const rt = interp.rt;
  try {
    const factory = new Function("$h", "$rt", ...inputs, jsBody);
    const helpers = rt.jitHelpers ?? jitHelpers;
    compiledFn = (...callArgs: unknown[]) => factory(helpers, rt, ...callArgs);
  } catch {
    return null;
  }

  const paramComments = inputs
    .map((p, i) => `${p}: ${jitTypeKey(inputTypes[i])}`)
    .join(", ");
  const source = `// JIT top-level: (${paramComments})\nfunction $top(${inputs.join(", ")}) {\n${jsBody}\n}`;

  const description = `top-level@${currentFile}(${inputs.map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`).join(", ")})`;
  interp.onJitCompile?.(description, source);

  return { fn: compiledFn, source };
}

/** Outcome of `runTopLevelCompiled`. `transient` distinguishes the
 *  recoverable JitBailToInterpreter case (cache preserved) from the
 *  hard JitFuncHandleBailError case (cache invalidated). */
export type TopLevelRunResult =
  | { ok: true }
  | { ok: false; transient: boolean };

/**
 * Phase 4 — execute the compiled fn against current env and write
 * back outputs. Re-fetches input values from env each call (values
 * may change between dispatches even when types don't).
 */
export function runTopLevelCompiled(
  interp: Interpreter,
  compiled: TopLevelCompiled,
  classification: TopLevelClassification
): TopLevelRunResult {
  const { inputs, outputs } = classification;

  const inputValues: unknown[] = [];
  for (const name of inputs) {
    inputValues.push(interp.env.get(name));
  }

  let result: unknown;
  try {
    result = compiled.fn(...inputValues);
  } catch (e) {
    if (e instanceof JitFuncHandleBailError) {
      console.warn(`Warning: ${e.message}`);
      return { ok: false, transient: false };
    }
    if (e instanceof JitBailToInterpreter) {
      return { ok: false, transient: true };
    }
    throw e;
  }

  if (outputs.length === 0) {
    // Nothing to write back; compiled body ran for side effects only.
  } else if (outputs.length === 1) {
    interp.env.set(outputs[0], ensureRuntimeValue(result) as RuntimeValue);
  } else {
    const arr = result as unknown[];
    for (let i = 0; i < outputs.length; i++) {
      interp.env.set(outputs[i], ensureRuntimeValue(arr[i]) as RuntimeValue);
    }
  }

  return { ok: true };
}

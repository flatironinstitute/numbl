/**
 * For/while loop body — phased pipeline.
 *
 * Mirrors `jitTopLevel.ts`'s structure: classify (cheap) → lower (IR)
 * → generate-JS (per-executor codegen) → run. The dispatcher's
 * `tryLower` runs the first two phases; the JS-JIT loop executor
 * consumes the lowered IR for the latter two.
 *
 *   1. `classifyLoop`  — cheap classification + cacheKey synthesis.
 *      Inputs analysis (rawInputs ∪ rawOutputs filtered against env),
 *      output live-out filter against post-siblings, type widening
 *      across runs, cacheKey from (loop location, type signature).
 *
 *   2. `lowerLoop`     — IR lowering. Wraps the loop stmt as a
 *      synthetic `FunctionDef`, calls `lowerFunction`. Returns null
 *      when lowerFunction itself declines.
 *
 *   3. `generateLoopJS` — JS codegen + `new Function`.
 *
 *   4. `runLoopCompiled` — gather inputs from env, invoke compiled
 *      fn, write back outputs. Translates JitBailToInterpreter into
 *      a transient bail (cache preserved) and JitFuncHandleBailError
 *      into a permanent bail (cache invalidated).
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
import {
  analyzeForLoop,
  analyzeWhileLoop,
  collectReadsFromSiblings,
} from "../../jit/jitLoopAnalysis.js";
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

export type LoopKind = "for" | "while";

/** Cheap pre-lowering data. Produced by `classifyLoop`. */
export interface LoopClassification {
  readonly stmt: Stmt & { type: "For" | "While" };
  readonly kind: LoopKind;
  readonly inputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly outputs: readonly string[];
  /** True when the loop body contains a `return` — JIT can't model
   *  early-return out of the synthetic loop fn, so the executor must
   *  decline. */
  readonly hasReturn: boolean;
  /** Stable string key derived from (loop location, input-type
   *  signature). Used by the dispatcher's lowering cache and by the
   *  executor's compile cache. */
  readonly cacheKey: string;
}

/** Lowered IR plus the classification it came from. */
export interface LoopLowered {
  readonly classification: LoopClassification;
  readonly result: LoweringResult;
}

/** A compiled JS artifact ready to run. */
export interface LoopCompiled {
  readonly fn: (...args: unknown[]) => unknown;
  readonly source: string;
}

/**
 * Phase 1 — cheap classification. Returns null when the loop is
 * unjittable on grounds we can decide without lowering (type-unknown
 * inputs). Codegen-feasibility checks (hasReturn, IO+bail-risk) are
 * exposed as flags / via the lowered IR for the executor's propose
 * to act on.
 *
 * Mutates `interp.loopLastInputTypes` for progressive type widening.
 *
 * `postSiblings` / `postIdx` describe the stmt list this loop sits
 * in (passed from the dispatcher's `ctx.siblings` / `ctx.headIndex`),
 * used to compute the live-out filter on outputs.
 */
export function classifyLoop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" | "While" },
  postSiblings: readonly Stmt[],
  postIdx: number
): LoopClassification | null {
  const kind: LoopKind = stmt.type === "For" ? "for" : "while";
  const analysis =
    stmt.type === "For" ? analyzeForLoop(stmt) : analyzeWhileLoop(stmt);

  // Filter inputs: only variables that exist in the current
  // environment and are not known constants/builtins. Promote any
  // rawOutputs entry that exists in env to an input as well —
  // preserves pre-loop values of conditionally-assigned locals
  // (write-only-in-body case).
  const inputCandidates: string[] = [];
  const seenCandidates = new Set<string>();
  for (const name of analysis.inputs) {
    if (seenCandidates.has(name)) continue;
    seenCandidates.add(name);
    inputCandidates.push(name);
  }
  for (const name of analysis.outputs) {
    if (seenCandidates.has(name)) continue;
    seenCandidates.add(name);
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

  // Outputs: dedupe, then filter to live-out set. Loop-internal
  // temporaries that aren't read post-loop end up unwritten so V8
  // can keep them in registers.
  const outputSet = new Set(analysis.outputs);
  let outputs = [...outputSet];

  const inputSet = new Set(inputs);
  const liveOut = new Set<string>(inputSet);
  if (stmt.type === "For") liveOut.add(stmt.varName);
  if (postSiblings.length > postIdx + 1) {
    collectReadsFromSiblings(postSiblings as Stmt[], postIdx + 1, liveOut);
  }
  outputs = outputs.filter(o => liveOut.has(o));

  // Progressive type widening: rare for inline loops but matters
  // when called from an interpreted outer loop where input types
  // can shift.
  const loc = stmt.span
    ? `${stmt.span.file}:${stmt.span.start}`
    : `loop:${kind}`;
  const prevLoopTypes = interp.loopLastInputTypes.get(loc);
  if (prevLoopTypes && prevLoopTypes.length === inputTypes.length) {
    for (let i = 0; i < inputTypes.length; i++) {
      inputTypes[i] = unifyJitTypes(inputTypes[i], prevLoopTypes[i]);
    }
  }
  interp.loopLastInputTypes.set(loc, inputTypes.slice());

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

/**
 * Phase 2 — IR lowering. Wraps the loop as a synthetic FunctionDef
 * and calls `lowerFunction`. Returns null when lowerFunction itself
 * declines.
 */
export function lowerLoop(
  interp: Interpreter,
  classification: LoopClassification
): LoopLowered | null {
  const { stmt, kind, inputs, inputTypes, outputs } = classification;

  const syntheticFn: FunctionDef = {
    name: `$loop_${kind}`,
    params: [...inputs],
    outputs: [...outputs],
    body: [stmt],
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
 * Phase 3 — JS codegen. Returns null only if `new Function` itself
 * throws (defensive — should not happen for IR that lowering accepted).
 */
export function generateLoopJS(
  interp: Interpreter,
  lowered: LoopLowered
): LoopCompiled | null {
  const { classification, result } = lowered;
  const { kind, inputs, inputTypes, outputs } = classification;

  const currentFile = interp.currentFile;
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
  const source = `// JIT loop (${kind}): (${paramComments})\nfunction $loop_${kind}(${inputs.join(", ")}) {\n${jsBody}\n}`;

  const line = interp.rt.$line ?? 0;
  const description = `loop:${kind}@${line}(${inputs.map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`).join(", ")})`;
  interp.onJitCompile?.(description, source);

  return { fn: compiledFn, source };
}

/** Outcome of `runLoopCompiled`. `transient` distinguishes the
 *  recoverable JitBailToInterpreter case from the hard
 *  JitFuncHandleBailError case. */
export type LoopRunResult = { ok: true } | { ok: false; transient: boolean };

/**
 * Phase 4 — execute. Re-fetches input values from env each call.
 */
export function runLoopCompiled(
  interp: Interpreter,
  compiled: LoopCompiled,
  classification: LoopClassification
): LoopRunResult {
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
    // Nothing to write back.
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

/**
 * JIT compilation for inline for/while loops.
 *
 * When the interpreter encounters a for or while loop, these functions
 * attempt to JIT-compile the loop as a synthetic function: inputs are
 * variables from the enclosing scope, outputs are variables assigned
 * inside the loop body. On success the compiled code runs and output
 * values are written back to the interpreter environment.
 */

import type { Interpreter } from "../interpreter.js";
import type { Stmt } from "../../parser/types.js";
import type { FunctionDef } from "../types.js";
import type { JitType } from "./jitTypes.js";
import { jitTypeKey, unifyJitTypes } from "./jitTypes.js";
import { lowerFunction } from "./jitLower.js";
import { generateJS } from "./jitCodegen.js";
import {
  jitHelpers,
  JitFuncHandleBailError,
  JitBailToInterpreter,
} from "./jitHelpers.js";
import { getCJitBackend } from "./cJitBackend.js";
import { inferJitType } from "../builtins/types.js";
import {
  analyzeForLoop,
  analyzeWhileLoop,
  collectReadsFromSiblings,
} from "./jitLoopAnalysis.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../../runtime/types.js";

// ── Known constants that should not be treated as input variables ────────

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

// ── Public entry points ─────────────────────────────────────────────────

/**
 * Attempt to JIT-compile and execute a for-loop statement.
 * Returns true if JIT succeeded, false to fall back to interpretation.
 */
export function tryJitFor(
  interp: Interpreter,
  stmt: Stmt & { type: "For" }
): boolean {
  const analysis = analyzeForLoop(stmt);
  if (analysis.hasReturn) return false;
  return tryJitLoop(interp, stmt, analysis.inputs, analysis.outputs, "for");
}

/**
 * Attempt to JIT-compile and execute a while-loop statement.
 * Returns true if JIT succeeded, false to fall back to interpretation.
 */
export function tryJitWhile(
  interp: Interpreter,
  stmt: Stmt & { type: "While" }
): boolean {
  const analysis = analyzeWhileLoop(stmt);
  if (analysis.hasReturn) return false;
  return tryJitLoop(interp, stmt, analysis.inputs, analysis.outputs, "while");
}

// ── Core implementation ─────────────────────────────────────────────────

function tryJitLoop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" | "While" },
  rawInputs: string[],
  rawOutputs: string[],
  kind: "for" | "while"
): boolean {
  // Filter inputs: only variables that exist in the current environment
  // and are not known constants/builtins.
  //
  // We also promote any `rawOutputs` entry that exists in the outer env
  // to an input. This preserves the pre-loop value of variables that the
  // body *might* assign but doesn't always — most importantly, write-
  // only locals (`clear_i_exists = false; for ... clear_i_exists = true;
  // end`) whose only in-body use is an assignment. Without this,
  // the synthetic JIT function can't see the pre-loop value and,
  // when the range is empty, the unassigned local returns as undefined
  // and clobbers the outer env. Promoting output-in-env names to inputs
  // also unifies with the in-body assignment type at the loop join via
  // the usual merge path.
  const inputCandidates: string[] = [];
  const seenCandidates = new Set<string>();
  for (const name of rawInputs) {
    if (seenCandidates.has(name)) continue;
    seenCandidates.add(name);
    inputCandidates.push(name);
  }
  for (const name of rawOutputs) {
    if (seenCandidates.has(name)) continue;
    seenCandidates.add(name);
    inputCandidates.push(name);
  }

  const inputs: string[] = [];
  const inputValues: unknown[] = [];
  const inputTypes: JitType[] = [];

  for (const name of inputCandidates) {
    if (KNOWN_CONSTANTS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) continue; // not a variable in scope — likely a function name
    let t = inferJitType(val);
    if (t.kind === "unknown") return false;
    // Strip exact literals from scalar params so the warmup key matches
    // the hot key (mirrors the function-level JIT in index.ts).
    if (t.kind === "number" && t.exact !== undefined) {
      const pruned: JitType = { kind: "number" };
      if (t.sign !== undefined) pruned.sign = t.sign;
      if (t.isInteger) pruned.isInteger = true;
      t = pruned;
    }
    inputs.push(name);
    inputValues.push(val);
    inputTypes.push(t);
  }

  // Outputs: all assigned variables (deduplicated). We then filter this to
  // only the variables the surrounding scope actually needs after the
  // loop. Loop-internal temporaries that aren't read post-loop end up
  // unwritten and the JIT'd function can keep them in registers — V8
  // pessimizes register allocation if too many locals are live at the
  // function exit.
  const outputSet = new Set(rawOutputs);
  let outputs = [...outputSet];

  // Live-out filter: a variable should be written back if any of:
  //   1. it's an input (was in the env before the loop), or
  //   2. it's the loop's iteration variable (MATLAB exposes the final
  //      value), or
  //   3. it's read by a sibling stmt that runs after this loop in the
  //      enclosing block.
  // For (3) we walk just the sibling-tail provided by execStmts on the
  // interpreter; cross-block flow is conservatively ignored (those vars
  // get covered by the input check anyway, since they'd have to make it
  // back into env via assignment somewhere upstream).
  const inputSet = new Set(inputs);
  const liveOut = new Set<string>(inputSet);
  if (stmt.type === "For") liveOut.add(stmt.varName);
  if (interp._postSiblings && interp._postSiblingsIdx > 0) {
    collectReadsFromSiblings(
      interp._postSiblings,
      interp._postSiblingsIdx,
      liveOut
    );
  }
  outputs = outputs.filter(o => liveOut.has(o));

  // Progressive type widening: unify with previously seen types to prevent
  // unbounded specializations when called from an interpreted loop.
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

  // Build cache key from AST location + input types
  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const cacheKey = `${loc}|${typeKey}`;

  // Check JS-JIT cache
  if (interp.loopJitCache.has(cacheKey)) {
    const entry = interp.loopJitCache.get(cacheKey)!;
    if (entry === null) return false; // previously failed
    return executeAndWriteBack(
      interp,
      entry.fn,
      inputValues,
      outputs,
      cacheKey
    );
  }

  // Fast path: previously-compiled C-JIT loop specialization.
  if (interp.optimization >= 2 && interp.loopCJitCache.has(cacheKey)) {
    const cEntry = interp.loopCJitCache.get(cacheKey);
    if (cEntry) {
      return executeAndWriteBack(
        interp,
        cEntry.fn,
        inputValues,
        outputs,
        cacheKey
      );
    }
    // cEntry === null means the C compile previously failed; fall through
    // to the JS-JIT path below.
  }

  // Build a synthetic FunctionDef wrapping the loop statement
  const syntheticFn: FunctionDef = {
    name: `$loop_${kind}`,
    params: inputs,
    outputs,
    body: [stmt],
  };

  // Attempt lowering
  const lowered = lowerFunction(
    syntheticFn,
    inputTypes,
    outputs.length,
    interp
  );
  if (!lowered) {
    interp.loopJitCache.set(cacheKey, null);
    interp.loopCJitCache.set(cacheKey, null);
    return false;
  }

  // ── C-JIT path (--opt >= 2) ─────────────────────────────────────────
  // Mirror jit/index.ts: try the backend; on null, fall through to JS-JIT.
  if (interp.optimization >= 2 && !interp.loopCJitCache.has(cacheKey)) {
    const backend = getCJitBackend();
    if (backend) {
      const compiledCFn = backend.tryCompile(
        interp,
        syntheticFn,
        lowered.body,
        lowered.outputNames,
        lowered.localVars,
        lowered.outputType,
        lowered.outputTypes,
        inputTypes,
        outputs.length
      );
      if (compiledCFn) {
        interp.loopCJitCache.set(cacheKey, { fn: compiledCFn });
        return executeAndWriteBack(
          interp,
          compiledCFn,
          inputValues,
          outputs,
          cacheKey
        );
      }
      interp.loopCJitCache.set(cacheKey, null);
    }
  }

  // Generate JavaScript
  const currentFile = interp.currentFile;
  const mainBody = generateJS(
    lowered.body,
    inputs,
    lowered.outputNames,
    outputs.length,
    lowered.localVars,
    currentFile,
    interp.fuse
  );

  // Prepend generated helper function definitions
  const parts: string[] = [];
  for (const [, code] of lowered.generatedFns) {
    parts.push(code.replace(/^/gm, "  "));
  }
  parts.push(mainBody);
  const jsBody = parts.join("\n");

  // Create compiled function — always pass $h and $rt for line tracking.
  // Prefer the per-runtime helpers (built once after all dynamic builtins
  // are registered) so V8 sees a stable hidden class on $h and inlines
  // the per-iter helper calls in hot loops.
  let compiledFn: (...args: unknown[]) => unknown;
  const rt = interp.rt;
  try {
    const factory = new Function("$h", "$rt", ...inputs, jsBody);
    const helpers = rt.jitHelpers ?? jitHelpers;
    compiledFn = (...callArgs: unknown[]) => factory(helpers, rt, ...callArgs);
  } catch {
    interp.loopJitCache.set(cacheKey, null);
    return false;
  }

  // Build source for logging
  const paramComments = inputs
    .map((p, i) => `${p}: ${jitTypeKey(inputTypes[i])}`)
    .join(", ");
  const source = `// JIT loop (${kind}): (${paramComments})\nfunction $loop_${kind}(${inputs.join(", ")}) {\n${jsBody}\n}`;

  interp.loopJitCache.set(cacheKey, { fn: compiledFn, source });

  // Fire logging callback
  const line = interp.rt.$line ?? 0;
  const description = `loop:${kind}@${line}(${inputs.map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`).join(", ")})`;
  interp.onJitCompile?.(description, source);

  // Execute
  return executeAndWriteBack(
    interp,
    compiledFn,
    inputValues,
    outputs,
    cacheKey
  );
}

function executeAndWriteBack(
  interp: Interpreter,
  compiledFn: (...args: unknown[]) => unknown,
  inputValues: unknown[],
  outputs: string[],
  cacheKey?: string
): boolean {
  let result: unknown;
  try {
    result = compiledFn(...inputValues);
  } catch (e) {
    if (e instanceof JitFuncHandleBailError) {
      // Function handle returned a different type than the JIT expected.
      // Warn, invalidate the cache entry, and fall back to interpretation.
      console.warn(`Warning: ${e.message}`);
      if (cacheKey) interp.loopJitCache.set(cacheKey, null);
      return false;
    }
    if (e instanceof JitBailToInterpreter) {
      // Helper hit a case the JIT can't handle (e.g. col-slice write that
      // requires growing dst). Silently fall back to the interpreter,
      // which re-runs the loop from the original env state.
      return false;
    }
    throw e; // Let other runtime errors propagate
  }

  // Write back output variables
  if (outputs.length === 1) {
    interp.env.set(outputs[0], ensureRuntimeValue(result) as RuntimeValue);
  } else {
    const arr = result as unknown[];
    for (let i = 0; i < outputs.length; i++) {
      interp.env.set(outputs[i], ensureRuntimeValue(arr[i]) as RuntimeValue);
    }
  }

  return true;
}

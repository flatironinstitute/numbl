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
import type { JitType, JitCacheEntry } from "./jitTypes.js";
import { jitTypeKey } from "./jitTypes.js";
import { lowerFunction } from "./jitLower.js";
import { generateJS } from "./jitCodegen.js";
import { jitHelpers } from "./jitHelpers.js";
import { inferJitType } from "../builtins/types.js";
import { analyzeForLoop, analyzeWhileLoop } from "./jitLoopAnalysis.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../../runtime/types.js";

// ── Loop JIT cache (keyed by cache string) ──────────────────────────────

const loopJitCache = new Map<string, JitCacheEntry | null>();

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
  // and are not known constants/builtins
  const inputs: string[] = [];
  const inputValues: unknown[] = [];
  const inputTypes: JitType[] = [];

  for (const name of rawInputs) {
    if (KNOWN_CONSTANTS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) continue; // not a variable in scope — likely a function name
    const t = inferJitType(val);
    if (t.kind === "unknown") return false;
    // Deduplicate
    if (inputs.includes(name)) continue;
    inputs.push(name);
    inputValues.push(val);
    inputTypes.push(t);
  }

  // Outputs: all assigned variables (deduplicated)
  const outputSet = new Set(rawOutputs);
  // Also include any input that is assigned (it's both input and output)
  const outputs = [...outputSet];

  // Build cache key from AST location + input types
  const loc = stmt.span
    ? `${stmt.span.file}:${stmt.span.start}`
    : `loop:${kind}`;
  const typeKey = inputs
    .map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`)
    .join(",");
  const cacheKey = `${loc}|${typeKey}`;

  // Check cache
  if (loopJitCache.has(cacheKey)) {
    const entry = loopJitCache.get(cacheKey)!;
    if (entry === null) return false; // previously failed
    // Log even on cache hit so the UI shows JIT info across re-runs
    interp.onJitCompile?.(
      `loop:${kind}@${interp.rt.$line ?? 0}(cached)`,
      entry.source
    );
    return executeAndWriteBack(interp, entry.fn, inputValues, outputs);
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
    loopJitCache.set(cacheKey, null);
    return false;
  }

  // Generate JavaScript
  const mainBody = generateJS(
    lowered.body,
    inputs,
    lowered.outputNames,
    outputs.length,
    lowered.localVars
  );

  // Prepend generated helper function definitions
  const parts: string[] = [];
  for (const [, code] of lowered.generatedFns) {
    parts.push(code.replace(/^/gm, "  "));
  }
  parts.push(mainBody);
  const jsBody = parts.join("\n");

  // Create compiled function
  let compiledFn: (...args: unknown[]) => unknown;
  try {
    if (lowered.hasTensorOps) {
      const factory = new Function("$h", ...inputs, jsBody);
      compiledFn = (...callArgs: unknown[]) => factory(jitHelpers, ...callArgs);
    } else {
      const factory = new Function(...inputs, jsBody);
      compiledFn = (...callArgs: unknown[]) => factory(...callArgs);
    }
  } catch {
    loopJitCache.set(cacheKey, null);
    return false;
  }

  // Build source for logging
  const paramComments = inputs
    .map((p, i) => `${p}: ${jitTypeKey(inputTypes[i])}`)
    .join(", ");
  const source = `// JIT loop (${kind}): (${paramComments})\nfunction $loop_${kind}(${inputs.join(", ")}) {\n${jsBody}\n}`;

  loopJitCache.set(cacheKey, { fn: compiledFn, source });

  // Fire logging callback
  const line = interp.rt.$line ?? 0;
  const description = `loop:${kind}@${line}(${inputs.map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`).join(", ")})`;
  interp.onJitCompile?.(description, source);

  // Execute
  return executeAndWriteBack(interp, compiledFn, inputValues, outputs);
}

function executeAndWriteBack(
  interp: Interpreter,
  compiledFn: (...args: unknown[]) => unknown,
  inputValues: unknown[],
  outputs: string[]
): boolean {
  let result: unknown;
  try {
    result = compiledFn(...inputValues);
  } catch {
    return false; // runtime error — fall back to interpreter
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

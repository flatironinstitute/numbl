/**
 * Shared helpers for the JS-JIT shape modules.
 *
 * The three shape modules (`jitTopLevel.ts`, `jitLoop.ts`,
 * `jitCall.ts`) all do the same handful of things — type inference
 * with `exact` pruning, progressive type widening, JS body assembly,
 * `new Function` instantiation, env writeback. Those bits live here
 * so the shape modules can stay focused on what's actually different
 * between them (synthetic-fn shape, cacheKey scheme, run-time entry).
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Runtime } from "../../runtime/runtime.js";
import type { JitType } from "../../jit/jitTypes.js";
import { unifyJitTypes } from "../../jit/jitTypes.js";
import { lowerFunction, type LoweringResult } from "../../jit/jitLower.js";
import type { Stmt } from "../../parser/types.js";
import type { FunctionDef } from "../../interpreter/types.js";
import { generateJS } from "./js/jitCodegen.js";
import {
  jitHelpers,
  JitFuncHandleBailError,
  JitBailToInterpreter,
} from "./js/jitHelpers.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../../runtime/types.js";

/** Names that look like variables but never resolve to env values
 *  inside the JIT (constants, literals, the special `end` slot, the
 *  imaginary unit). Skipped when collecting env inputs. */
export const KNOWN_CONSTANTS: ReadonlySet<string> = new Set([
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

/**
 * Drop `exact` from a numeric scalar `JitType`. Numeric `exact` only
 * survives unification when two consecutive specializations see the
 * *same* literal — almost never the case for variables — so stripping
 * up front means the first specialization's cacheKey already matches
 * later calls.
 */
export function pruneArgType(t: JitType): JitType {
  if (t.kind === "number" && t.exact !== undefined) {
    const pruned: JitType = { kind: "number" };
    if (t.sign !== undefined) pruned.sign = t.sign;
    if (t.isInteger) pruned.isInteger = true;
    return pruned;
  }
  return t;
}

/**
 * Progressive type widening: in-place unify each entry of `types`
 * with the corresponding entry of `prev`. No-op when shapes don't
 * match (different arity → different specialization, no widening).
 */
export function widenAgainst(
  types: JitType[],
  prev: readonly JitType[] | undefined
): void {
  if (!prev || prev.length !== types.length) return;
  for (let i = 0; i < types.length; i++) {
    types[i] = unifyJitTypes(types[i], prev[i]);
  }
}

/**
 * Gather env inputs for the synthetic FunctionDef of a top-level or
 * loop block. For each candidate name: skip known constants, skip
 * names not in env (likely fn names), infer the JIT type, prune
 * `exact`. Returns null if any candidate has an unknown type — that's
 * a structural blocker for lowering.
 */
export function gatherTypedEnvInputs(
  interp: Interpreter,
  candidates: readonly string[]
): { inputs: string[]; inputTypes: JitType[] } | null {
  const inputs: string[] = [];
  const inputTypes: JitType[] = [];
  for (const name of candidates) {
    if (KNOWN_CONSTANTS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) continue;
    const t = inferJitType(val);
    if (t.kind === "unknown") return null;
    inputs.push(name);
    inputTypes.push(pruneArgType(t));
  }
  return { inputs, inputTypes };
}

/**
 * Read the live env values for `inputs` in order. Used at run time
 * to bind the compiled fn's parameters before invocation.
 */
export function gatherEnvValues(
  interp: Interpreter,
  inputs: readonly string[]
): unknown[] {
  const values: unknown[] = [];
  for (const name of inputs) {
    values.push(interp.env.get(name));
  }
  return values;
}

/**
 * Build the JS body for a synthetic FunctionDef: prepend each
 * generated callee body (indented 2 spaces to nest inside the
 * outer function), then append the main body.
 */
export function assembleJsBody(
  result: LoweringResult,
  mainBody: string
): string {
  const parts: string[] = [];
  for (const [, code] of result.generatedFns) {
    parts.push(code.replace(/^/gm, "  "));
  }
  parts.push(mainBody);
  return parts.join("\n");
}

/**
 * Wrap a JS body string in a `new Function(...)` and bind it to the
 * runtime's helpers and `$rt`. Returns null when the body fails to
 * parse (defensive — should not happen for IR that lowering accepted).
 */
export function instantiateJsFn(
  rt: Runtime,
  paramNames: readonly string[],
  jsBody: string
): ((...args: unknown[]) => unknown) | null {
  try {
    const factory = new Function("$h", "$rt", ...paramNames, jsBody);
    const helpers = rt.jitHelpers ?? jitHelpers;
    return (...callArgs: unknown[]) => factory(helpers, rt, ...callArgs);
  } catch {
    return null;
  }
}

/**
 * Write a compiled fn's return value back into the interpreter env.
 * The compiled fn returns either a single value (for one-output fns)
 * or an array of values (for multi-output) — matching the JS-JIT
 * convention used in the wrapper layer.
 */
export function writeBackOutputs(
  interp: Interpreter,
  outputs: readonly string[],
  result: unknown
): void {
  if (outputs.length === 0) return;
  if (outputs.length === 1) {
    interp.env.set(outputs[0], ensureRuntimeValue(result) as RuntimeValue);
    return;
  }
  const arr = result as unknown[];
  for (let i = 0; i < outputs.length; i++) {
    interp.env.set(outputs[i], ensureRuntimeValue(arr[i]) as RuntimeValue);
  }
}

// ── Block-shaped lowering / codegen / run mechanics ──────────────────────
//
// Top-level script bodies and for/while loops both wrap a stmt list as
// a synthetic FunctionDef and run it through the same JS-JIT pipeline.
// The three helpers below capture that shared mechanism so each shape
// module only spells out its own wiring (synthetic-fn name, source
// labels, diagnostics).

/**
 * Build a synthetic `FunctionDef` from (name, inputs, outputs, body)
 * and lower it through `lowerFunction`. Returns null when lowering
 * declines (constructs the JS-JIT IR doesn't model).
 */
export function lowerSyntheticFn(
  interp: Interpreter,
  name: string,
  inputs: readonly string[],
  inputTypes: readonly JitType[],
  outputs: readonly string[],
  body: Stmt[]
): LoweringResult | null {
  const syntheticFn: FunctionDef = {
    name,
    params: [...inputs],
    outputs: [...outputs],
    body,
  };
  return lowerFunction(syntheticFn, [...inputTypes], outputs.length, interp);
}

/**
 * Generate JS for a synthetic-fn lowered IR and instantiate it.
 * Returns null when `new Function` throws (defensive — should not
 * happen for IR that lowering accepted). Both top-level and loop
 * codegen use this; the shape module then wraps the returned fn in
 * its `*Compiled` shape and emits its own source/diagnostics.
 */
export function generateSyntheticFnJS(
  interp: Interpreter,
  result: LoweringResult,
  inputs: readonly string[],
  outputCount: number
): { fn: (...args: unknown[]) => unknown; jsBody: string } | null {
  const mainBody = generateJS(
    result.body,
    [...inputs],
    result.outputNames,
    outputCount,
    result.localVars,
    interp.currentFile
  );
  const jsBody = assembleJsBody(result, mainBody);
  const fn = instantiateJsFn(interp.rt, inputs, jsBody);
  if (!fn) return null;
  return { fn, jsBody };
}

/** Outcome of `runSyntheticFnAgainstEnv`. `transient` distinguishes
 *  the recoverable JitBailToInterpreter case (cache preserved) from
 *  the hard JitFuncHandleBailError case (cache invalidated). */
export type SyntheticFnRunResult =
  | { ok: true }
  | { ok: false; transient: boolean };

/**
 * Gather input values from env, invoke the compiled fn, write back
 * outputs. Translates JIT bails to a SyntheticFnRunResult. Used by
 * both top-level and loop run phases.
 */
export function runSyntheticFnAgainstEnv(
  interp: Interpreter,
  fn: (...args: unknown[]) => unknown,
  inputs: readonly string[],
  outputs: readonly string[]
): SyntheticFnRunResult {
  const inputValues = gatherEnvValues(interp, inputs);

  let result: unknown;
  try {
    result = fn(...inputValues);
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

  writeBackOutputs(interp, outputs, result);
  return { ok: true };
}

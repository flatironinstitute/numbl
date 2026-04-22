/**
 * JIT compilation for the top-level script body (the main workspace).
 *
 * Wraps the list of non-function, non-classdef statements of a script as
 * a synthetic `FunctionDef` whose parameters are the live-in env vars
 * and whose outputs are every variable assigned in the script. On
 * success the compiled code runs once and all output values are written
 * back to the interpreter's workspace env.
 *
 * Mirrors `tryJitLoop` in jitLoop.ts — same lowering, same JS/C backend
 * pipeline, same progressive type widening. The differences:
 *   - the synthetic body is the list of stmts directly, not a single For/While
 *   - every assigned variable is live-out (the whole workspace is live)
 *   - cache key is per-Interpreter (a single script AST per interp run)
 */
import type { Interpreter } from "../interpreter/interpreter.js";
import type { Stmt } from "../parser/types.js";
import type { FunctionDef } from "../interpreter/types.js";
import type { JitType } from "./jitTypes.js";
import { jitTypeKey, unifyJitTypes } from "./jitTypes.js";
import { lowerFunction } from "./jitLower.js";
import { generateJS } from "./js/jitCodegen.js";
import {
  jitHelpers,
  JitFuncHandleBailError,
  JitBailToInterpreter,
} from "./js/jitHelpers.js";
import { getCJitBackend } from "./c/registry.js";
import { compileHybridCallees, compileHybridLoops } from "./c/hybrid.js";
import { CJitParityError, formatCJitParityMessage } from "./c/parityError.js";
import { inferJitType } from "../interpreter/builtins/types.js";
import { analyzeTopLevel } from "./jitLoopAnalysis.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import type { RuntimeValue } from "../runtime/types.js";
import { JIT_IO_BUILTINS, irHasBailRisk, irHasIO } from "./jitBailSafety.js";

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

/**
 * Attempt to JIT-compile and execute the top-level script body.
 * Returns true if JIT succeeded, false to fall back to interpretation.
 */
export function tryJitTopLevel(interp: Interpreter, stmts: Stmt[]): boolean {
  if (stmts.length === 0) return false;

  // MATLAB displays unsuppressed top-level statements. The JIT has no
  // emit for auto-display, so when display is on we bail for any stmt
  // that would normally print. Void-call ExprStmts (disp/fprintf/etc.)
  // don't auto-display — MATLAB doesn't set `ans` for them — so an
  // unsuppressed void call is fine to JIT.
  if (interp.rt.displayResults) {
    for (const s of stmts) {
      if (
        (s.type === "Assign" ||
          s.type === "AssignLValue" ||
          s.type === "MultiAssign") &&
        !s.suppressed
      ) {
        return false;
      }
      if (s.type === "ExprStmt" && !s.suppressed) {
        const e = s.expr;
        const isVoidCall =
          e.type === "FuncCall" &&
          (JIT_IO_BUILTINS.has(e.name) || e.name === "tic");
        if (!isVoidCall) return false;
      }
    }
  }

  const analysis = analyzeTopLevel(stmts);
  if (analysis.hasReturn) return false;

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
  const inputValues: unknown[] = [];
  const inputTypes: JitType[] = [];

  for (const name of inputCandidates) {
    if (KNOWN_CONSTANTS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) continue; // not a variable in scope (likely a fn name)
    let t = inferJitType(val);
    if (t.kind === "unknown") return false;
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

  // Every assigned name is live-out — no liveness filter here.
  const outputs = [...new Set(analysis.outputs)];

  // Cache key: fixed per-interp location + input-type signature. Progressive
  // type widening matches the loop-JIT pattern (rarely matters in practice
  // since the top-level body normally runs once, but keeps semantics
  // consistent if the same interp ever re-runs the same script).
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

  // JS-JIT cache fast path
  if (interp.loopJitCache.has(cacheKey)) {
    const entry = interp.loopJitCache.get(cacheKey)!;
    if (entry === null) return false;
    return executeAndWriteBack(
      interp,
      entry.fn,
      inputValues,
      outputs,
      cacheKey
    );
  }

  // C-JIT cache fast path
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
    // null means previous C-JIT compile failed; fall through to JS-JIT.
  }

  const syntheticFn: FunctionDef = {
    name: `$top`,
    params: inputs,
    outputs,
    body: stmts,
  };

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

  // Bail-safety gate: if the body contains I/O (disp/fprintf/…) and
  // *any* mid-execution bail could fire, decline to JIT. Re-running
  // via the interpreter after a bail would duplicate already-emitted
  // output, which the user would see.
  if (
    irHasIO(lowered.body, lowered.generatedIRBodies) &&
    irHasBailRisk(lowered.body, lowered.generatedIRBodies)
  ) {
    interp.loopJitCache.set(cacheKey, null);
    interp.loopCJitCache.set(cacheKey, null);
    return false;
  }

  // ── C-JIT path (--opt >= 2) ─────────────────────────────────────────
  let cJitBail: {
    kind: "infeasible" | "env";
    reason: string;
    line?: number;
  } | null = null;
  if (interp.optimization >= 2 && !interp.loopCJitCache.has(cacheKey)) {
    const backend = getCJitBackend();
    if (backend) {
      const res = backend.tryCompile(
        interp,
        syntheticFn,
        lowered.body,
        lowered.outputNames,
        lowered.localVars,
        lowered.outputType,
        lowered.outputTypes,
        inputTypes,
        outputs.length,
        lowered.generatedIRBodies
      );
      if (res.ok) {
        interp.loopCJitCache.set(cacheKey, { fn: res.fn });
        return executeAndWriteBack(
          interp,
          res.fn,
          inputValues,
          outputs,
          cacheKey
        );
      }
      interp.loopCJitCache.set(cacheKey, null);
      cJitBail = { kind: res.kind, reason: res.reason, line: res.line };
    }
  }

  // Hybrid: top-level body bailed from C-JIT but individual callees
  // and top-level For/While loops may still be C-feasible. Swap those
  // in before JS codegen so the JS-JIT'd top-level calls native code
  // at the inner boundaries.
  compileHybridCallees(interp, lowered.generatedIRBodies, lowered.generatedFns);
  compileHybridLoops(
    interp,
    lowered.body,
    lowered.endEnv,
    lowered.outputNames,
    lowered.generatedIRBodies,
    lowered.generatedFns
  );

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

  const parts: string[] = [];
  for (const [, code] of lowered.generatedFns) {
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
    interp.loopJitCache.set(cacheKey, null);
    return false;
  }

  if (
    interp.checkCJitParity &&
    cJitBail &&
    !irHasIO(lowered.body, lowered.generatedIRBodies)
  ) {
    throw new CJitParityError(
      formatCJitParityMessage({
        kind: cJitBail.kind,
        reason: cJitBail.reason,
        reasonLine: cJitBail.line,
        siteLabel: `top-level`,
        file: currentFile,
        callSiteLine: cJitBail.line ?? 0,
        argsDesc: inputs
          .map((n, i) => `${n}: ${jitTypeKey(inputTypes[i])}`)
          .join(", "),
      }),
      cJitBail.reason,
      cJitBail.kind
    );
  }

  const paramComments = inputs
    .map((p, i) => `${p}: ${jitTypeKey(inputTypes[i])}`)
    .join(", ");
  const source = `// JIT top-level: (${paramComments})\nfunction $top(${inputs.join(", ")}) {\n${jsBody}\n}`;

  interp.loopJitCache.set(cacheKey, { fn: compiledFn, source });

  const description = `top-level@${currentFile}(${inputs.map((n, i) => `${n}:${jitTypeKey(inputTypes[i])}`).join(", ")})`;
  interp.onJitCompile?.(description, source);

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
      console.warn(`Warning: ${e.message}`);
      if (cacheKey) interp.loopJitCache.set(cacheKey, null);
      return false;
    }
    if (e instanceof JitBailToInterpreter) {
      return false;
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

  return true;
}

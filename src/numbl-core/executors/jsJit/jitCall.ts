/**
 * JIT compilation entry point for interpreter function calls.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { FunctionDef } from "../../interpreter/types.js";
import {
  type JitType,
  type JitCacheEntry,
  computeJitCacheKey,
  jitTypeKey,
  unifyJitTypes,
} from "../../jit/jitTypes.js";
import { lowerFunction } from "../../jit/jitLower.js";
import { generateJS } from "./js/jitCodegen.js";
import { jitHelpers, JitBailToInterpreter } from "./js/jitHelpers.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import { irHasBailRisk, irHasIO } from "../../jit/jitBailSafety.js";
import { tryEmitScalarFnKernel } from "./e1/scalarFnKernel.js";

export const JIT_SKIP = Symbol("JIT_SKIP");

/** Augmented FunctionDef with the per-specialization JIT cache. */
interface FunctionDefWithCache extends FunctionDef {
  _jitCache?: Map<string, JitCacheEntry | null>;
  _lastJitArgTypes?: Map<number, JitType[]>;
}

// ── Main entry point ────────────────────────────────────────────────────

export function tryJitCall(
  interp: Interpreter,
  fn: FunctionDef,
  args: unknown[],
  nargout: number
): unknown | typeof JIT_SKIP {
  // Determine argument types. For numeric scalar params, drop the
  // literal `exact` field up front: it only survives unification when
  // two consecutive calls pass the *same* literal, which almost never
  // happens for user-function params (callers pass variables, not
  // constants). Stripping means the first call's cache key already
  // matches subsequent calls — one warmup suffices to land a reusable
  // specialization. String/char `value`, boolean `value`, and `sign` /
  // `isInteger` are preserved; they're useful for dispatch and widen
  // naturally via the progressive unification below.
  const argTypes: JitType[] = [];
  for (const arg of args) {
    const t = inferJitType(arg);
    if (t.kind === "unknown") return JIT_SKIP;
    if (t.kind === "number" && t.exact !== undefined) {
      const pruned: JitType = { kind: "number" };
      if (t.sign !== undefined) pruned.sign = t.sign;
      if (t.isInteger) pruned.isInteger = true;
      argTypes.push(pruned);
    } else {
      argTypes.push(t);
    }
  }

  // Progressive type widening: unify with previously seen types to prevent
  // unbounded specializations when called from an interpreted loop.
  const fnWithCache = fn as FunctionDefWithCache;
  if (!fnWithCache._lastJitArgTypes) {
    fnWithCache._lastJitArgTypes = new Map();
  }
  const prevTypes = fnWithCache._lastJitArgTypes.get(nargout);
  if (prevTypes && prevTypes.length === argTypes.length) {
    for (let i = 0; i < argTypes.length; i++) {
      argTypes[i] = unifyJitTypes(argTypes[i], prevTypes[i]);
    }
  }
  fnWithCache._lastJitArgTypes.set(nargout, argTypes.slice());

  const cacheKey = computeJitCacheKey(nargout, argTypes);

  // Check cache
  if (!fnWithCache._jitCache) {
    fnWithCache._jitCache = new Map();
  }

  if (fnWithCache._jitCache.has(cacheKey)) {
    const entry = fnWithCache._jitCache.get(cacheKey)!;
    if (entry === null) return JIT_SKIP; // previously failed
    return runWithCallFrame(interp, fn.name, entry.fn, args);
  }

  // Attempt lowering (pass interpreter for user function resolution)
  const lowered = lowerFunction(fn, argTypes, nargout, interp);
  if (!lowered) {
    fnWithCache._jitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // Bail-safety gate: a function that emits I/O (disp/fprintf/…) must
  // be bail-free, or a mid-execution bail would re-run the call under
  // the interpreter and duplicate already-printed output.
  if (
    irHasIO(lowered.body, lowered.generatedIRBodies) &&
    irHasBailRisk(lowered.body, lowered.generatedIRBodies)
  ) {
    fnWithCache._jitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // ── e1 scalar whole-function kernel path ─────────────────────────────
  // Under --opt e1, if the entire function is pure-scalar, emit a JS
  // wrapper that compiles the whole body to C on first call and
  // dispatches through `$h.compileKernel`. The C source is visible
  // inline in --dump-js output.
  if (interp.experimental === "e1") {
    const scalarKernel = tryEmitScalarFnKernel(
      interp,
      fn,
      lowered.body,
      lowered.outputNames,
      lowered.localVars,
      lowered.outputType,
      lowered.outputTypes,
      argTypes,
      nargout,
      lowered.generatedIRBodies
    );
    if (scalarKernel) {
      const rt = interp.rt;
      const helpers = rt.jitHelpers ?? jitHelpers;
      try {
        const factory = new Function(
          "$h",
          "$rt",
          `${scalarKernel.jsSource}\nreturn ${fn.name};`
        );
        const wrapped = factory(helpers, rt) as (...a: unknown[]) => unknown;
        const compiled = (...callArgs: unknown[]) => wrapped(...callArgs);
        const typeDesc = argTypes.map(jitTypeKey).join(", ");
        const paramComments = fn.params
          .map((p, i) => `${p}: ${jitTypeKey(argTypes[i])}`)
          .join(", ");
        const outputComments = lowered.outputNames
          .map(
            o =>
              `${o}: ${lowered.outputType ? jitTypeKey(lowered.outputType) : "unknown"}`
          )
          .join(", ");
        const source =
          `// JIT (e1 scalar kernel): ${fn.name}(${paramComments}) -> (${outputComments})\n` +
          `// from: ${interp.currentFile}\n` +
          scalarKernel.jsSource;
        fnWithCache._jitCache.set(cacheKey, { fn: compiled, source });
        const line = interp.rt.$line ?? 0;
        const description = `${fn.name}@${line}(${typeDesc}) -> e1-scalar-kernel`;
        interp.onJitCompile?.(description, source);
        return runWithCallFrame(interp, fn.name, compiled, args);
      } catch {
        // Fall through to the regular JS-JIT path on any hiccup.
      }
    }
  }

  // Generate JavaScript for the main function body
  const currentFile = interp.currentFile;
  const mainBody = generateJS(
    lowered.body,
    fn.params,
    lowered.outputNames,
    nargout,
    lowered.localVars,
    currentFile,
    interp.experimental,
    interp.par
  );

  // Prepend generated helper function definitions (indented to match main body)
  const parts: string[] = [];
  for (const [, code] of lowered.generatedFns) {
    parts.push(code.replace(/^/gm, "  "));
  }
  parts.push(mainBody);
  const jsBody = parts.join("\n");

  // Create function — always pass $h (helpers) and $rt (runtime) for line tracking
  let compiledFn: (...args: unknown[]) => unknown;
  const paramNames = fn.params.map(p => p);
  const rt = interp.rt;

  try {
    const factory = new Function("$h", "$rt", ...paramNames, jsBody);
    const helpers = rt.jitHelpers ?? jitHelpers;
    compiledFn = (...callArgs: unknown[]) => factory(helpers, rt, ...callArgs);
  } catch {
    fnWithCache._jitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // Cache and log
  const typeDesc = argTypes.map(jitTypeKey).join(", ");
  const paramComments = fn.params
    .map((p, i) => `${p}: ${jitTypeKey(argTypes[i])}`)
    .join(", ");
  const outputComments = lowered.outputNames
    .map(
      o =>
        `${o}: ${lowered.outputType ? jitTypeKey(lowered.outputType) : "unknown"}`
    )
    .join(", ");
  const fnComment = [
    `// JIT: ${fn.name}(${paramComments}) -> (${outputComments})`,
    `// from: ${interp.currentFile}`,
  ].join("\n");
  const source = `${fnComment}\nfunction ${fn.name}(${paramNames.join(", ")}) {\n${jsBody}\n}`;
  fnWithCache._jitCache.set(cacheKey, { fn: compiledFn, source });

  // Fire logging callback (include call-site line number)
  const line = interp.rt.$line ?? 0;
  const description = `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`;
  interp.onJitCompile?.(description, source);

  // Execute — let most runtime errors propagate. JitBailToInterpreter is
  // caught below as a signal to re-run via the interpreter.
  const result = runWithCallFrame(interp, fn.name, compiledFn, args);
  return result;
}

/**
 * Execute a JIT-compiled function with proper call frame tracking.
 *
 * If the JIT body throws `JitBailToInterpreter` (e.g. a scalar index write
 * needs tensor growth, which the JIT's hoisted aliases can't represent),
 * returns `JIT_SKIP` so the caller re-runs the function via the interpreter.
 * Side effects accumulated before the bail may re-run.
 */
function runWithCallFrame(
  interp: Interpreter,
  name: string,
  compiledFn: (...args: unknown[]) => unknown,
  args: unknown[]
): unknown | typeof JIT_SKIP {
  interp.rt.pushCallFrame(name);
  interp.rt.pushCleanupScope();
  try {
    return compiledFn(...args);
  } catch (e) {
    if (e instanceof JitBailToInterpreter) return JIT_SKIP;
    interp.rt.annotateError(e);
    throw e;
  } finally {
    interp.rt.popAndRunCleanups(fn => {
      if (fn.jsFn) {
        if (fn.jsFnExpectsNargout) fn.jsFn(0);
        else fn.jsFn();
      } else {
        interp.rt.dispatch(fn.name, 0, []);
      }
    });
    interp.rt.popCallFrame();
  }
}

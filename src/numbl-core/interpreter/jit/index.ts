/**
 * JIT compilation entry point for interpreter function calls.
 */

import type { Interpreter } from "../interpreter.js";
import type { FunctionDef } from "../types.js";
import {
  type JitType,
  type JitCacheEntry,
  computeJitCacheKey,
  jitTypeKey,
  unifyJitTypes,
} from "./jitTypes.js";
import { lowerFunction } from "./jitLower.js";
import { generateJS } from "./jitCodegen.js";
import { jitHelpers, JitBailToInterpreter } from "./jitHelpers.js";
import { inferJitType } from "../builtins/types.js";
import { checkCFeasibility } from "./c/cFeasibility.js";
import { generateC } from "./c/jitCodegenC.js";
import { generateNapiShim } from "./c/cNapiShim.js";
import { compileAndLoad } from "./c/cCompile.js";

export const JIT_SKIP = Symbol("JIT_SKIP");

/** C-JIT cache entry (parallel to JitCacheEntry but with a native-wrapped fn). */
interface CJitCacheEntry {
  fn: (...args: unknown[]) => unknown;
  source: string;
  cachedPath: string;
}

/** Augmented FunctionDef with JIT caches (JS and C specializations are parallel). */
interface FunctionDefWithCache extends FunctionDef {
  _jitCache?: Map<string, JitCacheEntry | null>;
  _cJitCache?: Map<string, CJitCacheEntry | null>;
  _lastJitArgTypes?: Map<number, JitType[]>;
}

// ── Main entry point ────────────────────────────────────────────────────

export function tryJitCall(
  interp: Interpreter,
  fn: FunctionDef,
  args: unknown[],
  nargout: number
): unknown | typeof JIT_SKIP {
  // Determine argument types
  const argTypes: JitType[] = [];
  for (const arg of args) {
    const t = inferJitType(arg);
    if (t.kind === "unknown") return JIT_SKIP;
    argTypes.push(t);
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

  // Fast path: previously-compiled C-JIT specialization.
  if (interp.optimization >= 2) {
    if (!fnWithCache._cJitCache) fnWithCache._cJitCache = new Map();
    const cEntry = fnWithCache._cJitCache.get(cacheKey);
    if (cEntry !== undefined && cEntry !== null) {
      return runWithCallFrame(interp, fn.name, cEntry.fn, args);
    }
  }

  // Attempt lowering (pass interpreter for user function resolution)
  const lowered = lowerFunction(fn, argTypes, nargout, interp);
  if (!lowered) {
    fnWithCache._jitCache.set(cacheKey, null);
    if (fnWithCache._cJitCache) fnWithCache._cJitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // ── C-JIT path (--opt >= 2) ───────────────────────────────────────────
  // Feasibility prepass: if the scalar-only whitelist covers this IR,
  // emit C, compile, and cache. On any bail, fall through to the JS path
  // below so the user gets JS-JIT behavior transparently.
  if (interp.optimization >= 2) {
    const cResult = tryBuildCJit(
      interp,
      fn,
      lowered.body,
      lowered.outputNames,
      lowered.localVars,
      lowered.outputType,
      argTypes,
      nargout,
      cacheKey
    );
    if (cResult) {
      // Also seed the JS cache with a reference to the C entry so the
      // per-call fast path above returns the C fn for subsequent calls
      // through this cacheKey without reaching into _cJitCache again.
      return runWithCallFrame(interp, fn.name, cResult.fn, args);
    }
    // Fall through to JS-JIT path (cResult = null already recorded _cJitCache miss).
  }

  // Generate JavaScript for the main function body
  const currentFile = interp.currentFile;
  const mainBody = generateJS(
    lowered.body,
    fn.params,
    lowered.outputNames,
    nargout,
    lowered.localVars,
    currentFile
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

/**
 * Attempt to produce a C-JIT specialization for `fn` with the given
 * lowered IR. Returns the cache entry on success or null on any bail
 * (infeasible IR, compile failure, load failure).
 *
 * On success, writes the entry into `fn._cJitCache[cacheKey]`. On any
 * failure, writes `null` into that slot so we don't retry each call.
 */
function tryBuildCJit(
  interp: Interpreter,
  fn: FunctionDef,
  body: import("./jitTypes.js").JitStmt[],
  outputNames: string[],
  localVars: Set<string>,
  outputType: JitType | null,
  argTypes: JitType[],
  nargout: number,
  cacheKey: string
): CJitCacheEntry | null {
  const fnWithCache = fn as FunctionDefWithCache;
  if (!fnWithCache._cJitCache) fnWithCache._cJitCache = new Map();

  const feas = checkCFeasibility(body, argTypes, outputType, nargout);
  if (!feas.ok) {
    fnWithCache._cJitCache.set(cacheKey, null);
    return null;
  }

  const gen = generateC(
    body,
    fn.params,
    outputNames,
    nargout,
    localVars,
    // Use a hash-ish name so the C entry point is stable across identical
    // C sources but unique across specializations.
    `${fn.name.replace(/[^A-Za-z0-9_]/g, "_")}_${cacheKey.length.toString(16)}`
  );

  const argKinds = argTypes.map(t =>
    t.kind === "boolean" ? ("boolean" as const) : ("number" as const)
  );
  const returnKind: "boolean" | "number" =
    outputType && outputType.kind === "boolean" ? "boolean" : "number";
  const { shim, exportName } = generateNapiShim(
    gen.cFnName,
    argKinds,
    returnKind
  );

  const log = interp.log;
  const loaded = compileAndLoad(gen.cSource, shim, exportName, log);
  if (!loaded) {
    fnWithCache._cJitCache.set(cacheKey, null);
    return null;
  }

  // Wrap the native function to match the (...callArgs) => unknown signature.
  // The native fn accepts/returns `double` (or throws for non-number args
  // via napi_get_value_double — which would only happen if callers violate
  // the feasibility contract, e.g. by widening types mid-run).
  const nativeFn = loaded.fn;
  const compiledFn = (...callArgs: unknown[]): unknown => {
    return nativeFn(...(callArgs as number[]));
  };

  const entry: CJitCacheEntry = {
    fn: compiledFn,
    source: gen.cSource + "\n\n" + shim,
    cachedPath: loaded.cachedPath,
  };
  fnWithCache._cJitCache.set(cacheKey, entry);

  // Fire the C-JIT-specific logging callback (for --dump-c).
  const line = interp.rt.$line ?? 0;
  const typeDesc = argTypes.map(jitTypeKey).join(", ");
  const description = `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`;
  interp.onCJitCompile?.(description, entry.source);

  return entry;
}

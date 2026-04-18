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
import { getCJitBackend } from "./cJitBackend.js";
import { CJitParityError, formatCJitParityMessage } from "./cJitParityError.js";

export const JIT_SKIP = Symbol("JIT_SKIP");

/** C-JIT cache entry (parallel to JitCacheEntry but with a native-wrapped fn). */
interface CJitCacheEntry {
  fn: (...args: unknown[]) => unknown;
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
  // Delegated to a pluggable backend registered at startup by the CLI
  // entry point (src/numbl-core/interpreter/jit/cJitInstall.ts). The
  // browser bundle never reaches the install module, so its Node-only
  // transitive deps (child_process, fs, ...) stay out of the web build.
  // On any bail we fall through to the JS-JIT path, unless
  // `--check-c-jit-parity` is on (see below).
  let cJitBail: {
    kind: "infeasible" | "env";
    reason: string;
    line?: number;
  } | null = null;
  if (interp.optimization >= 2) {
    const backend = getCJitBackend();
    if (backend) {
      if (!fnWithCache._cJitCache) fnWithCache._cJitCache = new Map();
      const res = backend.tryCompile(
        interp,
        fn,
        lowered.body,
        lowered.outputNames,
        lowered.localVars,
        lowered.outputType,
        lowered.outputTypes,
        argTypes,
        nargout
      );
      if (res.ok) {
        fnWithCache._cJitCache.set(cacheKey, { fn: res.fn });
        return runWithCallFrame(interp, fn.name, res.fn, args);
      }
      fnWithCache._cJitCache.set(cacheKey, null);
      cJitBail = { kind: res.kind, reason: res.reason, line: res.line };
    }
    // Fall through to JS-JIT path.
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
    interp.fuse
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

  // ── Parity check (--check-c-jit-parity) ───────────────────────────────
  // We reach here only if C-JIT was attempted and declined. JS-JIT just
  // compiled successfully, so this is the parity gap the flag is meant to
  // surface — throw instead of silently downgrading.
  if (interp.checkCJitParity && cJitBail) {
    throw new CJitParityError(
      formatCJitParityMessage({
        kind: cJitBail.kind,
        reason: cJitBail.reason,
        reasonLine: cJitBail.line,
        siteLabel: `fn ${fn.name}`,
        file: interp.currentFile,
        callSiteLine: interp.rt.$line ?? 0,
        argsDesc: argTypes.map(jitTypeKey).join(", "),
      }),
      cJitBail.reason,
      cJitBail.kind
    );
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

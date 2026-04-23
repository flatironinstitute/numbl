/**
 * JIT compilation entry point for interpreter function calls.
 */

import type { Interpreter } from "../interpreter/interpreter.js";
import type { FunctionDef } from "../interpreter/types.js";
import {
  type JitType,
  type JitCacheEntry,
  computeJitCacheKey,
  jitTypeKey,
  unifyJitTypes,
} from "./jitTypes.js";
import { lowerFunction } from "./jitLower.js";
import { generateJS } from "./js/jitCodegen.js";
import { jitHelpers, JitBailToInterpreter } from "./js/jitHelpers.js";
import { inferJitType } from "../interpreter/builtins/types.js";
import { getCJitBackend } from "./c/registry.js";
import { compileHybridCallees, compileHybridLoops } from "./c/hybrid.js";
import { CJitParityError, formatCJitParityMessage } from "./c/parityError.js";
import { irHasBailRisk, irHasIO } from "./jitBailSafety.js";
import { tryEmitScalarFnKernel } from "./e1/scalarFnKernel.js";

export const JIT_SKIP = Symbol("JIT_SKIP");

// Opt-in JIT activity tally (env: NUMBL_LOG_CJIT_MISSES=1). Categories:
//   cjit-ok      — C-JIT specialization compiled successfully
//   cjit-miss    — C-JIT bail (falls through to JS-JIT); carries reason
//   jsjit-ok     — JS-JIT specialization compiled successfully
//   jsjit-skip   — JS-JIT lowering failed (never reaches C-JIT either)
// One unique (category, fn, key) triple → one row; counter = number of
// distinct cache keys that hit that row during the run. Emitted at exit.
const LOG_CJIT_MISSES =
  typeof process !== "undefined" && !!process.env.NUMBL_LOG_CJIT_MISSES;
const _jitTally = new Map<string, number>();
let _jitTallyExitInstalled = false;

function _ensureJitTallyExitHook(): void {
  if (_jitTallyExitInstalled) return;
  _jitTallyExitInstalled = true;
  process.on("exit", () => {
    const entries = Array.from(_jitTally.entries()).sort((a, b) => b[1] - a[1]);
    process.stderr.write(`\n[NUMBL_LOG_CJIT_MISSES] ${entries.length} rows:\n`);
    for (const [k, n] of entries) {
      process.stderr.write(`  ${n.toString().padStart(6)}  ${k}\n`);
    }
  });
}

function logJitEvent(category: string, fnName: string, extra: string): void {
  const key = `${category}\t${fnName}\t${extra}`;
  _jitTally.set(key, (_jitTally.get(key) ?? 0) + 1);
  _ensureJitTallyExitHook();
}

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
    if (LOG_CJIT_MISSES) {
      const reason =
        (fn as { _lastLowerBailReason?: string })._lastLowerBailReason ??
        "lowering returned null";
      logJitEvent("jsjit-skip", fn.name, reason);
    }
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
    if (fnWithCache._cJitCache) fnWithCache._cJitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // ── C-JIT path (--opt >= 2) ───────────────────────────────────────────
  // Delegated to a pluggable backend registered at startup by the CLI
  // entry point (src/numbl-core/jit/c/install.ts). The
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
        nargout,
        lowered.generatedIRBodies
      );
      if (res.ok) {
        fnWithCache._cJitCache.set(cacheKey, { fn: res.fn });
        if (LOG_CJIT_MISSES) {
          logJitEvent("cjit-ok", fn.name, "");
        }
        return runWithCallFrame(interp, fn.name, res.fn, args);
      }
      fnWithCache._cJitCache.set(cacheKey, null);
      cJitBail = { kind: res.kind, reason: res.reason, line: res.line };
      if (LOG_CJIT_MISSES) {
        logJitEvent(
          "cjit-miss",
          fn.name,
          `${res.kind}: ${res.reason}${res.line ? ` @L${res.line}` : ""}`
        );
      }
    }
    // Fall through to JS-JIT path.

    // Hybrid mode: even though the outer C-JIT declined, try C-JIT on
    // (a) each lowered callee and (b) each top-level For/While loop in
    // the outer body. Successes are swapped into the JS-JIT'd outer
    // via forwarder stubs so the outer calls native code at the inner
    // boundaries. See hybrid.ts.
    compileHybridCallees(
      interp,
      lowered.generatedIRBodies,
      lowered.generatedFns
    );
    compileHybridLoops(
      interp,
      lowered.body,
      lowered.endEnv,
      lowered.outputNames,
      lowered.generatedIRBodies,
      lowered.generatedFns
    );
  }

  // ── e1 scalar whole-function kernel path ─────────────────────────────
  // Under --opt e1, if the entire function is pure-scalar, emit a JS
  // wrapper that compiles the whole body to C on first call and
  // dispatches through `$h.compileKernel`. This is the e1 counterpart
  // to the --opt 2 whole-function C-JIT, but with the C source visible
  // inline in --dump-js.
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
    interp.fuse,
    interp.experimental
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
  if (LOG_CJIT_MISSES) {
    logJitEvent("jsjit-ok", fn.name, "");
  }

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

/**
 * JIT compilation for scalar-only functions and loops in the interpreter.
 *
 * When optimization >= 1, the interpreter attempts to compile pure scalar
 * functions and for-loops to native JS before falling back to AST interpretation.
 */

import type { Interpreter } from "./interpreter.js";
import type { Expr } from "../parser/types.js";
import type {
  FunctionDef,
  JitCacheEntry,
  LoopJitCacheEntry,
  LoopAnalysis,
} from "./types.js";
import {
  isScalarCompilable,
  generateScalarJS,
  analyzeLoopForJit,
  generateLoopJS,
} from "./jitCodegen.js";

/** Sentinel: JIT declined, fall back to interpreter. */
export const JIT_SKIP = Symbol("JIT_SKIP");

type CompiledFn = (...args: number[]) => number | number[];

function jitCacheKey(args: unknown[], nargout: number): string {
  let key = `${nargout}`;
  for (let i = 0; i < args.length; i++) {
    key += `:${typeof args[i]}`;
  }
  return key;
}

/**
 * Try to JIT-compile and call a user function.
 * Returns JIT_SKIP if JIT is not applicable.
 */
export function tryJitCall(
  interp: Interpreter,
  fn: FunctionDef,
  args: unknown[],
  nargout: number
): unknown {
  if (interp.optimization < 1) return JIT_SKIP;

  const key = jitCacheKey(args, nargout);

  // Check cache attached to the AST node
  if (!fn._jitCache) fn._jitCache = new Map();
  const cached = fn._jitCache.get(key);
  if (cached === null) return JIT_SKIP; // known non-compilable
  if (cached !== undefined) {
    return cached.fn(...(args as number[]));
  }

  // For now, only compile when all args are plain JS numbers
  for (let i = 0; i < args.length; i++) {
    if (typeof args[i] !== "number") {
      fn._jitCache.set(key, null);
      return JIT_SKIP;
    }
  }

  // Try to compile
  try {
    const locals = isScalarCompilable(fn);
    if (locals === null) {
      fn._jitCache.set(key, null);
      return JIT_SKIP;
    }

    const jsBody = generateScalarJS(fn, nargout);
    const paramNames = fn.params;
    const fullSource = `function ${fn.name}(${paramNames.join(", ")}) {\n${jsBody}\n}`;

    const compiled = new Function(...paramNames, jsBody) as CompiledFn;

    const entry: JitCacheEntry = { fn: compiled, source: fullSource };
    fn._jitCache.set(key, entry);

    if (interp.onJitCompile) {
      const sig = `${fn.name}(${paramNames.join(", ")}) [${key}]`;
      interp.onJitCompile(sig, fullSource);
    }

    return compiled(...(args as number[]));
  } catch {
    fn._jitCache.set(key, null);
    return JIT_SKIP;
  }
}

// ── Loop-level JIT ───────────────────────────────────────────────────────

/** Type for the For statement AST node with an attached JIT cache. */
interface ForStmtWithCache {
  type: "For";
  varName: string;
  expr: Expr;
  body: import("../parser/types.js").Stmt[];
  span: import("../parser/types.js").Span;
  _jitLoopCache?: LoopJitCacheEntry | null;
}

type LoopJitFn = (...args: number[]) => number[] | null;

/**
 * Try to JIT-compile and execute a for-loop.
 * Returns JIT_SKIP if JIT is not applicable, or null on success
 * (the loop was executed via JIT and variables written back).
 */
export function tryJitForLoop(
  interp: Interpreter,
  stmt: ForStmtWithCache
): typeof JIT_SKIP | null {
  if (interp.optimization < 1) return JIT_SKIP;
  if (stmt.expr.type !== "Range") return JIT_SKIP;

  // Check cache on the AST node
  const cached = stmt._jitLoopCache;
  if (cached === null) return JIT_SKIP; // known non-compilable

  let analysis: LoopAnalysis;
  let compiled: LoopJitFn;

  if (cached !== undefined) {
    analysis = cached.analysis;
    compiled = cached.fn;
  } else {
    // First time: analyze and compile
    try {
      const result = analyzeLoopForJit(stmt.body, stmt.varName);
      if (result === null) {
        stmt._jitLoopCache = null;
        return JIT_SKIP;
      }
      analysis = result;

      const jsBody = generateLoopJS(stmt.body, stmt.varName, analysis);
      const paramNames = [...analysis.readVars, "$_rs", "$_rst", "$_re"];
      const fullSource = `function _loop(${paramNames.join(", ")}) {\n${jsBody}\n}`;

      compiled = new Function(...paramNames, jsBody) as LoopJitFn;

      stmt._jitLoopCache = { fn: compiled, source: fullSource, analysis };

      if (interp.onJitCompile) {
        interp.onJitCompile(`loop@${stmt.varName}`, fullSource);
      }
    } catch {
      stmt._jitLoopCache = null;
      return JIT_SKIP;
    }
  }

  // Evaluate range bounds
  const rangeExpr = stmt.expr as Extract<Expr, { type: "Range" }>;
  const startVal = interp.evalExpr(rangeExpr.start);
  const endVal = interp.evalExpr(rangeExpr.end);
  const stepVal = rangeExpr.step ? interp.evalExpr(rangeExpr.step) : 1;

  if (
    typeof startVal !== "number" ||
    typeof endVal !== "number" ||
    typeof stepVal !== "number"
  ) {
    return JIT_SKIP;
  }

  // Build argument list: read variables + range bounds
  const writeSet = new Set(analysis.allWriteVars);
  const args: number[] = [];
  for (const v of analysis.readVars) {
    const val = interp.env.get(v);
    if (typeof val === "number") {
      args.push(val);
    } else if (val === undefined && writeSet.has(v)) {
      // Variable doesn't exist but is also written — assume write-before-read
      args.push(0);
    } else {
      return JIT_SKIP; // not a number or non-existent external
    }
  }
  args.push(startVal, stepVal, endVal);

  // Call compiled loop
  const result = compiled(...args);

  if (result === null) {
    // Loop didn't execute (empty range) — don't touch any variables
    return null;
  }

  // Write back all output variables + loop variable
  for (let i = 0; i < analysis.allWriteVars.length; i++) {
    const val = result[i];
    if (typeof val === "number") {
      interp.env.set(analysis.allWriteVars[i], val);
    }
  }
  // Loop variable is the last element
  const loopVarVal = result[analysis.allWriteVars.length];
  if (typeof loopVarVal === "number") {
    interp.env.set(analysis.loopVar, loopVarVal);
  }

  return null;
}

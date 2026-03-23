/**
 * JIT compilation entry point for interpreter function calls.
 */

import type { RuntimeTensor } from "../../runtime/types.js";
import type { Interpreter } from "../interpreter.js";
import type { FunctionDef } from "../types.js";
import {
  type JitType,
  type JitCacheEntry,
  computeJitCacheKey,
} from "./jitTypes.js";
import { lowerFunction } from "./jitLower.js";
import { generateJS } from "./jitCodegen.js";
import { jitHelpers } from "./jitHelpers.js";

export const JIT_SKIP = Symbol("JIT_SKIP");

/** Augmented FunctionDef with JIT cache. */
interface FunctionDefWithCache extends FunctionDef {
  _jitCache?: Map<string, JitCacheEntry | null>;
}

// ── Type inference from runtime values ──────────────────────────────────

function runtimeValueToJitType(value: unknown): JitType {
  if (typeof value === "number") {
    return { kind: "number", nonneg: value >= 0 };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "complex"
  ) {
    return { kind: "complex" };
  }
  if (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: string }).kind === "tensor"
  ) {
    const t = value as RuntimeTensor;
    if (t.imag) return { kind: "complexTensor" };
    return { kind: "realTensor" };
  }
  return { kind: "unknown" };
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
    const t = runtimeValueToJitType(arg);
    if (t.kind === "unknown") return JIT_SKIP;
    argTypes.push(t);
  }

  const cacheKey = computeJitCacheKey(nargout, argTypes);
  const fnWithCache = fn as FunctionDefWithCache;

  // Check cache
  if (!fnWithCache._jitCache) {
    fnWithCache._jitCache = new Map();
  }

  if (fnWithCache._jitCache.has(cacheKey)) {
    const entry = fnWithCache._jitCache.get(cacheKey)!;
    if (entry === null) return JIT_SKIP; // previously failed
    return entry.fn(...args);
  }

  // Attempt lowering
  const lowered = lowerFunction(fn, argTypes, nargout);
  if (!lowered) {
    fnWithCache._jitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // Generate JavaScript
  const jsBody = generateJS(
    lowered.body,
    fn.params,
    lowered.outputNames,
    nargout,
    lowered.localVars,
    lowered.hasTensorOps
  );

  // Create function
  let compiledFn: (...args: unknown[]) => unknown;
  const paramNames = fn.params.map(p => p);

  try {
    if (lowered.hasTensorOps) {
      // Pass helpers as first parameter
      const factory = new Function("$h", ...paramNames, jsBody);
      compiledFn = (...callArgs: unknown[]) => factory(jitHelpers, ...callArgs);
    } else {
      const factory = new Function(...paramNames, jsBody);
      compiledFn = (...callArgs: unknown[]) => factory(...callArgs);
    }
  } catch {
    fnWithCache._jitCache.set(cacheKey, null);
    return JIT_SKIP;
  }

  // Cache and log
  const source = `function ${fn.name}(${paramNames.join(", ")}) {\n${jsBody}\n}`;
  fnWithCache._jitCache.set(cacheKey, { fn: compiledFn, source });

  // Fire logging callback
  const typeDesc = argTypes
    .map(t => {
      if (t.kind === "number") return t.nonneg ? "number+" : "number";
      return t.kind;
    })
    .join(", ");
  const description = `${fn.name}(${typeDesc}) -> nargout=${nargout}`;
  interp.onJitCompile?.(description, source);

  // Execute
  try {
    return compiledFn(...args);
  } catch {
    // Runtime error in JIT'd code - remove from cache and fall back
    fnWithCache._jitCache.set(cacheKey, null);
    return JIT_SKIP;
  }
}

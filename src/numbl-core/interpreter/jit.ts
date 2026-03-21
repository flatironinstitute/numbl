/**
 * JIT compilation for scalar-only functions in the interpreter.
 *
 * When optimization >= 1, the interpreter attempts to compile pure scalar
 * functions to native JS before falling back to AST interpretation.
 */

import type { Interpreter } from "./interpreter.js";
import type { FunctionDef, JitCacheEntry } from "./types.js";
import { isScalarCompilable, generateScalarJS } from "./jitCodegen.js";

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

/**
 * User-function call — phased pipeline.
 *
 * Mirrors `jitTopLevel.ts` and `jitLoop.ts`: classify (cheap) → lower
 * (IR) → generate-JS (per-executor codegen) → run. The dispatcher's
 * `tryLowerCall` runs the first two phases; the JS-JIT call executor
 * consumes the lowered IR for the latter two.
 *
 *   1. `classifyCall`  — cheap classification + cacheKey synthesis.
 *      Type-infer args (drop `exact` for numeric scalars), type
 *      widening using fn._lastJitArgTypes, cacheKey from
 *      (nargout, argType signature). Returns null on type-unknown
 *      args or `~` placeholder params.
 *
 *   2. `lowerCall`     — IR lowering. Calls `lowerFunction`. Returns
 *      null when lowerFunction declines.
 *
 *   3. `generateCallJS` — JS codegen + `new Function`. Tries the e1
 *      scalar whole-fn kernel under `--opt e1` and falls back to the
 *      regular JS body on any hiccup.
 *
 *   4. `runCallCompiled` — push call frame + cleanup scope, invoke
 *      compiled fn, pop frame. Translates JitBailToInterpreter into
 *      a transient bail.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { FunctionDef } from "../../interpreter/types.js";
import {
  type JitType,
  computeJitCacheKey,
  jitTypeKey,
  unifyJitTypes,
} from "../../jit/jitTypes.js";
import { lowerFunction, type LoweringResult } from "../../jit/jitLower.js";
import { generateJS } from "./js/jitCodegen.js";
import { jitHelpers, JitBailToInterpreter } from "./js/jitHelpers.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import { tryEmitScalarFnKernel } from "./e1/scalarFnKernel.js";

/** Augmented FunctionDef carrying the per-FunctionDef type-widening
 *  state. Ports the old `_lastJitArgTypes` field; the registry's
 *  lowering cache subsumes the old `_jitCache`. */
interface FunctionDefWithCache extends FunctionDef {
  _lastJitArgTypes?: Map<number, JitType[]>;
}

/** Cheap pre-lowering data. Produced by `classifyCall`. */
export interface CallClassification {
  readonly fn: FunctionDef;
  readonly nargout: number;
  readonly argTypes: readonly JitType[];
  /** Stable string key derived from (nargout, argType signature).
   *  The dispatcher's lowering cache and the executor's compile
   *  cache both key on this. */
  readonly cacheKey: string;
}

/** Lowered IR plus the classification it came from. */
export interface CallLowered {
  readonly classification: CallClassification;
  readonly result: LoweringResult;
}

/** A compiled JS artifact ready to run. */
export interface CallCompiled {
  readonly fn: (...args: unknown[]) => unknown;
  readonly source: string;
}

/**
 * Phase 1 — cheap classification. Returns null when:
 *   - The call has a `~` placeholder param (lowerFunction emits it
 *     as `v_~`, not a valid JS identifier).
 *   - Any arg has type-unknown.
 *
 * Mutates `fn._lastJitArgTypes` for progressive type widening.
 */
export function classifyCall(
  fn: FunctionDef,
  args: unknown[],
  nargout: number
): CallClassification | null {
  // Lower-level constraint: `~` params don't survive JS codegen.
  for (const p of fn.params) {
    if (p === "~") return null;
  }

  // Determine argument types. Drop `exact` for numeric scalars: it
  // only survives unification when two consecutive calls pass the
  // *same* literal (almost never, since callers usually pass
  // variables). Stripping means the first call's cacheKey already
  // matches subsequent calls — one warmup suffices.
  const argTypes: JitType[] = [];
  for (const arg of args) {
    const t = inferJitType(arg);
    if (t.kind === "unknown") return null;
    if (t.kind === "number" && t.exact !== undefined) {
      const pruned: JitType = { kind: "number" };
      if (t.sign !== undefined) pruned.sign = t.sign;
      if (t.isInteger) pruned.isInteger = true;
      argTypes.push(pruned);
    } else {
      argTypes.push(t);
    }
  }

  // Progressive type widening: unify with previously seen types so
  // a function called from an interpreted loop with shifting types
  // converges to a single specialization.
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

  return { fn, nargout, argTypes, cacheKey };
}

/**
 * Phase 2 — IR lowering. Returns null when `lowerFunction` declines
 * (constructs the JS-JIT IR doesn't model).
 */
export function lowerCall(
  interp: Interpreter,
  classification: CallClassification
): CallLowered | null {
  const { fn, nargout, argTypes } = classification;
  const result = lowerFunction(fn, [...argTypes], nargout, interp);
  if (!result) return null;
  return { classification, result };
}

/**
 * Phase 3 — JS codegen. Tries the e1 whole-fn scalar kernel first
 * under `--opt e1`; falls back to the regular JS-JIT body. Returns
 * null when both paths fail.
 */
export function generateCallJS(
  interp: Interpreter,
  lowered: CallLowered
): CallCompiled | null {
  const { classification, result } = lowered;
  const { fn, nargout, argTypes } = classification;

  // ── e1 scalar whole-function kernel path ─────────────────────────────
  // Under --opt e1, if the entire function is pure-scalar, emit a JS
  // wrapper that compiles the whole body to C on first call and
  // dispatches through `$h.compileKernel`. The C source is visible
  // inline in --dump-js output.
  if (interp.experimental === "e1") {
    const scalarKernel = tryEmitScalarFnKernel(
      interp,
      fn,
      result.body,
      result.outputNames,
      result.localVars,
      result.outputType,
      result.outputTypes,
      [...argTypes],
      nargout,
      result.generatedIRBodies
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
        const outputComments = result.outputNames
          .map(
            o =>
              `${o}: ${result.outputType ? jitTypeKey(result.outputType) : "unknown"}`
          )
          .join(", ");
        const source =
          `// JIT (e1 scalar kernel): ${fn.name}(${paramComments}) -> (${outputComments})\n` +
          `// from: ${interp.currentFile}\n` +
          scalarKernel.jsSource;
        const line = interp.rt.$line ?? 0;
        const description = `${fn.name}@${line}(${typeDesc}) -> e1-scalar-kernel`;
        interp.onJitCompile?.(description, source);
        return { fn: compiled, source };
      } catch {
        // Fall through to the regular JS-JIT path.
      }
    }
  }

  // Generate JavaScript for the main function body.
  const currentFile = interp.currentFile;
  const mainBody = generateJS(
    result.body,
    fn.params,
    result.outputNames,
    nargout,
    result.localVars,
    currentFile,
    interp.experimental,
    interp.par
  );

  // Prepend generated helper function definitions (indented to match main body)
  const parts: string[] = [];
  for (const [, code] of result.generatedFns) {
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
    return null;
  }

  // Cache and log
  const typeDesc = argTypes.map(jitTypeKey).join(", ");
  const paramComments = fn.params
    .map((p, i) => `${p}: ${jitTypeKey(argTypes[i])}`)
    .join(", ");
  const outputComments = result.outputNames
    .map(
      o =>
        `${o}: ${result.outputType ? jitTypeKey(result.outputType) : "unknown"}`
    )
    .join(", ");
  const fnComment = [
    `// JIT: ${fn.name}(${paramComments}) -> (${outputComments})`,
    `// from: ${interp.currentFile}`,
  ].join("\n");
  const source = `${fnComment}\nfunction ${fn.name}(${paramNames.join(", ")}) {\n${jsBody}\n}`;

  const line = interp.rt.$line ?? 0;
  const description = `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`;
  interp.onJitCompile?.(description, source);

  return { fn: compiledFn, source };
}

/** Outcome of `runCallCompiled`. `transient` distinguishes the
 *  recoverable JitBailToInterpreter case (cache preserved) from a
 *  hard failure (cache invalidated). */
export type CallRunOutcome =
  | { ok: true; result: unknown }
  | { ok: false; transient: boolean };

/**
 * Phase 4 — execute the compiled fn with proper call frame tracking.
 * If the JIT body throws JitBailToInterpreter (e.g. a scalar index
 * write that needs tensor growth), returns transient bail so the
 * caller falls back to the AST interpreter.
 */
export function runCallCompiled(
  interp: Interpreter,
  fn: FunctionDef,
  compiled: CallCompiled,
  args: unknown[]
): CallRunOutcome {
  interp.rt.pushCallFrame(fn.name);
  interp.rt.pushCleanupScope();
  try {
    const result = compiled.fn(...args);
    return { ok: true, result };
  } catch (e) {
    if (e instanceof JitBailToInterpreter) {
      return { ok: false, transient: true };
    }
    interp.rt.annotateError(e);
    throw e;
  } finally {
    interp.rt.popAndRunCleanups(cf => {
      if (cf.jsFn) {
        if (cf.jsFnExpectsNargout) cf.jsFn(0);
        else cf.jsFn();
      } else {
        interp.rt.dispatch(cf.name, 0, []);
      }
    });
    interp.rt.popCallFrame();
  }
}

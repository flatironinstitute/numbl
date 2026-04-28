/**
 * User-function call — phased pipeline.
 *
 *   1. `classifyCall`  — cheap classification + cacheKey synthesis.
 *      Type-infer args, prune `exact` for numeric scalars, type
 *      widening using fn._lastJitArgTypes. Returns null on
 *      type-unknown args or `~` placeholder params.
 *
 *   2. `lowerCall`     — IR lowering.
 *
 *   3. `generateCallJS` — JS codegen + `new Function`. Tries the e1
 *      whole-fn scalar kernel under `--opt e1` first, falls back to
 *      the regular JS body.
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
} from "../../jit/jitTypes.js";
import { lowerFunction, type LoweringResult } from "../../jit/jitLower.js";
import { generateJS } from "./js/jitCodegen.js";
import { jitHelpers, JitBailToInterpreter } from "./js/jitHelpers.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import { tryEmitScalarFnKernel } from "./e1/scalarFnKernel.js";
import {
  assembleJsBody,
  instantiateJsFn,
  pruneArgType,
  widenAgainst,
} from "./shared.js";

/** Augmented FunctionDef carrying the per-FunctionDef type-widening
 *  state. The registry's lowering cache subsumes the old per-fn
 *  `_jitCache`. */
interface FunctionDefWithCache extends FunctionDef {
  _lastJitArgTypes?: Map<number, JitType[]>;
}

/** Cheap pre-lowering data. */
export interface CallClassification {
  readonly fn: FunctionDef;
  readonly nargout: number;
  readonly argTypes: readonly JitType[];
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

export function classifyCall(
  fn: FunctionDef,
  args: unknown[],
  nargout: number
): CallClassification | null {
  // `~` params don't survive JS codegen (lowerFunction emits them as
  // `v_~`, not a valid identifier).
  for (const p of fn.params) {
    if (p === "~") return null;
  }

  const argTypes: JitType[] = [];
  for (const arg of args) {
    const t = inferJitType(arg);
    if (t.kind === "unknown") return null;
    argTypes.push(pruneArgType(t));
  }

  // Progressive type widening: unify with previously seen types so a
  // function called from an interpreted loop with shifting types
  // converges to a single specialization.
  const fnWithCache = fn as FunctionDefWithCache;
  if (!fnWithCache._lastJitArgTypes) {
    fnWithCache._lastJitArgTypes = new Map();
  }
  widenAgainst(argTypes, fnWithCache._lastJitArgTypes.get(nargout));
  fnWithCache._lastJitArgTypes.set(nargout, argTypes.slice());

  const cacheKey = computeJitCacheKey(nargout, argTypes);

  return { fn, nargout, argTypes, cacheKey };
}

export function lowerCall(
  interp: Interpreter,
  classification: CallClassification
): CallLowered | null {
  const { fn, nargout, argTypes } = classification;
  const result = lowerFunction(fn, [...argTypes], nargout, interp);
  if (!result) return null;
  return { classification, result };
}

export function generateCallJS(
  interp: Interpreter,
  lowered: CallLowered
): CallCompiled | null {
  const { classification, result } = lowered;
  const { fn, nargout, argTypes } = classification;

  // Description / source comments shared by both codegen paths.
  const paramComments = fn.params
    .map((p, i) => `${p}: ${jitTypeKey(argTypes[i])}`)
    .join(", ");
  const outputComments = result.outputNames
    .map(
      o =>
        `${o}: ${result.outputType ? jitTypeKey(result.outputType) : "unknown"}`
    )
    .join(", ");
  const typeDesc = argTypes.map(jitTypeKey).join(", ");
  const line = interp.rt.$line ?? 0;

  // ── e1 scalar whole-function kernel path ─────────────────────────────
  // Under --opt e1, if the entire function is pure-scalar, emit a JS
  // wrapper that compiles the body to C on first call and dispatches
  // through `$h.compileKernel`.
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
      try {
        const factory = new Function(
          "$h",
          "$rt",
          `${scalarKernel.jsSource}\nreturn ${fn.name};`
        );
        const helpers = interp.rt.jitHelpers ?? jitHelpers;
        const wrapped = factory(helpers, interp.rt) as (
          ...a: unknown[]
        ) => unknown;
        const compiled = (...callArgs: unknown[]) => wrapped(...callArgs);
        const source =
          `// JIT (e1 scalar kernel): ${fn.name}(${paramComments}) -> (${outputComments})\n` +
          `// from: ${interp.currentFile}\n` +
          scalarKernel.jsSource;
        interp.onJitCompile?.(
          `${fn.name}@${line}(${typeDesc}) -> e1-scalar-kernel`,
          source
        );
        return { fn: compiled, source };
      } catch {
        // Fall through to the regular JS-JIT path.
      }
    }
  }

  // Regular JS-JIT path.
  const mainBody = generateJS(
    result.body,
    fn.params,
    result.outputNames,
    nargout,
    result.localVars,
    interp.currentFile,
    interp.experimental,
    interp.par
  );
  const jsBody = assembleJsBody(result, mainBody);

  const compiledFn = instantiateJsFn(interp.rt, fn.params, jsBody);
  if (!compiledFn) return null;

  const fnComment = [
    `// JIT: ${fn.name}(${paramComments}) -> (${outputComments})`,
    `// from: ${interp.currentFile}`,
  ].join("\n");
  const source = `${fnComment}\nfunction ${fn.name}(${fn.params.join(", ")}) {\n${jsBody}\n}`;
  interp.onJitCompile?.(
    `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`,
    source
  );

  return { fn: compiledFn, source };
}

export type CallRunOutcome =
  | { ok: true; result: unknown }
  | { ok: false; transient: boolean };

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

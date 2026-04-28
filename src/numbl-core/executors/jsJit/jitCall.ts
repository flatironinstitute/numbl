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
 *   3. `generateCallJS` — JS codegen + `new Function`.
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
} from "../../jitTypes.js";
import { lowerFunction, type LoweringResult } from "./lower/jitLower.js";
import { generateJS } from "./codegen/jitCodegen.js";
import { JitBailToInterpreter } from "./helpers/jitHelpers.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import {
  assembleJsBody,
  instantiateJsFn,
  pruneArgType,
  widenAgainst,
} from "./shared.js";

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
  nargout: number,
  prevArgTypes: readonly JitType[] | undefined
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

  // Progressive type widening so a function called from an
  // interpreted loop with shifting types converges to a single
  // specialization. The dispatcher records the unified result in the
  // lowering cache.
  widenAgainst(argTypes, prevArgTypes);

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

  const mainBody = generateJS(
    result.body,
    fn.params,
    result.outputNames,
    nargout,
    result.localVars,
    interp.currentFile
  );
  const jsBody = assembleJsBody(result, mainBody);

  const compiledFn = instantiateJsFn(interp.rt, fn.params, jsBody);
  if (!compiledFn) return null;

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
  const source = `${fnComment}\nfunction ${fn.name}(${fn.params.join(", ")}) {\n${jsBody}\n}`;
  const typeDesc = argTypes.map(jitTypeKey).join(", ");
  const line = interp.rt.$line ?? 0;
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

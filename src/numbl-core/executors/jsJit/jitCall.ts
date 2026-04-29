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
  buildJitSourceComment,
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
  /**
   * Effective JS parameter names for this specialization. Mirrors
   * `fn.params` for non-varargin functions. For varargin functions,
   * the trailing `varargin` is replaced with one synthetic name per
   * variadic arg (`$va_0`, `$va_1`, …) so each variadic arg becomes
   * a regular JS function parameter. argTypes is one-to-one with
   * effectiveParams.
   */
  readonly effectiveParams: readonly string[];
  /** Number of variadic args (0 when fn has no varargin). */
  readonly nVarargin: number;
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

  // Detect varargin: when the trailing param is `varargin`, the call
  // shape allows additional trailing args. We expand the params list
  // (`[t, varargin]` → `[t, $va_0, $va_1]` for a 3-arg call) so the
  // rest of the JIT pipeline can treat each variadic arg as a regular
  // JS parameter — no per-call cell allocation, and `varargin{i}` /
  // `nargin` resolve via the expanded shape.
  const hasVarargin =
    fn.params.length > 0 && fn.params[fn.params.length - 1] === "varargin";
  const regularParamCount = hasVarargin
    ? fn.params.length - 1
    : fn.params.length;

  if (hasVarargin) {
    if (args.length < regularParamCount) return null;
  } else {
    if (args.length !== fn.params.length) return null;
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
  // lowering cache. widenAgainst no-ops on length mismatch, so each
  // varargin shape (e.g. 2-arg vs 3-arg call) gets its own widening
  // history naturally.
  widenAgainst(argTypes, prevArgTypes);

  const nVarargin = hasVarargin ? args.length - regularParamCount : 0;
  const effectiveParams: string[] = hasVarargin
    ? [
        ...fn.params.slice(0, regularParamCount),
        ...Array.from({ length: nVarargin }, (_, k) => `$va_${k}`),
      ]
    : fn.params.slice();

  const cacheKey = computeJitCacheKey(nargout, argTypes);

  return { fn, nargout, argTypes, cacheKey, effectiveParams, nVarargin };
}

export function lowerCall(
  interp: Interpreter,
  classification: CallClassification
): CallLowered | null {
  const { fn, nargout, argTypes, effectiveParams, nVarargin } = classification;
  const result = lowerFunction(
    fn,
    [...argTypes],
    nargout,
    interp,
    undefined,
    undefined,
    undefined,
    { effectiveParams: [...effectiveParams], nVarargin }
  );
  if (!result) return null;
  return { classification, result };
}

export function generateCallJS(
  interp: Interpreter,
  lowered: CallLowered
): CallCompiled | null {
  const { classification, result } = lowered;
  const { fn, nargout, argTypes, effectiveParams } = classification;

  const mainBody = generateJS(
    result.body,
    [...effectiveParams],
    result.outputNames,
    nargout,
    result.localVars,
    interp.currentFile
  );
  const jsBody = assembleJsBody(result, mainBody);

  const compiledFn = instantiateJsFn(interp.rt, effectiveParams, jsBody);
  if (!compiledFn) return null;

  const paramComments = effectiveParams
    .map((p, i) => `${p}: ${jitTypeKey(argTypes[i])}`)
    .join(", ");
  const outputComments = result.outputNames
    .map(
      o =>
        `${o}: ${result.outputType ? jitTypeKey(result.outputType) : "unknown"}`
    )
    .join(", ");
  const bodyStmts = fn.body;
  const definedIn = bodyStmts[0]?.span?.file ?? interp.currentFile;
  const sourceComment =
    bodyStmts.length > 0 && bodyStmts[0].span
      ? buildJitSourceComment(
          interp,
          bodyStmts[0].span.file,
          bodyStmts[0].span.start,
          bodyStmts[bodyStmts.length - 1].span.end
        ) + "\n"
      : "";
  const fnComment = [
    `// JIT: ${fn.name}(${paramComments}) -> (${outputComments})`,
    `// from: ${definedIn}`,
  ].join("\n");
  const source = `${sourceComment}${fnComment}\nfunction ${fn.name}(${effectiveParams.join(", ")}) {\n${jsBody}\n}`;
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

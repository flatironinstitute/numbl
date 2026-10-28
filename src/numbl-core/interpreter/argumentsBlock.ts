/**
 * Function `arguments` blocks: defaults and name-value struct parameters.
 */

import type { Expr } from "../parser/types.js";
import { isRuntimeChar, isRuntimeString } from "../runtime/types.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { toString } from "../runtime/convert.js";

import type { FunctionDef } from "./types.js";
import type { Interpreter } from "./interpreter.js";
import { evalExpr } from "./evalExpressions.js";

/** Number of leading parameters among `regularParams` (the parameters
 *  before `varargin`) that are bound positionally. Name-value struct
 *  parameters declared in an `arguments` block (entries like `opts.method`)
 *  are filled from trailing arguments by `processArgumentsBlocks` instead,
 *  so positional binding stops before them. */
export function positionalParamCount(
  fn: FunctionDef,
  regularParams: readonly string[]
): number {
  let count = regularParams.length;
  for (const block of fn.argumentsBlocks ?? []) {
    if (block.kind === "Output") continue;
    for (const e of block.entries) {
      const dot = e.name.indexOf(".");
      if (dot < 0) continue;
      const idx = fn.params.indexOf(e.name.slice(0, dot));
      if (idx >= 0 && idx < count) count = idx;
    }
  }
  return count;
}

/** Field names declared by dotted (name-value) entries across a function's
 *  input arguments blocks. */
export function nameValueFieldNames(fn: FunctionDef): Set<string> {
  const names = new Set<string>();
  for (const block of fn.argumentsBlocks ?? []) {
    if (block.kind === "Output" || block.kind === "OutputRepeating") continue;
    for (const e of block.entries) {
      const dot = e.name.indexOf(".");
      if (dot >= 0) names.add(e.name.slice(dot + 1));
    }
  }
  return names;
}

/** Index into `extras` where the trailing name-value section starts: the
 *  first char/string argument matching a declared option field name.
 *  Everything before it belongs to the repeating (varargin) section. */
export function nameValueTailStart(
  extras: unknown[],
  fieldNames: Set<string>
): number {
  if (fieldNames.size === 0) return extras.length;
  for (let i = 0; i < extras.length; i++) {
    if (extras[i] === undefined) continue;
    const rv = ensureRuntimeValue(extras[i]);
    if (
      (isRuntimeChar(rv) || isRuntimeString(rv)) &&
      fieldNames.has(toString(rv))
    ) {
      return i;
    }
  }
  return extras.length;
}

/**
 * Apply `arguments`-block defaults and build name-value struct parameters,
 * binding the results into the running frame's environment.
 *
 * Must be called with the function's frame running and the caller-supplied
 * positional arguments already bound to their parameters. Default
 * expressions are evaluated in that scope so they can reference earlier
 * arguments (e.g. `mergeIdx = surfaceop.defaultIdx(dom)`), matching MATLAB.
 * Entries are matched to parameters by name via `fn.params`, which also
 * handles the constructor case where the output variable is prepended to
 * `params` (shifting argument positions by one).
 */
export function processArgumentsBlocks(
  interp: Interpreter,
  fn: FunctionDef,
  args: unknown[]
): void {
  const argBlocks = fn.argumentsBlocks;
  if (!argBlocks || argBlocks.length === 0) return;
  const env = interp.frame.env;

  for (const block of argBlocks) {
    if (block.kind === "Output") continue;
    const entries = block.entries;
    if (!entries || entries.length === 0) continue;

    // Split dotted entries (name-value struct fields) from positional ones.
    const nvGroups = new Map<
      string,
      { field: string; defaultExpr: Expr | null }[]
    >();
    const regularEntries = entries.filter(e => {
      const dotIdx = e.name.indexOf(".");
      if (dotIdx >= 0) {
        const paramName = e.name.slice(0, dotIdx);
        const field = e.name.slice(dotIdx + 1);
        let group = nvGroups.get(paramName);
        if (!group) {
          group = [];
          nvGroups.set(paramName, group);
        }
        group.push({ field, defaultExpr: e.defaultValue });
        return false;
      }
      return true;
    });

    // Fill any unsupplied positional argument from its default expression, in
    // declaration order, binding each into the scope so later defaults can
    // see it.
    for (const entry of regularEntries) {
      const pIdx = fn.params.indexOf(entry.name);
      if (pIdx < 0) continue;
      const provided = pIdx < args.length && args[pIdx] !== undefined;
      if (!provided && entry.defaultValue) {
        env.set(
          entry.name,
          ensureRuntimeValue(evalExpr(interp, entry.defaultValue))
        );
      }
    }

    // Assemble each name-value struct parameter from the trailing arguments
    // (those past the struct parameter's position) plus field defaults.
    const vararginIdx = fn.params.indexOf("varargin");
    for (const [paramName, fields] of nvGroups) {
      const pIdx = fn.params.indexOf(paramName);
      let nvArgs: unknown[];
      if (pIdx >= 0 && vararginIdx >= 0 && vararginIdx < pIdx) {
        // varargin consumes a variable number of arguments; the name-value
        // tail starts at the first declared option name after it.
        const extras = args.slice(vararginIdx);
        nvArgs = extras.slice(
          nameValueTailStart(extras, nameValueFieldNames(fn))
        );
      } else {
        nvArgs = args.slice(pIdx >= 0 ? pIdx : args.length);
      }
      const defaults: Record<string, unknown> = {};
      for (const { field, defaultExpr } of fields) {
        if (defaultExpr) defaults[field] = evalExpr(interp, defaultExpr);
      }
      env.set(paramName, interp.rt.buildNameValueStruct(nvArgs, defaults));
    }
  }
}

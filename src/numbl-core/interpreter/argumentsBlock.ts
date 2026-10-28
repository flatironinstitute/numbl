/**
 * Function `arguments` blocks: binding and validating inputs (positional,
 * repeating and name-value arguments, defaults) and validating outputs.
 *
 * A function's blocks are compiled once into an `ArgsSpec` (cached per
 * AST); each call then only walks the compiled entries.
 */

import type { ArgumentsBlock, Expr } from "../parser/types.js";
import {
  isRuntimeChar,
  isRuntimeString,
  type RuntimeValue,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { valueShape } from "../runtime/kinds.js";

import type { Environment, FunctionDef } from "./types.js";
import type { Interpreter } from "./interpreter.js";
import { evalExpr } from "./evalExpressions.js";
import {
  compileCheck,
  validateValue,
  type CompiledCheck,
} from "./valueValidation.js";

interface InputArg {
  readonly name: string;
  readonly check: CompiledCheck | null;
  readonly defaultValue: Expr | null;
}

interface NameValueField {
  readonly field: string;
  readonly check: CompiledCheck | null;
  readonly defaultValue: Expr | null;
}

interface NameValueGroup {
  /** The struct parameter (`opts` in `opts.Name`). */
  readonly param: string;
  readonly fields: readonly NameValueField[];
}

/** A function's compiled `arguments` blocks. Parameter indices count all
 *  of `fn.params`, including a constructor's leading instance parameter. */
export interface ArgsSpec {
  /** The parameters bound by position, in order. */
  readonly positional: readonly InputArg[];
  /** One past the last positional parameter declared without a default. */
  readonly numRequired: number;
  /** Repeating arguments (one group), or null. A `varargin` parameter
   *  outside a `Repeating` block counts as an unchecked repeating group. */
  readonly repeating: readonly InputArg[] | null;
  readonly nameValue: readonly NameValueGroup[];
  readonly outputs: ReadonlyMap<string, CompiledCheck> | null;
  /** A call with every argument supplied needs no processing, so a JIT
   *  executor may run it on the raw arguments. */
  readonly jitDirect: boolean;
  /** Only positional checks: after validation, a JIT executor may run the
   *  call on the converted arguments. */
  readonly jitAfterValidation: boolean;
}

const specCache = new WeakMap<ArgumentsBlock[], Map<string, ArgsSpec>>();

/** The compiled `arguments` blocks of `fn`, or null when it has none. */
export function argumentsSpec(fn: FunctionDef): ArgsSpec | null {
  const blocks = fn.argumentsBlocks;
  if (!blocks || blocks.length === 0) return null;
  let byParams = specCache.get(blocks);
  if (!byParams) {
    byParams = new Map();
    specCache.set(blocks, byParams);
  }
  // A constructor's FunctionDef shares the method's blocks but prepends
  // the instance parameter, so key on the parameter list as well.
  const key = fn.params.join(",");
  let spec = byParams.get(key);
  if (!spec) {
    spec = compileSpec(fn.params, blocks);
    byParams.set(key, spec);
  }
  return spec;
}

function compileSpec(params: string[], blocks: ArgumentsBlock[]): ArgsSpec {
  const declared = new Map<string, InputArg>();
  const repeatingNames: string[] = [];
  const nvGroups = new Map<string, NameValueField[]>();
  let outputs: Map<string, CompiledCheck> | null = null;

  for (const block of blocks) {
    const isOutput =
      block.kind === "Output" || block.kind === "OutputRepeating";
    for (const e of block.entries) {
      const check = compileCheck(e);
      if (isOutput) {
        if (check) (outputs ??= new Map()).set(e.name, check);
        continue;
      }
      const dot = e.name.indexOf(".");
      if (dot >= 0) {
        const param = e.name.slice(0, dot);
        let group = nvGroups.get(param);
        if (!group) nvGroups.set(param, (group = []));
        group.push({
          field: e.name.slice(dot + 1),
          check,
          defaultValue: e.defaultValue,
        });
        continue;
      }
      declared.set(e.name, {
        name: e.name,
        check,
        defaultValue: e.defaultValue,
      });
      if (block.kind === "Repeating") repeatingNames.push(e.name);
    }
  }

  const positional: InputArg[] = [];
  let numRequired = 0;
  for (const p of params) {
    if (nvGroups.has(p) || repeatingNames.includes(p) || p === "varargin")
      break;
    const arg = declared.get(p) ?? { name: p, check: null, defaultValue: null };
    positional.push(arg);
    if (declared.has(p) && !arg.defaultValue) numRequired = positional.length;
  }

  let repeating: InputArg[] | null = null;
  if (repeatingNames.length > 0) {
    repeating = repeatingNames.map(n => declared.get(n)!);
  } else if (params.includes("varargin")) {
    repeating = [{ name: "varargin", check: null, defaultValue: null }];
  }

  const nameValue: NameValueGroup[] = [];
  for (const p of params) {
    const fields = nvGroups.get(p);
    if (fields) nameValue.push({ param: p, fields });
  }

  const positionalChecks = positional.some(a => a.check);
  const plainRepeating =
    repeating === null ||
    (repeating.length === 1 &&
      repeating[0].name === "varargin" &&
      !repeating[0].check);
  const simple = nameValue.length === 0 && outputs === null;
  return {
    positional,
    numRequired,
    repeating,
    nameValue,
    outputs,
    jitDirect: simple && plainRepeating && !positionalChecks,
    jitAfterValidation: simple && repeating === null && positionalChecks,
  };
}

function argError(id: string, message: string): RuntimeError {
  return new RuntimeError(message, null, { identifier: id });
}

/** A char row vector or string scalar. */
function textScalar(v: unknown): string | null {
  if (v === undefined) return null;
  const rv = ensureRuntimeValue(v);
  if (isRuntimeString(rv)) return rv;
  if (isRuntimeChar(rv)) {
    const shape = valueShape(rv);
    if (shape.length === 2 && shape[0] <= 1) return rv.value;
  }
  return null;
}

/** Quote and join names the way MATLAB lists choices: 'a', 'b', or 'c'. */
function listNames(names: string[]): string {
  const q = names.map(n => `'${n}'`);
  if (q.length <= 1) return q.join("");
  if (q.length === 2) return `${q[0]} or ${q[1]}`;
  return `${q.slice(0, -1).join(", ")}, or ${q[q.length - 1]}`;
}

/** Match a name-value argument name: case-insensitive, exact or a unique
 *  prefix. */
function matchName(
  spec: ArgsSpec,
  text: string
): { group: NameValueGroup; field: NameValueField } | "ambiguous" | null {
  const lower = text.toLowerCase();
  const prefixHits: { group: NameValueGroup; field: NameValueField }[] = [];
  for (const group of spec.nameValue) {
    for (const field of group.fields) {
      const f = field.field.toLowerCase();
      if (f === lower) return { group, field };
      if (lower.length > 0 && f.startsWith(lower))
        prefixHits.push({ group, field });
    }
  }
  if (prefixHits.length === 0) return null;
  return prefixHits.length === 1 ? prefixHits[0] : "ambiguous";
}

function allNames(spec: ArgsSpec): string[] {
  return spec.nameValue.flatMap(g => g.fields.map(f => f.field));
}

/** Index in `args` where the name-value section starts: the first
 *  position (past the required arguments) holding a declared name with an
 *  even number of arguments from there on. Without such a position, extra
 *  arguments past the positional parameters are parsed as name-value
 *  arguments anyway, so their errors are reported. */
function nameValueStart(spec: ArgsSpec, args: unknown[]): number {
  const n = args.length;
  const P = spec.positional.length;
  const lo = spec.repeating ? P : spec.numRequired;
  const hi = spec.repeating ? n : Math.min(P, n);
  for (let s = lo; s <= hi && s < n; s++) {
    if ((n - s) % 2 !== 0) continue;
    const text = textScalar(args[s]);
    if (text !== null && matchName(spec, text) !== null) return s;
  }
  return !spec.repeating && n > P ? P : n;
}

/**
 * Bind and validate a call's inputs per the function's `arguments` blocks,
 * in the running (callee) frame. `args` includes any leading instance
 * argument of a constructor; `posOffset` is the number of such arguments,
 * which do not count toward reported argument positions.
 *
 * Defaults are evaluated only for missing arguments, in declaration order,
 * in the callee's scope, so they can reference earlier arguments.
 */
export function bindArguments(
  interp: Interpreter,
  spec: ArgsSpec,
  args: unknown[],
  posOffset: number
): void {
  const env = interp.frame.env;
  const n = args.length;
  const P = spec.positional.length;

  // ── Parse the argument list ──
  if (n < spec.numRequired) {
    throw argError(
      "MATLAB:minrhs",
      `Invalid argument list. Function requires ${spec.numRequired - n} more input(s).`
    );
  }
  let nvStart = n;
  let provided: Map<NameValueField, unknown> | null = null;
  if (spec.nameValue.length > 0) {
    nvStart = nameValueStart(spec, args);
    provided = parseNameValueArgs(spec, args, nvStart, posOffset);
  } else if (!spec.repeating && n > P) {
    throw argError("MATLAB:TooManyInputs", "Too many input arguments.");
  }

  // ── Positional arguments ──
  const supplied = Math.min(nvStart, n);
  for (let i = 0; i < P; i++) {
    const a = spec.positional[i];
    if (i < supplied && args[i] !== undefined) {
      let v = ensureRuntimeValue(args[i]);
      if (a.check)
        v = validateValue(
          interp,
          a.check,
          v,
          () => `Invalid argument at position ${i - posOffset + 1}. `,
          x => env.set(a.name, x)
        );
      env.set(a.name, v);
    } else if (a.defaultValue) {
      let v = ensureRuntimeValue(evalExpr(interp, a.defaultValue));
      if (a.check)
        v = validateValue(
          interp,
          a.check,
          v,
          () => `Invalid default value for argument '${a.name}'. `,
          x => env.set(a.name, x)
        );
      env.set(a.name, v);
    }
  }

  // ── Repeating arguments ──
  if (spec.repeating)
    bindRepeating(interp, spec.repeating, args, P, nvStart, posOffset);

  // ── Name-value arguments ──
  if (provided) {
    for (const group of spec.nameValue)
      bindNameValueGroup(interp, env, group, provided);
  }
}

/** Parse the name-value pairs `args[start..]` into the declared fields
 *  they name, raising MATLAB's errors for malformed lists and unknown or
 *  ambiguous names. */
function parseNameValueArgs(
  spec: ArgsSpec,
  args: unknown[],
  start: number,
  posOffset: number
): Map<NameValueField, unknown> {
  const n = args.length;
  const P = spec.positional.length;
  const provided = new Map<NameValueField, unknown>();
  for (let j = start; j < n; j += 2) {
    const position = j - posOffset + 1;
    const text = textScalar(args[j]);
    if (text === null) {
      const max = P - posOffset;
      const min = Math.max(0, spec.numRequired - posOffset);
      const count = min >= max ? `exactly ${max}` : `${min} to ${max}`;
      throw argError(
        "MATLAB:TooManyInputs",
        `Invalid argument at position ${position}. Function requires ${count} positional input(s).`
      );
    }
    if (j + 1 >= n) {
      throw argError(
        "MATLAB:TooManyInputs",
        `Invalid argument at position ${position}. Make sure name-value arguments include both the name and the value.`
      );
    }
    const m = matchName(spec, text);
    if (m === null) {
      throw argError(
        "MATLAB:TooManyInputs",
        `Invalid argument name '${text}'. Name must be ${listNames(allNames(spec))}.`
      );
    }
    if (m === "ambiguous") {
      const lower = text.toLowerCase();
      const hits = allNames(spec).filter(f =>
        f.toLowerCase().startsWith(lower)
      );
      throw argError(
        "MATLAB:functionValidation:AmbiguousInputName",
        `Ambiguous argument name '${text}'. Name must be ${listNames(hits)}.`
      );
    }
    provided.set(m.field, args[j + 1]);
  }
  return provided;
}

function bindRepeating(
  interp: Interpreter,
  group: readonly InputArg[],
  args: unknown[],
  start: number,
  end: number,
  posOffset: number
): void {
  const env = interp.frame.env;
  const k = group.length;
  const count = Math.max(0, end - start);
  if (count % k !== 0) {
    throw argError(
      "MATLAB:functionValidation:RepeatingInputsNotInGroups",
      `Invalid argument list. Repeating arguments, if specified, must be in groups of ${k}.`
    );
  }
  const groups = count / k;
  const values: RuntimeValue[][] = group.map(() => []);
  for (let g = 0; g < groups; g++) {
    for (let j = 0; j < k; j++) {
      const a = group[j];
      const idx = start + g * k + j;
      let v = ensureRuntimeValue(args[idx]);
      if (a.check)
        v = validateValue(
          interp,
          a.check,
          v,
          () => `Invalid argument at position ${idx - posOffset + 1}. `,
          x => env.set(a.name, x)
        );
      values[j].push(v);
    }
  }
  for (let j = 0; j < k; j++)
    env.set(group[j].name, RTV.cell(values[j], [1, groups]));
}

function bindNameValueGroup(
  interp: Interpreter,
  env: Environment,
  group: NameValueGroup,
  provided: Map<NameValueField, unknown>
): void {
  const fields = new Map<string, RuntimeValue>();
  // Defaults and call-expression validators may read the struct built so
  // far (`opts.lim`), so expose it before evaluating them.
  const expose = () => env.set(group.param, RTV.struct(new Map(fields)));
  for (const f of group.fields) {
    const bind = (v: RuntimeValue) => {
      fields.set(f.field, v);
      expose();
    };
    if (provided.has(f)) {
      let v = ensureRuntimeValue(provided.get(f));
      if (f.check)
        v = validateValue(
          interp,
          f.check,
          v,
          () => `Invalid value for '${f.field}' argument. `,
          bind
        );
      fields.set(f.field, v);
    } else if (f.defaultValue) {
      expose();
      let v = ensureRuntimeValue(evalExpr(interp, f.defaultValue));
      if (f.check)
        v = validateValue(
          interp,
          f.check,
          v,
          () => `Invalid default value for argument '${f.field}'. `,
          bind
        );
      fields.set(f.field, v);
    }
  }
  env.set(group.param, RTV.struct(fields));
}

/** Validate a call's outputs against its `arguments (Output)`
 *  declarations, rebinding converted values. As in MATLAB, every assigned
 *  output is checked, requested or not; unassigned ones are left for the
 *  caller's "not assigned" error. */
export function validateOutputs(interp: Interpreter, spec: ArgsSpec): void {
  const checks = spec.outputs;
  if (!checks) return;
  const env = interp.frame.env;
  for (const [name, check] of checks) {
    const v = env.get(name);
    if (v === undefined) continue;
    env.set(
      name,
      validateValue(
        interp,
        check,
        v,
        () => `Invalid output '${name}'. `,
        x => env.set(name, x)
      )
    );
  }
}

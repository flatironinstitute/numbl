/**
 * `builtin()`: the way to define an interpreter builtin.
 *
 * A builtin is an `apply` function plus explicit arity metadata. `apply`
 * receives runtime values and is responsible for rejecting what it cannot
 * handle (see args.ts). Arity is checked before `apply` runs, with
 * MATLAB's messages and identifiers; then the argument-kind policy
 * (`accepts`, runtime/argKinds.ts) converts or rejects the non-dense kinds
 * (sparse) the builtin does not declare.
 *
 *   builtin({
 *     name: "hypot",
 *     nargin: 2,
 *     apply: args => ...,
 *   });
 *
 * A builtin that needs the calling scope (its variables, caller, input
 * names; eval, exist, nargin, ...) declares `scope: true` and reads it as
 * `ctx.scope`.
 */

import type { Runtime } from "../../runtime/runtime.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { RuntimeError } from "../../runtime/error.js";
import { getCurrentRuntime } from "../../runtime/refcount.js";
import {
  registerIBuiltin,
  registerClassMethod,
  type BuiltinHelp,
  type IBuiltin,
} from "../../runtime/builtinRegistry.js";
import type { CallerScope } from "../callerScope.js";
import {
  applyArgPolicy,
  resolveArgPolicy,
  type ArgKindPolicy,
} from "../../runtime/argKinds.js";
import { checkNargin, checkNargout } from "./args.js";
import { firstGpuArg, outputToGpu } from "../../runtime/gpu/gpuArray.js";
import { gpuLogEnabled, logTransfer } from "../../runtime/gpu/gpuLog.js";
import {
  hasSingleArg,
  isSingle,
  toSingleResult,
} from "../../runtime/single.js";

/** An argument count: exactly `n`, or an inclusive range `[min, max]`
 *  (max may be Infinity). For `nargout`, a single number `n` means up to
 *  `n` outputs (`[0, n]`), since a call may always request none. */
export type Arity = number | readonly [number, number];

/** Per-builtin context passed to `apply`. */
export interface BuiltinContext {
  /** The builtin's name, for error messages. */
  readonly name: string;
  /** The runtime executing the current call (output, file I/O, figures,
   *  search path, ...). */
  readonly rt: Runtime;
}

/** The context of a builtin declared with `scope: true`. */
export interface ScopedBuiltinContext extends BuiltinContext {
  /** The scope the builtin was called from. */
  readonly scope: CallerScope;
}

export type BuiltinApply<C extends BuiltinContext = BuiltinContext> = (
  args: RuntimeValue[],
  nargout: number,
  ctx: C
) => RuntimeValue | RuntimeValue[] | void;

interface BuiltinSpecBase {
  name: string;
  help?: BuiltinHelp;
  /** Accepted input count. Default: any number. */
  nargin?: Arity;
  /** Accepted output count. Default 1 (up to one output); 0 marks a void
   *  builtin, whose result is discarded and which errors when an output
   *  is requested. */
  nargout?: Arity;
  /** Set when MATLAB implements the function as an M-file (`which name`
   *  shows a .m path, e.g. angle, linspace). MATLAB then reports too many
   *  inputs/outputs as MATLAB:TooManyInputs / MATLAB:TooManyOutputs rather
   *  than MATLAB:maxrhs / MATLAB:maxlhs. */
  mfile?: boolean;
  /** How the builtin takes the non-dense value kinds (see
   *  runtime/argKinds.ts). By default a sparse argument is converted to a
   *  dense tensor before `apply` runs; `accepts: { sparse: true }` passes
   *  it through for a builtin that handles sparse itself, and
   *  `sparse: "error"` (or `{ identifier, message }`) rejects it. */
  accepts?: ArgKindPolicy;
  /** The outputs that take a single argument's class (MATLAB: `sin`,
   *  `sum`, `reshape`, the value but not the index of `max`, ...): when an
   *  argument is single, these outputs are converted to single when they
   *  are double (see runtime/single.ts). `true` means every output; a list
   *  gives output positions, counting from 1; a function chooses from the
   *  arguments (`sum(x, 'double')` is double). Other outputs stay as
   *  `apply` returns them (`size`, `find`, an index output). */
  singleOutputs?: SingleOutputs | ((args: RuntimeValue[]) => SingleOutputs);
  /** With a gpuArray argument (gathered to the host under the default
   *  `accepts`), the numeric and logical outputs are returned as gpuArrays
   *  on its device, as MATLAB runs the function on the GPU. Set for the
   *  builtins whose results stay on the host (`size`, `isempty`,
   *  `isequal`, ...). */
  hostOutputs?: boolean;
}

/** Outputs that take a single argument's class: `true` (every output) or
 *  1-based output positions. */
export type SingleOutputs = true | readonly number[];

export interface PlainBuiltinSpec extends BuiltinSpecBase {
  scope?: false;
  /** Called after the arity check. Return one value, or an array of
   *  values when `nargout > 1`. */
  apply: BuiltinApply;
}

export interface ScopedBuiltinSpec extends BuiltinSpecBase {
  /** The builtin needs the calling scope, given to `apply` as
   *  `ctx.scope`. */
  scope: true;
  apply: BuiltinApply<ScopedBuiltinContext>;
}

export type BuiltinSpec = PlainBuiltinSpec | ScopedBuiltinSpec;

function inputRange(a: Arity | undefined): [number, number] {
  if (a === undefined) return [0, Infinity];
  return typeof a === "number" ? [a, a] : [a[0], a[1]];
}

function outputRange(a: Arity | undefined): [number, number] {
  if (a === undefined) return [0, 1];
  return typeof a === "number" ? [0, a] : [a[0], a[1]];
}

function validRange(name: string, what: string, [lo, hi]: number[]): void {
  if (
    !(
      Number.isInteger(lo) &&
      lo >= 0 &&
      (Number.isInteger(hi) || hi === Infinity) &&
      hi >= lo
    )
  )
    throw new Error(`builtin '${name}': invalid ${what} [${lo}, ${hi}]`);
}

function activeRuntime(): Runtime {
  const rt = getCurrentRuntime();
  if (!rt) throw new RuntimeError("no active runtime");
  return rt as Runtime;
}

const NO_OUTPUTS: readonly number[] = [];

/** The results of a call with a single argument: each declared output (see
 *  `singleOutputs`) as single, and every single output rounded (a builtin
 *  that builds its result from `v.cls` may hold values computed in
 *  double). */
function singleResults(
  result: RuntimeValue | RuntimeValue[],
  outputs: SingleOutputs,
  args: RuntimeValue[]
): RuntimeValue | RuntimeValue[] {
  const one = (v: RuntimeValue, pos: number): RuntimeValue =>
    outputs === true || outputs.includes(pos)
      ? toSingleResult(v, args)
      : isSingle(v)
        ? toSingleResult(v, args)
        : v;
  if (!Array.isArray(result))
    return result === undefined ? result : one(result, 1);
  return result.map((v, i) => one(v, i + 1));
}

/** Build the IBuiltin for a spec without registering it. */
export function makeBuiltin(spec: BuiltinSpec): IBuiltin {
  const { name } = spec;
  const mfile = spec.mfile ?? false;
  const nargin = inputRange(spec.nargin);
  const nargout = outputRange(spec.nargout);
  validRange(name, "nargin", nargin);
  validRange(name, "nargout", nargout);
  const [inMin, inMax] = nargin;
  const [outMin, outMax] = nargout;
  const isVoid = outMax === 0;

  const plain: BuiltinContext = {
    name,
    get rt(): Runtime {
      return activeRuntime();
    },
  };
  // Only a builtin declared with `scope: true` gets `ctx.scope`.
  const ctx = spec.scope
    ? Object.defineProperty(plain, "scope", {
        get(): CallerScope {
          const scope = activeRuntime().callerScope?.();
          if (!scope) throw new RuntimeError(`${name}: no calling scope`);
          return scope as CallerScope;
        },
      })
    : plain;
  const apply = spec.apply as BuiltinApply;
  const policy = resolveArgPolicy(name, spec.accepts);
  const singleOutputs = spec.singleOutputs ?? NO_OUTPUTS;
  const uploadsToGpu =
    !spec.hostOutputs && (spec.accepts?.gpu ?? "convert") === "convert";

  return {
    name,
    help: spec.help,
    nargin,
    nargout,
    call: (args, n) => {
      checkNargin(args.length, inMin, inMax, mfile);
      checkNargout(n, outMin, outMax, mfile);
      const cargs = applyArgPolicy(policy, args);
      const t0 = gpuLogEnabled ? performance.now() : 0;
      let result = apply(cargs, n, ctx) as RuntimeValue | RuntimeValue[];
      if (gpuLogEnabled && cargs !== args && firstGpuArg(args))
        logTransfer("host", name, 0, performance.now() - t0);
      if (isVoid) return undefined;
      if (hasSingleArg(cargs))
        result = singleResults(
          result,
          typeof singleOutputs === "function"
            ? singleOutputs(cargs)
            : singleOutputs,
          cargs
        );
      // A gathered gpuArray argument: the outputs go back to its device.
      const gpu = uploadsToGpu && cargs !== args ? firstGpuArg(args) : null;
      if (gpu === null) return result;
      return Array.isArray(result)
        ? result.map(v =>
            v === undefined ? v : outputToGpu(v, gpu.device, false, name)
          )
        : result === undefined
          ? result
          : outputToGpu(result, gpu.device, false, name);
    },
  };
}

/** Define and register a builtin. */
export function builtin(spec: BuiltinSpec): void {
  registerIBuiltin(makeBuiltin(spec));
}

/** Define a method of a TypeScript-implemented class (`spec.name` is the
 *  method name). It is called like a classdef method: by `obj.name(...)`,
 *  by `name(...)` with an argument of the class, and by the operator whose
 *  method name it is. See `registerClassMethod`. */
export function classMethod(className: string, spec: PlainBuiltinSpec): void {
  registerClassMethod(className, makeBuiltin(spec));
}

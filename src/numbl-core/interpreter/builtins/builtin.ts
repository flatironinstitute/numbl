/**
 * `builtin()`: the way to define an interpreter builtin.
 *
 * A builtin is an `apply` function plus explicit arity metadata. There are
 * no argument-type rules: `apply` receives runtime values and is
 * responsible for rejecting what it cannot handle (see args.ts). Arity is
 * checked before `apply` runs, with MATLAB's messages and identifiers.
 *
 *   builtin({
 *     name: "hypot",
 *     nargin: 2,
 *     apply: args => ...,
 *   });
 *
 * `builtin()` produces an ordinary IBuiltin (with a `direct` entry), so all
 * call paths (interpreter, JIT fallback, `builtin(...)`, method dispatch,
 * function handles) work unchanged.
 */

import type { Runtime } from "../../runtime/runtime.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { RuntimeError } from "../../runtime/error.js";
import { getCurrentRuntime } from "../../runtime/refcount.js";
import {
  registerIBuiltin,
  type BuiltinHelp,
  type DirectBuiltin,
  type IBuiltin,
  type IBuiltinResolution,
} from "./types.js";
import { checkNargin, checkNargout } from "./args.js";

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

export type BuiltinApply = (
  args: RuntimeValue[],
  nargout: number,
  ctx: BuiltinContext
) => RuntimeValue | RuntimeValue[] | void;

export interface BuiltinSpec {
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
  /** Called after the arity check. Return one value, or an array of
   *  values when `nargout > 1`. */
  apply: BuiltinApply;
}

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

/** MATLAB's `nargin(@f)` for a builtin: the maximum input count, or -1 when
 *  it is unbounded (MATLAB reports -1 for nearly all variadic builtins,
 *  e.g. zeros, numel, fprintf, cellfun). */
function reportedNargin([, hi]: readonly [number, number]): number {
  return hi === Infinity ? -1 : hi;
}

/** Build the IBuiltin for a spec without registering it. */
export function makeBuiltin(spec: BuiltinSpec): IBuiltin {
  const { name, apply } = spec;
  const mfile = spec.mfile ?? false;
  const nargin = inputRange(spec.nargin);
  const nargout = outputRange(spec.nargout);
  validRange(name, "nargin", nargin);
  validRange(name, "nargout", nargout);
  const [inMin, inMax] = nargin;
  const [outMin, outMax] = nargout;
  const isVoid = outMax === 0;

  const ctx: BuiltinContext = {
    name,
    get rt(): Runtime {
      const rt = getCurrentRuntime();
      if (!rt) throw new RuntimeError("no active runtime");
      return rt as Runtime;
    },
  };

  const call: DirectBuiltin["call"] = (args, n) => {
    checkNargin(args.length, inMin, inMax, mfile);
    checkNargout(n, outMin, outMax, mfile);
    const result = apply(args, n, ctx);
    return isVoid ? undefined : (result as RuntimeValue | RuntimeValue[]);
  };

  // Legacy entry for callers that still go through resolve. It accepts any
  // argument types; `call` does the checking.
  const resolution: IBuiltinResolution = {
    outputTypes: isVoid ? [] : [{ kind: "unknown" }],
    apply: call as IBuiltinResolution["apply"],
  };

  return {
    name,
    help: spec.help,
    resolve: () => resolution,
    direct: { nargin, nargout, reportedNargin: reportedNargin(nargin), call },
  };
}

/** Define and register a builtin. */
export function builtin(spec: BuiltinSpec): void {
  registerIBuiltin(makeBuiltin(spec));
}

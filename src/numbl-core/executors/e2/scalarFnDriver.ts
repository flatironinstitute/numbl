/**
 * e2 — whole-function scalar C-kernel — codegen + run phases.
 *
 * The dispatcher's call-shape lowering pipeline runs the shared
 * classify+lower work via `tryLowerCall`. This module exposes the e2
 * codegen and run phases:
 *
 *   - `compileE2ScalarFn(interp, lowered)` — feasibility check,
 *     `generateC`, `compileFn`. Returns the koffi-bound C function +
 *     output kinds, or null when the body isn't a pure-scalar fn.
 *
 *   - `runE2ScalarFn(args, compiled, nargout)` — coerce args (bool
 *     → 0/1), allocate per-output Float64Array(1) buffers, invoke
 *     the C function, unpack results.
 *
 * The "scalar args" eligibility gate runs in the executor's
 * propose() — see `e2/scalarFnCKernelExecutor.ts`.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { RuntimeError } from "../../runtime/error.js";
import { jitTypeKey } from "../../jit/jitTypes.js";
import { generateC } from "../../jit/c/assemble.js";
import { checkCFeasibility } from "../../jit/c/feasibility.js";
import { getE2CompileFn } from "./compileFn.js";
import type { CallLowered } from "../jsJit/jitCall.js";

export interface E2ScalarFnCompiled {
  readonly fn: (...args: unknown[]) => unknown;
  readonly outputKinds: ("number" | "boolean")[];
}

/**
 * Phase: codegen. Returns null when:
 *   - Any declared output type isn't scalar number/boolean.
 *   - `checkCFeasibility` rejects the body (Index writes, tic/toc, …).
 *   - `generateC` throws.
 *   - The kernel needs runtime infrastructure not provided by the
 *     plain koffi call shape (tic state, error flag, disp callback).
 *
 * Throws RuntimeError if the C compilation step fails (matches the
 * other e2 drivers' "compile failures are hard errors" policy).
 */
export function compileE2ScalarFn(
  interp: Interpreter,
  lowered: CallLowered
): E2ScalarFnCompiled | null {
  const { classification, result } = lowered;
  const { fn, nargout, argTypes } = classification;

  // All declared outputs must be scalar-ish — otherwise the
  // out-buffer convention breaks.
  for (const t of result.outputTypes) {
    if (t.kind !== "number" && t.kind !== "boolean") return null;
  }

  const feas = checkCFeasibility(
    result.body,
    fn.params,
    [...argTypes],
    result.outputType,
    result.outputTypes,
    nargout,
    result.generatedIRBodies
  );
  if (!feas.ok) return null;

  let gen;
  try {
    gen = generateC(
      result.body,
      fn.params,
      result.outputNames,
      nargout,
      result.localVars,
      [...argTypes],
      result.outputType,
      result.outputTypes,
      fn.name.replace(/[^A-Za-z0-9_]/g, "_"),
      false,
      false,
      result.generatedIRBodies
    );
  } catch {
    return null;
  }

  // tic/toc/disp/error-flag would require runtime infrastructure we
  // don't pass to scalar koffi calls — bail rather than crash.
  if (gen.needsTicState || gen.needsErrorFlag || gen.needsDispCb) return null;

  let compiled;
  try {
    compiled = getE2CompileFn()(
      gen.cSource,
      gen.koffiSignature,
      gen.cFnName,
      msg => process.stderr.write(`[e2] ${msg}\n`)
    );
  } catch (e) {
    throw new RuntimeError(
      `--opt e2: scalar-function kernel compilation failed for ${fn.name}: ${
        e instanceof Error ? e.message : String(e)
      }`
    );
  }
  if (!compiled) return null;

  const outputKinds = gen.outputDescs.map(o =>
    o.kind === "boolean" ? ("boolean" as const) : ("number" as const)
  );

  const paramDesc = fn.params
    .map((p, i) => `${p}: ${jitTypeKey(argTypes[i])}`)
    .join(", ");
  const file = fn.body[0]?.span?.file ?? interp.currentFile;
  const line = interp.rt.$line ?? 0;
  interp.onCCompile?.(
    `e2 kernel: scalar-fn ${fn.name}(${paramDesc}) -> ${outputKinds.length} scalar(s) @ ${file}:${line}`,
    gen.cSource
  );

  return { fn: compiled, outputKinds };
}

/**
 * Phase: run. Coerces booleans to 0/1, allocates per-output
 * Float64Array(1) buffers, invokes the C function, unpacks. The
 * koffi call runs to completion or raises — there is no
 * mid-execution bail mechanism.
 */
export function runE2ScalarFn(
  args: readonly unknown[],
  compiled: E2ScalarFnCompiled,
  nargout: number
): unknown {
  const callArgs: unknown[] = [];
  for (const a of args) {
    callArgs.push(typeof a === "boolean" ? (a ? 1 : 0) : a);
  }
  const outBufs: Float64Array[] = [];
  for (let k = 0; k < compiled.outputKinds.length; k++) {
    const buf = new Float64Array(1);
    outBufs.push(buf);
    callArgs.push(buf);
  }
  compiled.fn(...callArgs);

  if (compiled.outputKinds.length === 0) return undefined;
  const unpack = (k: number): RuntimeValue => {
    const raw = outBufs[k][0];
    return compiled.outputKinds[k] === "boolean" ? raw !== 0 : raw;
  };
  if (nargout <= 1) return unpack(0);
  const out: RuntimeValue[] = [];
  for (let k = 0; k < compiled.outputKinds.length; k++) {
    out.push(unpack(k));
  }
  return out;
}

/**
 * e2 — whole-function scalar C-kernel driver.
 *
 * Mirrors what e1 does for pure-scalar functions (benchmarks/scalar_bench.m's
 * `run_bench` is the motivating case) but triggers straight from the
 * interpreter's `callUserFunction` entry, not through the JS-JIT outer.
 * Under `--opt e2` the JS-JIT is disabled (optimization clamped to 0),
 * so we can't lean on `tryEmitScalarFnKernel` + the `$h.compileKernel`
 * plumbing; instead we invoke the shared lowering + C-emit pipeline
 * directly and call the resulting koffi function with plain scalar
 * args and Float64Array(1) out-buffers per output.
 *
 * Scope:
 *   - All args are scalar `number` or `boolean` RuntimeValues.
 *   - Declared outputs (the first `nargout || 1` of them) all lower to
 *     scalar / boolean types.
 *   - The body survives `checkCFeasibility` (no tic/toc, no Index
 *     writes, no disp, etc.).
 *
 * Outside this envelope we return `E2_SKIP` and the caller proceeds
 * with the interpreter path. Compilation failures are HARD errors —
 * mirrors the e2 multi-reduction/chain drivers' policy.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { FunctionDef } from "../../interpreter/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { RuntimeError } from "../../runtime/error.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import { type JitType, jitTypeKey, unifyJitTypes } from "../../jit/jitTypes.js";
import { lowerFunction } from "../../jit/jitLower.js";
import { generateC } from "../../jit/c/assemble.js";
import { checkCFeasibility } from "../../jit/c/feasibility.js";
import { getE2CompileFn } from "./compileFn.js";

export const E2_SKIP = Symbol("E2_SKIP");

interface ScalarFnCacheEntry {
  fn: (...args: unknown[]) => unknown;
  outputKinds: ("number" | "boolean")[];
}

/** Per-FunctionDef cache, keyed off the FunctionDef identity. */
const scalarFnCache = new WeakMap<
  FunctionDef,
  {
    lastArgTypes: Map<number, JitType[]>;
    entries: Map<string, ScalarFnCacheEntry | "BAILED">;
  }
>();

/** Try to run `fn(args)` via a whole-function C kernel. Returns
 *  `E2_SKIP` to fall through to the interpreter. */
export function tryE2ScalarFn(
  interp: Interpreter,
  fn: FunctionDef,
  args: unknown[],
  nargout: number
): unknown | typeof E2_SKIP {
  // Gate 0: param count matches. MATLAB allows fewer args (nargin), but
  // generateC's ABI is positional/fixed-arity — bail on under-supply.
  if (args.length !== fn.params.length) return E2_SKIP;

  // Gate 1: no `~` placeholder params (lowerFunction emits them as
  // `v_~`, which is not a valid C identifier).
  for (const p of fn.params) {
    if (p === "~") return E2_SKIP;
  }

  // Gate 2: every arg is a plain scalar number / boolean. Tensors, cells,
  // structs, chars, etc. need the tensor-aware lowering path we don't
  // have here.
  for (const a of args) {
    if (typeof a !== "number" && typeof a !== "boolean") return E2_SKIP;
  }

  // Infer arg JitTypes. Strip `exact` on numbers so per-call literals
  // don't explode the cache (mirrors tryJitCall's pruning).
  const argTypes: JitType[] = [];
  for (const a of args) {
    const t = inferJitType(a as RuntimeValue);
    if (t.kind === "number" && t.exact !== undefined) {
      const pruned: JitType = { kind: "number" };
      if (t.sign !== undefined) pruned.sign = t.sign;
      if (t.isInteger) pruned.isInteger = true;
      argTypes.push(pruned);
    } else if (t.kind === "number" || t.kind === "boolean") {
      argTypes.push(t);
    } else {
      return E2_SKIP;
    }
  }

  let cache = scalarFnCache.get(fn);
  if (!cache) {
    cache = { lastArgTypes: new Map(), entries: new Map() };
    scalarFnCache.set(fn, cache);
  }

  // Progressive widening against the previous call's types — mirrors
  // tryJitCall, so a function called with shifting scalar shapes lands
  // on a unified specialization rather than thrashing the cache.
  const prev = cache.lastArgTypes.get(nargout);
  if (prev && prev.length === argTypes.length) {
    for (let i = 0; i < argTypes.length; i++) {
      argTypes[i] = unifyJitTypes(argTypes[i], prev[i]);
    }
  }
  cache.lastArgTypes.set(nargout, argTypes.slice());

  const sig = `${nargout}|${argTypes.map(jitTypeKey).join(",")}`;
  let entry = cache.entries.get(sig);
  if (entry === "BAILED") return E2_SKIP;

  if (!entry) {
    const built = buildEntry(interp, fn, argTypes, nargout);
    if (built === null) {
      cache.entries.set(sig, "BAILED");
      return E2_SKIP;
    }
    entry = built;
    cache.entries.set(sig, entry);
  }

  // Call: pass each scalar arg (coerce booleans to 0/1 for koffi's
  // `double` slot), then one Float64Array(1) per output.
  const callArgs: unknown[] = [];
  for (const a of args) {
    callArgs.push(typeof a === "boolean" ? (a ? 1 : 0) : a);
  }
  const outBufs: Float64Array[] = [];
  for (let k = 0; k < entry.outputKinds.length; k++) {
    const buf = new Float64Array(1);
    outBufs.push(buf);
    callArgs.push(buf);
  }
  entry.fn(...callArgs);

  // Return one or many — matches MATLAB / interpreter convention
  // (`nargout <= 1` returns a single value; more returns an array).
  if (entry.outputKinds.length === 0) return undefined;
  const unpack = (k: number): RuntimeValue => {
    const raw = outBufs[k][0];
    return entry.outputKinds[k] === "boolean" ? raw !== 0 : raw;
  };
  if (nargout <= 1) return unpack(0);
  const out: RuntimeValue[] = [];
  for (let k = 0; k < entry.outputKinds.length; k++) {
    out.push(unpack(k));
  }
  return out;
}

function buildEntry(
  interp: Interpreter,
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number
): ScalarFnCacheEntry | null {
  // Lower the function body against the known scalar arg types. This
  // reuses the JS-JIT lowerer, which infers output types + collects
  // reachable callees' IR.
  const lowered = lowerFunction(fn, argTypes, nargout, interp);
  if (!lowered) return null;

  // All declared outputs (the ones we'll materialize) must be
  // scalar-ish, otherwise the out-buffer convention breaks.
  for (const t of lowered.outputTypes) {
    if (t.kind !== "number" && t.kind !== "boolean") return null;
  }

  const feas = checkCFeasibility(
    lowered.body,
    fn.params,
    argTypes,
    lowered.outputType,
    lowered.outputTypes,
    nargout,
    lowered.generatedIRBodies
  );
  if (!feas.ok) return null;

  let gen;
  try {
    gen = generateC(
      lowered.body,
      fn.params,
      lowered.outputNames,
      nargout,
      lowered.localVars,
      argTypes,
      lowered.outputType,
      lowered.outputTypes,
      fn.name.replace(/[^A-Za-z0-9_]/g, "_"),
      false,
      false,
      lowered.generatedIRBodies
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

/**
 * e1 (experimental) — whole-function scalar kernel emission.
 *
 * Complements [kernelEmit.ts](./kernelEmit.ts) (which handles tensor
 * fusible chains) by covering the other big win case: a user function
 * that is entirely scalar arithmetic — e.g. the inner loop of a
 * Horner-style series, a Runge-Kutta step on a handful of doubles,
 * benchmarks/scalar_bench.m's `run_bench(N, M)`.
 *
 * Under `--opt e1`, when a JIT-able function's signature and body are
 * purely scalar, we call `generateC()` (the same emitter the C-JIT
 * uses at `--opt 2`) and wrap its output with a thin inline JS
 * function that shells out to `$h.compileKernel(...)`. The C source
 * and koffi signature are inlined as JS string literals, so
 * `--dump-js` shows the complete picture.
 *
 * Scope for the prototype:
 *   - All params are scalar doubles / booleans (CParamDesc.kind === "scalar")
 *   - All outputs are scalar / boolean (COutputDesc.kind === "scalar" | "boolean")
 *   - No tic/toc, no Index reads (no errFlag), no disp(...) calls
 *
 * Anything outside that envelope returns `null` and the caller falls
 * back to the plain JS-JIT path, which still benefits from e1's
 * per-chain tensor kernels.
 */

import type { FunctionDef } from "../../interpreter/types.js";
import type { JitStmt, JitType } from "../jitTypes.js";
import type { GeneratedFn } from "../jitLower.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import { checkCFeasibility } from "../c/feasibility.js";
import { generateC } from "../c/assemble.js";
import { isOpenmpAvailable } from "./openmpFlag.js";

export interface ScalarFnKernelResult {
  /** The inline-compileKernel JS source. The JIT caller splices this
   *  in place of the normal JS-JIT body. */
  jsSource: string;
  /** Content-addressed kernel name from generateC, for logging. */
  kernelName: string;
  /** Raw C source (also embedded in `jsSource` as a string literal).
   *  Exposed for `--dump-c` / logging. */
  cSource: string;
}

/**
 * Try to emit a whole-function scalar kernel for the given lowered IR.
 * Returns null when the function is not a pure-scalar candidate.
 */
export function tryEmitScalarFnKernel(
  interp: Interpreter,
  fn: FunctionDef,
  body: JitStmt[],
  outputNames: string[],
  localVars: Set<string>,
  outputType: JitType | null,
  outputTypes: JitType[],
  argTypes: JitType[],
  nargout: number,
  generatedIRBodies: Map<string, GeneratedFn>
): ScalarFnKernelResult | null {
  // Gate 1: all arg types are scalar-ish. Bail early if any tensor /
  // complex / struct / string shows up — generateC might still succeed,
  // but the wrapper marshaling below assumes pure doubles. `complex_or_number`
  // is rejected because we can't statically rule out a complex runtime
  // value, and the scalar wrapper doesn't have a re/im marshalling path.
  for (const t of argTypes) {
    if (t.kind !== "number" && t.kind !== "boolean") return null;
  }
  // Gate 2: the declared/inferred outputs are all scalar-ish. A tensor
  // output (e.g. `function y = f(x)` where y is a vector) needs the
  // full marshaling path in install.ts, which we don't replicate here.
  for (const t of outputTypes) {
    if (t.kind !== "number" && t.kind !== "boolean") return null;
  }
  if (
    outputType &&
    outputType.kind !== "number" &&
    outputType.kind !== "boolean"
  ) {
    return null;
  }

  // Gate 3: run the C-JIT feasibility check. Rejects bodies with any
  // tensor write-back, Index writes, member-index writes, etc. The
  // check is per-function; nested callees are validated separately
  // when they're first emitted.
  const feas = checkCFeasibility(
    body,
    fn.params,
    argTypes,
    outputType,
    outputTypes,
    nargout,
    generatedIRBodies
  );
  if (!feas.ok) return null;

  // Emit C. `fuse`/`par` don't apply to scalar-only bodies (no tensor
  // loops to fuse), so pass both as false to keep the C source simple.
  let gen;
  try {
    gen = generateC(
      body,
      fn.params,
      outputNames,
      nargout,
      localVars,
      argTypes,
      outputType,
      outputTypes,
      fn.name.replace(/[^A-Za-z0-9_]/g, "_"),
      false,
      interp.par && isOpenmpAvailable(),
      generatedIRBodies
    );
  } catch {
    return null;
  }

  // Gate 4: the resulting signature must be pure-scalar. If any param
  // is a tensor (argTypes gate should have prevented this but let's be
  // defensive) or any output is a tensor, bail. Also bail on any
  // "trailer" features that need extra per-call state.
  for (const p of gen.paramDescs) {
    if (p.kind !== "scalar") return null;
  }
  for (const o of gen.outputDescs) {
    if (o.kind !== "scalar" && o.kind !== "boolean") return null;
  }
  if (gen.needsTicState || gen.needsErrorFlag || gen.needsDispCb) return null;

  // Build the JS wrapper. One `double *` out-pointer per output — the
  // JS side allocates a Float64Array(1) buffer per output, calls, then
  // reads back. Scalar params pass through as-is.
  const paramList = fn.params.join(", ");
  const numOut = gen.outputDescs.length;
  const outBufs: string[] = [];
  for (let k = 0; k < numOut; k++) {
    outBufs.push(`const __o${k} = new Float64Array(1);`);
  }
  const callArgs = [...fn.params, ...outBufs.map((_, k) => `__o${k}`)].join(
    ", "
  );

  // Coerce boolean / number return according to the declared output
  // kinds. A single output returns a plain value (boolean or number);
  // multiple outputs return an array (matches the JS-JIT convention
  // already used elsewhere).
  const toReturn = (k: number): string => {
    if (gen.outputDescs[k].kind === "boolean") return `__o${k}[0] !== 0`;
    return `__o${k}[0]`;
  };
  const returnExpr =
    numOut === 0
      ? "undefined"
      : numOut === 1
        ? toReturn(0)
        : `[${Array.from({ length: numOut }, (_, k) => toReturn(k)).join(", ")}]`;

  // Emit the JS. The kernel is cached on `$h.$kernels[<name>]` so the
  // same specialization used elsewhere dedupes to one `cc` invocation.
  const kernelKey = JSON.stringify(gen.cFnName);
  const cSrcJs = JSON.stringify(gen.cSource);
  const koffiSigJs = JSON.stringify(gen.koffiSignature);

  const lines: string[] = [];
  lines.push(`function ${fn.name}(${paramList}) {`);
  lines.push(
    `  $h.$kernels[${kernelKey}] ??= $h.compileKernel(${cSrcJs}, ${koffiSigJs});`
  );
  for (const d of outBufs) lines.push(`  ${d}`);
  lines.push(`  $h.$kernels[${kernelKey}](${callArgs});`);
  lines.push(`  return ${returnExpr};`);
  lines.push(`}`);
  const jsSource = lines.join("\n");

  return {
    jsSource,
    kernelName: gen.cFnName,
    cSource: gen.cSource,
  };
}

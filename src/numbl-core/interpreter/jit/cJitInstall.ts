/**
 * Installs the real C-JIT backend. Side-effect import only.
 *
 * Must be imported exactly once from a Node-only entry point (currently
 * src/cli.ts). The browser bundle never reaches this file, so the
 * Node-only dependencies of [c/cCompile.ts](./c/cCompile.ts) stay out
 * of the web build.
 */

import { registerCJitBackend } from "./cJitBackend.js";
import { checkCFeasibility } from "./c/cFeasibility.js";
import { generateC } from "./c/jitCodegenC.js";
import { generateNapiShim, type ReturnCKind } from "./c/cNapiShim.js";
import { compileAndLoad } from "./c/cCompile.js";
import { C_JIT_BAIL_SENTINEL } from "./c/cJitHelpers.js";
import { jitTypeKey } from "./jitTypes.js";
import { JitBailToInterpreter } from "./jitHelpersIndex.js";

registerCJitBackend({
  tryCompile(
    interp,
    fn,
    body,
    outputNames,
    localVars,
    outputType,
    argTypes,
    nargout
  ) {
    const feas = checkCFeasibility(body, argTypes, outputType, nargout);
    if (!feas.ok) return null;

    // Stable C function name — same for all specializations of this source
    // function. Each .node module holds exactly one function, so there's no
    // symbol collision across specializations. Keeping the name stable means
    // two specializations that produce identical C (e.g. after the usual
    // exact-literal→widened-number type unification) share a single on-disk
    // cache entry rather than each spending ~50ms on a fresh cc invocation.
    const gen = generateC(
      body,
      fn.params,
      outputNames,
      nargout,
      localVars,
      argTypes,
      outputType,
      fn.name.replace(/[^A-Za-z0-9_]/g, "_")
    );

    const returnKind: ReturnCKind = gen.returnIsTensor
      ? "tensor"
      : outputType && outputType.kind === "boolean"
        ? "boolean"
        : "number";
    const { shim, exportName } = generateNapiShim(
      gen.cFnName,
      gen.paramDescs,
      returnKind,
      gen.usesTensors
    );

    const loaded = compileAndLoad(gen.cSource, shim, exportName, interp.log);
    if (!loaded) return null;

    // Fire --dump-c callback.
    const line = interp.rt.$line ?? 0;
    const typeDesc = argTypes.map(jitTypeKey).join(", ");
    const description = `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`;
    interp.onCJitCompile?.(description, gen.cSource + "\n\n" + shim);

    const nativeFn = loaded.fn;
    return (...callArgs: unknown[]): unknown => {
      try {
        return nativeFn(...callArgs);
      } catch (e) {
        // The tensor helpers throw a JS error with the bail sentinel
        // whenever they hit an unsupported input (complex, mismatched
        // shape, matrix sum, ...). Convert to JitBailToInterpreter so
        // the interpreter retries through the slow path, same as the
        // JS-JIT's `return undefined` branches do.
        if (e instanceof Error && e.message === C_JIT_BAIL_SENTINEL) {
          throw new JitBailToInterpreter("C-JIT tensor helper bail");
        }
        throw e;
      }
    };
  },
});

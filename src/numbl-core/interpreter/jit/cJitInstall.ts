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
import { generateNapiShim } from "./c/cNapiShim.js";
import { compileAndLoad } from "./c/cCompile.js";
import { jitTypeKey } from "./jitTypes.js";

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
      fn.name.replace(/[^A-Za-z0-9_]/g, "_")
    );

    const argKinds = argTypes.map(t =>
      t.kind === "boolean" ? ("boolean" as const) : ("number" as const)
    );
    const returnKind: "boolean" | "number" =
      outputType && outputType.kind === "boolean" ? "boolean" : "number";
    const { shim, exportName } = generateNapiShim(
      gen.cFnName,
      argKinds,
      returnKind
    );

    const loaded = compileAndLoad(gen.cSource, shim, exportName, interp.log);
    if (!loaded) return null;

    // Fire --dump-c callback.
    const line = interp.rt.$line ?? 0;
    const typeDesc = argTypes.map(jitTypeKey).join(", ");
    const description = `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`;
    interp.onCJitCompile?.(description, gen.cSource + "\n\n" + shim);

    const nativeFn = loaded.fn;
    return (...callArgs: unknown[]): unknown =>
      nativeFn(...(callArgs as number[]));
  },
});

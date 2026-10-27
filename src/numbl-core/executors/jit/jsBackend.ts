/**
 * JS backend: `compileSpec` emits a JS factory, materialized with
 * `new Function` and run in-process.
 */

import { compileSpec, type Type as CompilerType } from "../../jit/index.js";
import { jitTypeToCompilerType } from "./typeAdapter.js";
import { numblToJit, jitToNumbl, isGrowBail } from "./valueAdapter.js";
import { getOrCreateSession } from "./session.js";
import { buildHostHelpers, type JitHostHelpers } from "./hostHelpers.js";
import type { JitBackend } from "./backend.js";

interface JsArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
  readonly argTypes: readonly CompilerType[];
  /** Needed to rebuild a returned struct (an untagged JS object). */
  readonly outputTypes: readonly CompilerType[];
  readonly nargout: number;
}

export const jsBackend: JitBackend<JsArtifact> = {
  lang: "js",
  prefix: "jit",

  available: () => true,

  mapType: jitTypeToCompilerType,

  compile(req, interp) {
    const { workspace, lowerer } = getOrCreateSession(interp);
    const { source, cName, outputTypes } = compileSpec({
      workspace,
      lowerer,
      funcDecl: req.funcDecl,
      argTypes: req.argTypes,
      nargout: req.nargout,
    });
    interp.onJitCompile?.(`${req.where}:${cName}${req.describe}`, source, "js");
    const factory = new Function(source)() as (
      $h: JitHostHelpers
    ) => (...args: unknown[]) => unknown;
    return {
      specFn: factory(buildHostHelpers(interp.rt)),
      argTypes: req.argTypes,
      outputTypes,
      nargout: req.nargout,
    };
  },

  invoke(art, args, interp, where) {
    try {
      // An unbound input (read-before-write in a script) passes through
      // as undefined; the spec throws at its first use.
      const jsArgs = args.map((v, i) =>
        v === undefined ? undefined : numblToJit(v, art.argTypes[i])
      );
      const ret = art.specFn(...jsArgs);
      // Return convention: nargout 0 → nothing, 1 → bare value, ≥2 → array.
      const n = art.nargout;
      const raw = n === 0 ? [] : n === 1 ? [ret] : ret;
      if (!Array.isArray(raw)) {
        throw new Error(`${where}: expected array for nargout=${n}`);
      }
      return raw.map((v, i) =>
        v === undefined ? undefined : jitToNumbl(v, art.outputTypes[i])
      );
    } catch (e) {
      // Any runtime error bails so the interpreter reproduces numbl's
      // canonical behavior. A grow-store sentinel (`v(k) = x` past the
      // runtime extent) is a JIT limitation, so it also warns.
      if (isGrowBail(e)) {
        interp.onJitBail?.(
          `${where}: indexed-store array growth; falling back to interpreter`
        );
      }
      return null;
    }
  },
};

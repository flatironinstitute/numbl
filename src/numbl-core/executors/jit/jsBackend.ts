/**
 * JS backend: `compileSpec` emits a JS factory, materialized with
 * `new Function` and run in-process.
 */

import {
  compileSpec,
  UnsupportedConstruct,
  type Type as CompilerType,
} from "../../jit/index.js";
import { typeToString } from "../../jit/lowering/types.js";
import { decodeLoc } from "../../jit/codegen/runtime.js";
import { CancellationError, RuntimeError } from "../../runtime/error.js";
import { numblToJit, jitToNumbl } from "./valueAdapter.js";
import { getOrCreateSession } from "./session.js";
import { buildHostHelpers, type JitHostHelpers } from "./hostHelpers.js";
import { outputIsReproducible, type JitBackend } from "./backend.js";
import { JitRaise, isTaggedJsRaise } from "./raise.js";

interface JsArtifact {
  readonly specFn: (...args: unknown[]) => unknown;
  readonly argTypes: readonly CompilerType[];
  /** Needed to rebuild a returned struct (an untagged JS object). */
  readonly outputTypes: readonly CompilerType[];
  readonly nargout: number;
  /** The module's current statement location (`$loc`). */
  readonly loc: () => number;
  readonly locFiles: ReadonlyArray<string>;
  readonly recursive: boolean;
  readonly streamsOutput: boolean;
  readonly where: string;
}

export const jsBackend: JitBackend<JsArtifact> = {
  lang: "js",
  prefix: "jit",

  available: () => true,

  accepts: t => t.kind !== "Unknown",

  compile(req, interp) {
    const { workspace, lowerer } = getOrCreateSession(interp);
    const {
      source,
      cName,
      outputTypes,
      locFiles,
      recursive,
      activatedSnippets,
    } = compileSpec({
      workspace,
      lowerer,
      funcDecl: req.funcDecl,
      argTypes: req.argTypes,
      nargout: req.nargout,
    });
    // An output `jitToNumbl` cannot rebuild (a handle, class instance or
    // cell) would only fail after the unit ran: decline up front.
    const stuck = outputTypes.find(t => !returnable(t));
    if (stuck !== undefined) {
      throw new UnsupportedConstruct(
        `a ${typeToString(stuck)} output cannot be returned to the interpreter`
      );
    }
    interp.onJitCompile?.(`${req.where}:${cName}${req.describe}`, source, "js");
    const factory = new Function(source)() as (
      $h: JitHostHelpers
    ) => (...args: unknown[]) => unknown;
    const helpers = buildHostHelpers(interp.rt);
    const specFn = factory(helpers);
    return {
      specFn,
      argTypes: req.argTypes,
      outputTypes,
      nargout: req.nargout,
      loc: helpers.loc ?? (() => -1),
      locFiles,
      recursive,
      streamsOutput: outputIsReproducible(activatedSnippets),
      where: req.where,
    };
  },

  recursive: art => art.recursive,

  streamsOutput: art => art.streamsOutput,

  invoke(art, args) {
    const jsArgs = args.map((v, i) =>
      v === undefined ? undefined : numblToJit(v, art.argTypes[i])
    );
    let ret: unknown;
    try {
      ret = art.specFn(...jsArgs);
    } catch (e) {
      throw toJitRaise(e, art);
    }
    // Return convention: nargout 0 → nothing, 1 → bare value, ≥2 → array.
    const n = art.nargout;
    const raw = n === 0 ? [] : n === 1 ? [ret] : ret;
    if (!Array.isArray(raw)) {
      throw new Error(`${art.where}: expected array for nargout=${n}`);
    }
    return raw.map((v, i) =>
      v === undefined ? undefined : jitToNumbl(v, art.outputTypes[i])
    );
  },
};

/** Whether `jitToNumbl` can rebuild a value of this type. */
function returnable(t: CompilerType): boolean {
  switch (t.kind) {
    case "Struct":
      return t.fields.every(f => returnable(f.ty));
    case "Handle":
    case "Class":
    case "Cell":
      return false;
    default:
      return true;
  }
}

/** Normalize what the emitted code threw (see `raise.ts`). Runtime errors
 *  and bails arrive as the module's tagged `JitRaise`; a RuntimeError comes
 *  from a host helper the code called (e.g. a plot builtin) and carries the
 *  interpreter's own wording. A JS stack overflow is left to the
 *  interpreter's re-run, which raises its own recursion-limit error.
 *  Anything else is a JIT bug: it propagates as an internal error rather
 *  than being masked by a silent re-run. */
function toJitRaise(e: unknown, art: JsArtifact): Error {
  if (e instanceof CancellationError) return e;
  const loc = () => decodeLoc(art.loc(), art.locFiles);
  if (isTaggedJsRaise(e)) {
    return new JitRaise(e.kind, e.identifier, e.raiseMessage, loc());
  }
  if (e instanceof RuntimeError) {
    return new JitRaise("error", e.identifier, e.message, loc());
  }
  if (e instanceof RangeError && /call stack/i.test(e.message)) {
    return new JitRaise("error", "", null, loc());
  }
  const msg = e instanceof Error ? e.message : String(e);
  return new Error(`Internal JIT error in ${art.where}: ${msg}`, {
    cause: e,
  });
}

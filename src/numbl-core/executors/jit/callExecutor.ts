/**
 * `<prefix>-call` — compiles a user-function call specialized on its
 * argument types and nargout.
 *
 * Declines `nargout === 0` (bare `f();`): the interpreter still binds the
 * first declared output to `ans`, but a nargout=0 spec returns nothing.
 * Declines inside a loop that is itself being compiled (see
 * `insideCompiledLoop`): the call then runs inside that compiled loop.
 */

import { RERUN, type Executor } from "../types.js";
import type { CallClassification } from "../classification.js";
import { jitTypeKey } from "../../jitTypes.js";
import type { Span } from "../../parser/index.js";
import type { FunctionDef } from "../../interpreter/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import type { Type as CompilerType } from "../../jit/index.js";
import {
  compileUnit,
  declinedByAssertC,
  insideCompiledLoop,
  invokeUnit,
  mapTypes,
  type FuncStmt,
  type JitBackend,
} from "./backend.js";
import { scopeOfStmts } from "./raise.js";

interface CallData {
  readonly c: CallClassification;
  readonly argTypes: CompilerType[];
  readonly args: readonly unknown[];
}

/** Build a parser-shaped `FuncStmt` from a `FunctionDef`.
 *
 *  `FunctionDef` carries no span, but the compiler keys its spec cache on
 *  the defining file, so derive it from the first body statement —
 *  otherwise two same-named functions from different files (e.g. after
 *  addpath/rmpath shadowing) would share one spec. An empty body has no
 *  shadowing concern and falls back to `<jit>`. */
function funcStmtFromDef(fd: FunctionDef): FuncStmt {
  const span: Span = {
    file: fd.body[0]?.span?.file ?? "<jit>",
    start: 0,
    end: 0,
  };
  return {
    type: "Function",
    name: fd.name,
    functionId: fd.name,
    params: [...fd.params],
    outputs: [...fd.outputs],
    body: fd.body,
    argumentsBlocks: fd.argumentsBlocks ?? [],
    span,
  };
}

export function makeCallExecutor<A>(
  backend: JitBackend<A>
): Executor<"call", CallData, A | null> {
  const name = `${backend.prefix}-call`;
  return {
    name,
    kind: "call",

    propose(lowered, interp) {
      if (!backend.available(interp)) return null;
      if (insideCompiledLoop(interp)) return null;
      const c = lowered.classification;
      if (declinedByAssertC(backend, interp, c.assertsCJit)) return null;
      if (c.nargout === 0) return null;
      const argTypes = mapTypes(backend, c.argTypes);
      return argTypes && { c, argTypes, args: lowered.args };
    },

    cacheKey: d => d.c.cacheKey,

    compile(d, interp) {
      const { c } = d;
      return compileUnit(
        backend,
        {
          where: name,
          funcDecl: funcStmtFromDef(c.fn),
          argTypes: d.argTypes,
          nargout: c.nargout,
          synthetic: false,
          describe: `(${c.argTypes.map(jitTypeKey).join(", ")}) -> nargout=${c.nargout}`,
        },
        interp
      );
    },

    streamsOutput: art => art !== null && backend.streamsOutput(art),

    run(art, d, interp) {
      if (art === null) return null;
      const fd = d.c.fn;
      const scope = {
        ...scopeOfStmts(fd.body, "<jit>", fd.name),
        reentrant: backend.recursive(art),
      };
      const outs = invokeUnit(
        backend,
        art,
        d.args as readonly RuntimeValue[],
        scope,
        interp,
        name
      );
      if (!Array.isArray(outs)) return outs;
      // An unassigned output is the interpreter's error to raise.
      if (outs.includes(undefined)) return RERUN;
      // Same shape as the interpreter's call path: a bare value for
      // nargout 1, an array otherwise.
      return { result: d.c.nargout >= 2 ? outs : outs[0] };
    },
  };
}

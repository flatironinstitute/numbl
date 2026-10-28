/**
 * `<prefix>-loop` — compiles a single For/While stmt, wrapped as a
 * synthetic function whose params are the variables the loop reads from
 * the env and whose outputs are the variables it assigns that are live
 * after the loop (filtered in classification).
 *
 * Fires for outermost loops only, unless every enclosing loop was itself
 * declined (see `insideCompiledLoop`).
 */

import type { Executor } from "../types.js";
import type { LoopClassification } from "../classification.js";
import { jitTypeKey } from "../../jitTypes.js";
import type { Stmt } from "../../parser/index.js";
import type { Type as CompilerType } from "../../jit/index.js";
import {
  compileUnit,
  declinedByAssertC,
  insideCompiledLoop,
  mapTypes,
  runInEnv,
  synthesizeFuncStmt,
  type JitBackend,
} from "./backend.js";
import { scopeOfStmts } from "./raise.js";

interface LoopData {
  readonly c: LoopClassification;
  readonly argTypes: CompilerType[];
}

export function makeLoopExecutor<A>(
  backend: JitBackend<A>
): Executor<"loop", LoopData, A | null> {
  const name = `${backend.prefix}-loop`;
  return {
    name,
    kind: "loop",

    propose(lowered, interp) {
      if (!backend.available(interp)) return null;
      if (insideCompiledLoop(interp)) return null;
      const c = lowered.classification;
      if (declinedByAssertC(backend, interp, c.assertsCJit)) return null;
      // A `return` inside the loop has no place in the function-body model.
      if (c.hasReturn) return null;
      const argTypes = mapTypes(backend, c.inputTypes);
      return argTypes && { c, argTypes };
    },

    cacheKey: d => d.c.cacheKey,

    compile(d, interp) {
      const { c } = d;
      // Capture-free handle inputs are inlined as in-scope `<name> = @...`
      // assignments at the top of the body (not params), reducing the
      // boundary case to the supported in-scope handle case.
      const handleDefs: Stmt[] = c.constHandles.map(h => ({
        type: "Assign",
        name: h.name,
        expr: h.expr,
        suppressed: true,
        span: { file: c.currentFile, start: 0, end: 0 },
      }));
      // The name embeds the loop's source offset so two same-shape loops
      // in one file don't collide in the compiler's spec cache.
      return compileUnit(
        backend,
        {
          where: name,
          funcDecl: synthesizeFuncStmt(
            `$loop_${c.stmt.span?.start ?? 0}`,
            c.inputs,
            c.outputs,
            [...handleDefs, c.stmt],
            c.currentFile
          ),
          argTypes: d.argTypes,
          nargout: c.outputs.length,
          synthetic: true,
          describe: `(${c.inputTypes.map(jitTypeKey).join(", ")}) -> outputs=${c.outputs.length}`,
        },
        interp
      );
    },

    run(art, d, interp) {
      if (art === null) return null;
      const scope = scopeOfStmts([d.c.stmt], d.c.currentFile);
      return runInEnv(
        backend,
        art,
        d.c.inputs,
        d.c.outputs,
        scope,
        interp,
        name
      );
    },
  };
}

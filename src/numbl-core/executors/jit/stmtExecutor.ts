/**
 * `<prefix>-stmt` — compiles one interpreted assignment whose right-hand
 * side is a large element-wise expression, so that it runs as one fused
 * kernel loop instead of one interpreter pass (and one temporary) per
 * operation. It covers statements no other shape reaches: those in a
 * script or function body the whole-scope or call shape declined, and in
 * the body of a declined loop.
 *
 * `tryLowerStmt` gates it cheaply (an AST verdict cached per node, then a
 * size check on the inputs); a compiled unit that holds no kernel (the
 * operand shapes did not fuse) is cached as a decline, so such a
 * statement is not compiled or invoked again. Registered for the JS
 * backend only: a C compile costs far more than one statement can save.
 */

import type { Executor } from "../types.js";
import type { StmtClassification } from "../classification.js";
import {
  acceptsAll,
  compileUnit,
  describeTypes,
  insideCompiledLoop,
  runInEnv,
  synthesizeFuncStmt,
  type JitBackend,
} from "./backend.js";
import { scopeOfStmts } from "./raise.js";

interface StmtData {
  readonly c: StmtClassification;
}

export function makeStmtExecutor<A>(
  backend: JitBackend<A>
): Executor<"stmt", StmtData, A | null> {
  const name = `${backend.prefix}-stmt`;
  return {
    name,
    kind: "stmt",

    propose(lowered, interp) {
      if (!backend.available(interp)) return null;
      if (insideCompiledLoop(interp)) return null;
      const c = lowered.classification;
      return acceptsAll(backend, c.inputTypes) ? { c } : null;
    },

    cacheKey: d => d.c.cacheKey,

    compile(d, interp) {
      const { c } = d;
      const art = compileUnit(
        backend,
        {
          where: name,
          funcDecl: synthesizeFuncStmt(
            `$stmt_${c.stmt.span?.start ?? 0}`,
            c.inputs,
            c.outputs,
            [c.stmt],
            c.currentFile
          ),
          argTypes: [...c.inputTypes],
          nargout: c.outputs.length,
          synthetic: true,
          // `y = <expr>` reads its inputs and assigns a fresh result.
          borrowsInputs: true,
          describe: `${describeTypes(c.inputTypes)} -> outputs=${c.outputs.length}`,
        },
        interp
      );
      return art !== null && backend.kernels(art) > 0 ? art : null;
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

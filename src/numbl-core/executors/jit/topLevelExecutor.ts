/**
 * `<prefix>-top-level` — compiles a whole script body, wrapped as a
 * synthetic function whose params are the variables the body reads from
 * the env and whose outputs are the variables it assigns.
 *
 * The interpreter calls `Registry.tryRunWholeScope` once at script start.
 * On a decline the per-stmt loop runs instead, where the loop and call
 * executors still get their chance.
 */

import type { Executor } from "../types.js";
import type { TopLevelClassification } from "../classification.js";
import { jitTypeKey } from "../../jitTypes.js";
import type { Type as CompilerType } from "../../jit/index.js";
import {
  compileUnit,
  declinedByAssertC,
  mapTypes,
  runInEnv,
  synthesizeFuncStmt,
  type JitBackend,
} from "./backend.js";

interface TopLevelData {
  readonly c: TopLevelClassification;
  readonly argTypes: CompilerType[];
}

/** The interpreter echoes unsuppressed assignments ("x = 10"); the
 *  emitted code does not, so scripts with them decline.
 *
 *  Unsuppressed bare expressions pass on the JS backend: a void call
 *  (`disp(...)`) prints on its own, and `compileSpec` rejects a
 *  value-yielding one (`1 + 2`, which would echo "ans = 3"). The C backend
 *  keeps the stricter gate (bare expressions decline too) because its
 *  whole-scope path is less battle-tested; such scripts fall to JS-JIT at
 *  --opt 2 unless they opt in with `%!numbl:assert_jit c`, which the
 *  annotator adds only after verifying the script C-JITs correctly. */
function displayGateOk(
  backend: JitBackend,
  c: TopLevelClassification
): boolean {
  const strict = backend.lang === "c";
  if (strict && c.assertsCJit) return true;
  for (const s of c.stmts) {
    if (
      (s.type === "Assign" ||
        s.type === "MultiAssign" ||
        (strict && s.type === "ExprStmt")) &&
      !s.suppressed
    )
      return false;
  }
  return true;
}

export function makeTopLevelExecutor<A>(
  backend: JitBackend<A>
): Executor<"top-level", TopLevelData, A | null> {
  const name = `${backend.prefix}-top-level`;
  return {
    name,
    kind: "top-level",

    propose(lowered, interp) {
      if (!backend.available(interp)) return null;
      // Defensive: whole-scope only fires at script entry.
      if (interp.loopDepth > 0) return null;
      const c = lowered.classification;
      if (declinedByAssertC(backend, interp, c.assertsCJit)) return null;
      // A top-level `return` has no place in the function-body model.
      if (c.hasReturn) return null;
      if (!displayGateOk(backend, c)) return null;
      const argTypes = mapTypes(backend, c.inputTypes);
      return argTypes && { c, argTypes };
    },

    cacheKey: d => d.c.cacheKey,

    compile(d, interp) {
      const { c } = d;
      return compileUnit(
        backend,
        {
          where: name,
          funcDecl: synthesizeFuncStmt(
            "$top",
            c.inputs,
            c.outputs,
            [...c.stmts],
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
      return runInEnv(backend, art, d.c.inputs, d.c.outputs, interp, name);
    },
  };
}

import { describe, it, expect } from "vitest";
import { parseMFile } from "../numbl-core/parser/index.js";
import { Workspace, Lowerer, tensorDouble } from "../numbl-core/jit/index.js";
import { specializeUserFunction } from "../numbl-core/jit/lowering/specialize.js";
import { inlinePass } from "../numbl-core/jit/codegen/inlinePass.js";
import type { IRFunc } from "../numbl-core/jit/lowering/ir.js";

/** Lower the first function in `source` for one tensor argument. */
function lowerFn(source: string, shape: number[]): IRFunc {
  const fileName = "f.m";
  const ast = parseMFile(source, fileName);
  const ws = new Workspace(fileName, []);
  ws.addFile({ name: fileName, source, ast });
  ws.finalize();
  const lowerer = new Lowerer(ws);
  const decl = (ast.body as { type: string }[]).find(
    s => s.type === "Function"
  );
  return specializeUserFunction.call(
    lowerer,
    decl as never,
    [tensorDouble(shape)],
    undefined,
    undefined,
    undefined,
    1,
    undefined
  );
}

describe("inlinePass", () => {
  it("fuses more than 32 single-use temps into one statement", () => {
    // 40 terms: 40 `x .* k` temps plus 39 partial sums, all single-use.
    const terms = Array.from({ length: 40 }, (_, k) => `x .* ${k + 1}`);
    const source = `function y = f(x)\n  y = ${terms.join(" + ")};\nend\n`;
    const fn = lowerFn(source, [4, 1]);
    expect(fn.body.length).toBeGreaterThan(33);

    inlinePass({ topLevelStmts: [], functions: new Map([[fn.cName, fn]]) });
    const assigns = fn.body.filter(s => s.kind === "Assign");
    expect(assigns).toHaveLength(1);
  });
});

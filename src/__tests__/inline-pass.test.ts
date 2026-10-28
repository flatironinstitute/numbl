import { describe, it, expect } from "vitest";
import { parseMFile } from "../numbl-core/parser/index.js";
import { CodeIndex } from "../numbl-core/workspace/codeIndex.js";
import { Workspace, Lowerer, tensorDouble } from "../numbl-core/jit/index.js";
import { specializeUserFunction } from "../numbl-core/jit/lowering/specialize.js";
import { inlineFunc } from "../numbl-core/jit/codegen/inlinePass.js";
import type { IRFunc, IRStmt } from "../numbl-core/jit/lowering/ir.js";

/** Lower the first function in `source`, every argument a tensor of
 *  `shape`, nargout = its declared output count. */
function lowerFn(source: string, shape: number[]): IRFunc {
  const fileName = "f.m";
  const ast = parseMFile(source, fileName);
  const code = new CodeIndex(fileName, source);
  code.setMainFile(ast);
  const decl = ast.body.find(s => s.type === "Function");
  if (decl?.type !== "Function") throw new Error("no function");
  code.buildFunctionIndex();
  const ws = Workspace.fromCodeIndex(code, fileName, [
    { name: fileName, source, ast },
  ]);
  const lowerer = new Lowerer(ws);
  return specializeUserFunction.call(
    lowerer,
    decl,
    decl.params.map(() => tensorDouble(shape)),
    undefined,
    undefined,
    undefined,
    decl.outputs.length,
    undefined
  );
}

/** Names assigned by Assign statements, nested bodies included. */
function assignedNames(stmts: IRStmt[]): string[] {
  const out: string[] = [];
  for (const s of stmts) {
    if (s.kind === "Assign") out.push(s.name);
    if (s.kind === "If") {
      out.push(...assignedNames(s.thenBody), ...assignedNames(s.elseBody));
    } else if (s.kind === "For" || s.kind === "While") {
      out.push(...assignedNames(s.body));
    }
  }
  return out;
}

describe("inlinePass", () => {
  it("fuses more than 32 single-use temps into one statement", () => {
    // 40 terms: 40 `x .* k` temps plus 39 partial sums, all single-use.
    const terms = Array.from({ length: 40 }, (_, k) => `x .* ${k + 1}`);
    const source = `function y = f(x)\n  y = ${terms.join(" + ")};\nend\n`;
    const fn = lowerFn(source, [4, 1]);
    expect(fn.body.length).toBeGreaterThan(33);
    expect(assignedNames(inlineFunc(fn).body)).toEqual(["y"]);
  });

  it("fuses an elementwise call on an expression into one statement", () => {
    const source = "function y = f(x)\n  y = 1 ./ realsqrt(x .* x + 1);\nend\n";
    const fn = lowerFn(source, [4, 1]);
    expect(assignedNames(inlineFunc(fn).body)).toEqual(["y"]);
  });

  it("fuses two-argument max with a tensor into one statement", () => {
    const source = "function y = f(x)\n  y = 2 * max(x .* x - 1, 0);\nend\n";
    const fn = lowerFn(source, [4, 1]);
    expect(assignedNames(inlineFunc(fn).body)).toEqual(["y"]);
  });

  it("does not mutate the input function", () => {
    const source = "function y = f(x)\n  y = 1 ./ realsqrt(x .* x + 1);\nend\n";
    const fn = lowerFn(source, [4, 1]);
    const before = fn.body.slice();
    const out = inlineFunc(fn);
    expect(out).not.toBe(fn);
    expect(fn.body).toEqual(before);
  });

  it("keeps a loop-body def that is read after the loop", () => {
    // `w` has one read inside the loop body, but is also an output
    // and read after the loop: its def must survive.
    const source =
      "function [z, w] = f(a, b)\n  z = a;\n  for k = 1:3\n" +
      "    w = a .* b;\n    z = w + 1;\n  end\n  q = w;\nend\n";
    const fn = lowerFn(source, [4, 1]);
    expect(assignedNames(inlineFunc(fn).body)).toContain("w");
  });

  it("inlines a chain inside a loop body", () => {
    const source =
      "function y = f(a, b)\n  y = a;\n  for k = 1:3\n" +
      "    y = a .* b + y .* b - a;\n  end\nend\n";
    const fn = lowerFn(source, [4, 1]);
    expect(assignedNames(inlineFunc(fn).body)).toEqual(["y", "y"]);
  });

  it("inlines into a consumer that overwrites a producer operand", () => {
    // `v = v + 0.1 .* (a .* v)`: the consumer writes `v`, which the
    // folded temps read; the RHS is evaluated before the store.
    const source = "function v = f(a, v)\n  v = v + 0.1 .* (a .* v);\nend\n";
    const fn = lowerFn(source, [4, 1]);
    expect(assignedNames(inlineFunc(fn).body)).toEqual(["v"]);
  });

  it("does not inline across an intervening write to an operand", () => {
    const source =
      "function y = f(a, b)\n  t = a .* b;\n  a = b;\n  y = t + a;\nend\n";
    const fn = lowerFn(source, [4, 1]);
    expect(assignedNames(inlineFunc(fn).body)).toEqual(["t", "a", "y"]);
  });
});

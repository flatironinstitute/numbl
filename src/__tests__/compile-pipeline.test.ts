import { describe, it, expect, beforeAll } from "vitest";
import { parseMFile } from "../numbl-core/parser/index.js";
import { CodeIndex } from "../numbl-core/workspace/codeIndex.js";
import {
  Workspace,
  Lowerer,
  compileSpec,
  compileSpecC,
  scalarDouble,
  tensorDoubleFromDims,
  type Type,
} from "../numbl-core/jit/index.js";
import { setCSnippets } from "../numbl-core/jit/codegen/runtime.js";
import {
  renameCIdentifiers,
  stripCComments,
} from "../numbl-core/jit/codegen/cSource.js";
import { C_SNIPPETS } from "../numbl-core/jit/builtins/runtime/snippets.c.gen.js";

// On a matrix of unknown size, `mask` lowers to a compare that C fuses
// into one kernel loop. JS fuses only part of it (not the power), so its
// kernel needs a shape guard whose broadcasting fallback evaluates the
// compare on tensors, which the JS emitter rejects. `inc` and `twice` are
// unrelated.
const SOURCE = `function r = mask(v)
  m = (v .^ 2 + (v > 0)) > 1;
  r = v(m);
end
function y = inc(x)
  y = x + 1;
end
function y = twice(x)
  y = inc(x) + inc(x);
end
`;

const matrix = tensorDoubleFromDims([{ kind: "unknown" }, { kind: "unknown" }]);

function session() {
  const fileName = "f.m";
  const ast = parseMFile(SOURCE, fileName);
  const code = new CodeIndex(fileName, SOURCE);
  code.setMainFile(ast);
  code.buildFunctionIndex();
  const workspace = Workspace.fromCodeIndex(code, fileName, [
    { name: fileName, source: SOURCE, ast },
  ]);
  const lowerer = new Lowerer(workspace);
  const fn = (name: string) => {
    const decl = ast.body.find(s => s.type === "Function" && s.name === name);
    if (decl?.type !== "Function") throw new Error(`no function ${name}`);
    return decl;
  };
  const args = (name: string, argTypes: Type[]) => ({
    workspace,
    lowerer,
    funcDecl: fn(name),
    argTypes,
    nargout: 1,
  });
  return { lowerer, args };
}

describe("compile pipeline", () => {
  beforeAll(() => setCSnippets(C_SNIPPETS));

  it("emits only the specs the entry reaches", () => {
    const { lowerer, args } = session();
    compileSpec(args("inc", [scalarDouble()]));
    const twice = compileSpec(args("twice", [scalarDouble()]));
    compileSpecC(args("mask", [matrix]));
    const inc = compileSpec(args("inc", [scalarDouble()]));
    expect(lowerer.specializations.size).toBe(3);
    expect(twice.source).toContain(twice.cName);
    expect(inc.source).not.toContain(twice.cName);
    expect(inc.source).not.toContain("mask__");
  });

  it("does not let a spec only C can emit break later JS compiles", () => {
    const { args } = session();
    compileSpecC(args("mask", [matrix]));
    expect(() => compileSpec(args("mask", [matrix]))).toThrow(
      /don't share static shape/
    );
    expect(() => compileSpec(args("inc", [scalarDouble()]))).not.toThrow();
  });

  it("emits the same C for specs that differ only in argument types", () => {
    const { args } = session();
    const col = tensorDoubleFromDims([
      { kind: "unknown" },
      { kind: "exact", value: 1 },
    ]);
    // `inc` enters the spec cache before `twice`, so call order (not cache
    // order) must decide the order of the emitted functions.
    compileSpecC(args("inc", [col]));
    const a = compileSpecC(args("twice", [col]));
    const b = compileSpecC(args("twice", [scalarDouble("positive")]));
    const c = compileSpecC(args("twice", [scalarDouble()]));
    expect(a.signature.cName).toBe("_jit_fn0_twice__host");
    expect(stripCComments(a.source)).not.toMatch(/__[0-9a-f]{8}\b/);
    expect(b.cName).not.toBe(c.cName);
    expect(stripCComments(b.source)).toBe(stripCComments(c.source));
    expect(b.signature.cName).toBe(c.signature.cName);
  });
});

describe("C source passes", () => {
  const src = [
    "/* f__1 */ static int f__1(int x); // f__1",
    `void g(void) { puts("f__1 // /* \\" f__1"); char c = '"'; f__1(1e5); }`,
  ].join("\n");

  it("renames identifiers only outside comments and literals", () => {
    expect(renameCIdentifiers(src, new Map([["f__1", "h"]]))).toBe(
      [
        "/* f__1 */ static int h(int x); // f__1",
        `void g(void) { puts("f__1 // /* \\" f__1"); char c = '"'; h(1e5); }`,
      ].join("\n")
    );
  });

  it("strips comments", () => {
    expect(stripCComments(src)).toBe(
      [
        " static int f__1(int x); ",
        `void g(void) { puts("f__1 // /* \\" f__1"); char c = '"'; f__1(1e5); }`,
      ].join("\n")
    );
  });
});

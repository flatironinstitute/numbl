import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { loadJsUserFunctions } from "../numbl-core/jsUserFunctions.js";
import { parseMFile } from "../numbl-core/parser/index.js";

// ── jsUserFunctions ──────────────────────────────────────────────────

describe("jsUserFunctions", () => {
  it("loads a simple JS user function", () => {
    const jsFiles = [
      {
        name: "myadd.js",
        source: `register({
          apply: (args) => args[0] + args[1]
        });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.has("myadd")).toBe(true);
    expect(fns.get("myadd")!.length).toBe(1);
  });

  it("loads a JS function with custom check", () => {
    const jsFiles = [
      {
        name: "double_it.js",
        source: `register({
          check: (argTypes, nargout) => ({ outputTypes: [{ kind: 'Unknown' }] }),
          apply: (args) => args[0] * 2
        });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.has("double_it")).toBe(true);
  });

  it("throws if no register() call", () => {
    const jsFiles = [
      {
        name: "empty.js",
        source: `// no register call`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/must call register/);
  });

  it("throws if apply is not a function", () => {
    const jsFiles = [
      {
        name: "bad.js",
        source: `register({ apply: 42 });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/apply function/);
  });

  it("throws on JS syntax errors", () => {
    const jsFiles = [
      {
        name: "broken.js",
        source: `register({{{`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/Error loading/);
  });

  it("derives function name from path", () => {
    const jsFiles = [
      {
        name: "/some/path/myFunc.js",
        source: `register({ apply: (args) => 0 });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.has("myFunc")).toBe(true);
  });
});

// ── Parser edge cases ────────────────────────────────────────────────

describe("parser edge cases", () => {
  it("parses function without end keyword", () => {
    const code = `function y = foo(x)\n  y = x + 1;\n`;
    const ast = parseMFile(code, "test.m");
    expect(ast).toBeDefined();
  });

  it("parses multiple functions without end keywords", () => {
    const code = `function y = foo(x)\n  y = bar(x);\nfunction y = bar(x)\n  y = x * 2;\n`;
    const ast = parseMFile(code, "test.m");
    expect(ast).toBeDefined();
  });

  it("parses function with end keyword normally", () => {
    const code = `function y = foo(x)\n  y = x + 1;\nend\n`;
    const ast = parseMFile(code, "test.m");
    expect(ast).toBeDefined();
  });

  it("throws on invalid tokens", () => {
    expect(() => parseMFile("x = @#$;", "test.m")).toThrow();
  });

  it("parses functions with blocks but no end", () => {
    const code = `function y = foo(x)\n  if x > 0\n    y = x;\n  else\n    y = -x;\n  end\n`;
    const ast = parseMFile(code, "test.m");
    expect(ast).toBeDefined();
  });
});

// ── compare.ts via isequal ───────────────────────────────────────────

describe("compare / isequal", () => {
  it("compares numbers", () => {
    const result = executeCode("x = isequal(1, 1);");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("compares different numbers", () => {
    const result = executeCode("x = isequal(1, 2);");
    expect(result.variableValues["x"]).toBe(false);
  });

  it("compares logicals", () => {
    const result = executeCode("x = isequal(true, true);");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("compares char arrays", () => {
    const result = executeCode("x = isequal('abc', 'abc');");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("compares tensors", () => {
    const result = executeCode("x = isequal([1 2 3], [1 2 3]);");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("compares cells", () => {
    const result = executeCode("x = isequal({1, 2}, {1, 2});");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("compares complex numbers", () => {
    const result = executeCode("x = isequal(1+2i, 1+2i);");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("returns false for number vs string", () => {
    const result = executeCode("x = isequal(1, 'a');");
    expect(result.variableValues["x"]).toBe(false);
  });

  it("compares complex tensors", () => {
    const result = executeCode("x = isequal([1+2i, 3], [1+2i, 3]);");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("returns false for tensors with different imag", () => {
    const result = executeCode("x = isequal([1+2i, 3], [1+3i, 3]);");
    expect(result.variableValues["x"]).toBe(false);
  });
});

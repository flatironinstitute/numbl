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

  it("does not provide wasm without directive (no fallback by name)", () => {
    const jsFiles = [
      {
        name: "myfunc.js",
        source: `register({ apply: () => wasm });`,
      },
    ];
    // Even though there's a matching wasm file by name, wasm should be undefined
    // without a directive
    const wasmFiles = [
      { name: "myfunc.wasm", source: "", data: new Uint8Array([]) },
    ];
    const fns = loadJsUserFunctions(jsFiles, wasmFiles);
    const branch = fns.get("myfunc")![0];
    expect(branch.apply([], 1)).toBeUndefined();
  });

  it("provides wasm when directive is present and wasm file exists", () => {
    // Build a minimal valid wasm module (empty module: magic + version + no sections)
    const wasmBytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ]);
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `// wasm: mymod\nregister({ apply: () => (wasm !== undefined) });`,
      },
    ];
    const wasmFiles = [
      { name: "/path/to/mymod.wasm", source: "", data: wasmBytes },
    ];
    const fns = loadJsUserFunctions(jsFiles, wasmFiles);
    const branch = fns.get("myfunc")![0];
    expect(branch.apply([], 1)).toBe(true);
  });

  it("passes native as undefined when no directive", () => {
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `register({ apply: () => (typeof native === 'undefined') });`,
      },
    ];
    const mockBridge = { load: () => ({ fake: true }) };
    const fns = loadJsUserFunctions(jsFiles, [], mockBridge);
    const branch = fns.get("myfunc")![0];
    expect(branch.apply([], 1)).toBe(true);
  });

  it("passes native library when directive and bridge are present", () => {
    const mockLib = { myFunction: () => 42 };
    const mockBridge = { load: () => mockLib };
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `// native: mylib\nregister({ apply: () => native });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles, [], mockBridge);
    const branch = fns.get("myfunc")![0];
    expect(branch.apply([], 1)).toBe(mockLib);
  });

  it("leaves native undefined when bridge load fails", () => {
    const mockBridge = {
      load: () => {
        throw new Error("not found");
      },
    };
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `// native: mylib\nregister({ apply: () => (typeof native === 'undefined') });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles, [], mockBridge);
    const branch = fns.get("myfunc")![0];
    expect(branch.apply([], 1)).toBe(true);
  });

  it("parses both wasm and native directives", () => {
    const wasmBytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ]);
    const mockLib = { fn: () => 1 };
    const mockBridge = { load: () => mockLib };
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `// wasm: mymod\n// native: mylib\nregister({ apply: () => [wasm !== undefined, native !== undefined] });`,
      },
    ];
    const wasmFiles = [
      { name: "/path/to/mymod.wasm", source: "", data: wasmBytes },
    ];
    const fns = loadJsUserFunctions(jsFiles, wasmFiles, mockBridge);
    const branch = fns.get("myfunc")![0];
    const result = branch.apply([], 1);
    expect(result).toEqual([true, true]);
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

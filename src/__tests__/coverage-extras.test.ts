import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { loadJsUserFunctions } from "../numbl-core/jsUserFunctions.js";
import type { RuntimeValue } from "../numbl-core/runtime/types.js";
import { parseMFile } from "../numbl-core/parser/index.js";

// ── jsUserFunctions ──────────────────────────────────────────────────

/** Helper: resolve an IBuiltin and call its apply with given args. */
function callIBuiltin(
  ibs: ReturnType<typeof loadJsUserFunctions>,
  name: string,
  args: unknown[],
  nargout = 1
) {
  const ib = ibs.find(b => b.name === name);
  if (!ib) throw new Error(`IBuiltin '${name}' not found`);
  const res = ib.resolve(
    args.map(() => ({ kind: "number" as const })),
    nargout
  );
  if (!res) throw new Error(`resolve returned null for '${name}'`);
  return res.apply(args as RuntimeValue[], nargout);
}

describe("jsUserFunctions", () => {
  it("loads a simple JS user function", () => {
    const jsFiles = [
      {
        name: "myadd.js",
        source: `register({
          resolve: (argTypes, nargout) => ({
            outputTypes: [{ kind: 'number' }],
            apply: (args) => args[0] + args[1]
          })
        });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.find(b => b.name === "myadd")).toBeDefined();
    expect(callIBuiltin(fns, "myadd", [3, 4])).toBe(7);
  });

  it("imports a library file via importJS", () => {
    const jsFiles = [
      {
        name: "_helpers.js",
        source: `return { add: function(a, b) { return a + b; } };`,
      },
      {
        name: "usehelper.js",
        source: `var H = importJS("_helpers");
register({ resolve: () => ({ outputTypes: [{ kind: 'number' }], apply: (args) => H.add(args[0], args[1]) }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.find(b => b.name === "usehelper")).toBeDefined();
    expect(fns.find(b => b.name === "_helpers")).toBeUndefined();
    expect(callIBuiltin(fns, "usehelper", [3, 4])).toBe(7);
  });

  it("caches library — executes once for multiple imports", () => {
    const jsFiles = [
      {
        name: "_counter.js",
        source: `var c = { n: 0 }; c.n++; return c;`,
      },
      {
        name: "a.js",
        source: `var C = importJS("_counter");
register({ resolve: () => ({ outputTypes: [{ kind: 'number' }], apply: () => C.n }) });`,
      },
      {
        name: "b.js",
        source: `var C = importJS("_counter");
register({ resolve: () => ({ outputTypes: [{ kind: 'number' }], apply: () => C.n }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(callIBuiltin(fns, "a", [])).toBe(1);
    expect(callIBuiltin(fns, "b", [])).toBe(1);
  });

  it("throws on importJS for nonexistent library", () => {
    const jsFiles = [
      {
        name: "bad.js",
        source: `importJS("_nope"); register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/not found/);
  });

  it("throws on circular dependency", () => {
    const jsFiles = [
      {
        name: "_a.js",
        source: `var b = importJS("_b"); return { x: 1 };`,
      },
      {
        name: "_b.js",
        source: `var a = importJS("_a"); return { y: 2 };`,
      },
      {
        name: "trigger.js",
        source: `importJS("_a"); register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/Circular dependency/);
  });

  it("throws when library calls register()", () => {
    const jsFiles = [
      {
        name: "_badlib.js",
        source: `register({ resolve: () => null }); return {};`,
      },
      {
        name: "trigger.js",
        source: `importJS("_badlib"); register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(
      /must not call register/
    );
  });

  it("supports library importing another library", () => {
    const jsFiles = [
      {
        name: "_base.js",
        source: `return { mul: function(a, b) { return a * b; } };`,
      },
      {
        name: "_derived.js",
        source: `var B = importJS("_base");
return { square: function(x) { return B.mul(x, x); } };`,
      },
      {
        name: "usederived.js",
        source: `var D = importJS("_derived");
register({ resolve: () => ({ outputTypes: [{ kind: 'number' }], apply: (args) => D.square(args[0]) }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(callIBuiltin(fns, "usederived", [5])).toBe(25);
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

  it("throws if resolve is not a function", () => {
    const jsFiles = [
      {
        name: "bad.js",
        source: `register({ resolve: 42 });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/resolve function/);
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
        source: `register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.find(b => b.name === "myFunc")).toBeDefined();
  });

  it("does not provide wasm without directive (no fallback by name)", () => {
    const jsFiles = [
      {
        name: "myfunc.js",
        source: `register({ resolve: () => ({ outputTypes: [], apply: () => wasm }) });`,
      },
    ];
    const wasmFiles = [
      { name: "myfunc.wasm", source: "", data: new Uint8Array([]) },
    ];
    const fns = loadJsUserFunctions(jsFiles, wasmFiles);
    expect(callIBuiltin(fns, "myfunc", [])).toBeUndefined();
  });

  it("provides wasm when directive is present and wasm file exists", () => {
    const wasmBytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ]);
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `// wasm: mymod\nregister({ resolve: () => ({ outputTypes: [], apply: () => (wasm !== undefined) }) });`,
      },
    ];
    const wasmFiles = [
      { name: "/path/to/mymod.wasm", source: "", data: wasmBytes },
    ];
    const fns = loadJsUserFunctions(jsFiles, wasmFiles);
    expect(callIBuiltin(fns, "myfunc", [])).toBe(true);
  });

  it("passes native as undefined when no directive", () => {
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `register({ resolve: () => ({ outputTypes: [], apply: () => (typeof native === 'undefined') }) });`,
      },
    ];
    const mockBridge = { load: () => ({ fake: true }) };
    const fns = loadJsUserFunctions(jsFiles, [], mockBridge);
    expect(callIBuiltin(fns, "myfunc", [])).toBe(true);
  });

  it("passes native library when directive and bridge are present", () => {
    const mockLib = { myFunction: () => 42 };
    const mockBridge = { load: () => mockLib };
    const jsFiles = [
      {
        name: "/path/to/myfunc.js",
        source: `// native: mylib\nregister({ resolve: () => ({ outputTypes: [], apply: () => native }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles, [], mockBridge);
    expect(callIBuiltin(fns, "myfunc", [])).toBe(mockLib);
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
        source: `// native: mylib\nregister({ resolve: () => ({ outputTypes: [], apply: () => (typeof native === 'undefined') }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles, [], mockBridge);
    expect(callIBuiltin(fns, "myfunc", [])).toBe(true);
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
        source: `// wasm: mymod\n// native: mylib\nregister({ resolve: () => ({ outputTypes: [], apply: () => [wasm !== undefined, native !== undefined] }) });`,
      },
    ];
    const wasmFiles = [
      { name: "/path/to/mymod.wasm", source: "", data: wasmBytes },
    ];
    const fns = loadJsUserFunctions(jsFiles, wasmFiles, mockBridge);
    const result = callIBuiltin(fns, "myfunc", []);
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

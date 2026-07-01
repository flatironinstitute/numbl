import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { loadJsUserFunctions } from "../numbl-core/jsUserFunctions.js";
import type { RuntimeValue } from "../numbl-core/runtime/types.js";
import { parseMFile } from "../numbl-core/parser/index.js";

// ── jsUserFunctions ──────────────────────────────────────────────────

/** Helper: resolve an IBuiltin and call its apply with given args. */
function callIBuiltin(
  loaded: ReturnType<typeof loadJsUserFunctions>,
  name: string,
  args: unknown[],
  nargout = 1
) {
  const entry = loaded.find(b => b.name === name);
  if (!entry) throw new Error(`IBuiltin '${name}' not found`);
  const ib = entry.builtin;
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
        name: "myadd.numbl.js",
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
        name: "_helpers.numbl.js",
        source: `return { add: function(a, b) { return a + b; } };`,
      },
      {
        name: "usehelper.numbl.js",
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
        name: "_counter.numbl.js",
        source: `var c = { n: 0 }; c.n++; return c;`,
      },
      {
        name: "a.numbl.js",
        source: `var C = importJS("_counter");
register({ resolve: () => ({ outputTypes: [{ kind: 'number' }], apply: () => C.n }) });`,
      },
      {
        name: "b.numbl.js",
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
        name: "bad.numbl.js",
        source: `importJS("_nope"); register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/not found/);
  });

  it("throws on circular dependency", () => {
    const jsFiles = [
      {
        name: "_a.numbl.js",
        source: `var b = importJS("_b"); return { x: 1 };`,
      },
      {
        name: "_b.numbl.js",
        source: `var a = importJS("_a"); return { y: 2 };`,
      },
      {
        name: "trigger.numbl.js",
        source: `importJS("_a"); register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/Circular dependency/);
  });

  it("throws when library calls register()", () => {
    const jsFiles = [
      {
        name: "_badlib.numbl.js",
        source: `register({ resolve: () => null }); return {};`,
      },
      {
        name: "trigger.numbl.js",
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
        name: "_base.numbl.js",
        source: `return { mul: function(a, b) { return a * b; } };`,
      },
      {
        name: "_derived.numbl.js",
        source: `var B = importJS("_base");
return { square: function(x) { return B.mul(x, x); } };`,
      },
      {
        name: "usederived.numbl.js",
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
        name: "empty.numbl.js",
        source: `// no register call`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/must call register/);
  });

  it("throws if resolve is not a function", () => {
    const jsFiles = [
      {
        name: "bad.numbl.js",
        source: `register({ resolve: 42 });`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/resolve function/);
  });

  it("throws on JS syntax errors", () => {
    const jsFiles = [
      {
        name: "broken.numbl.js",
        source: `register({{{`,
      },
    ];
    expect(() => loadJsUserFunctions(jsFiles)).toThrow(/Error loading/);
  });

  it("derives function name from path", () => {
    const jsFiles = [
      {
        name: "/some/path/myFunc.numbl.js",
        source: `register({ resolve: () => ({ outputTypes: [], apply: () => 0 }) });`,
      },
    ];
    const fns = loadJsUserFunctions(jsFiles);
    expect(fns.find(b => b.name === "myFunc")).toBeDefined();
  });

  it("does not provide wasm without directive (no fallback by name)", () => {
    const jsFiles = [
      {
        name: "myfunc.numbl.js",
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
        name: "/path/to/myfunc.numbl.js",
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
        name: "/path/to/myfunc.numbl.js",
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
        name: "/path/to/myfunc.numbl.js",
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
        name: "/path/to/myfunc.numbl.js",
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
        name: "/path/to/myfunc.numbl.js",
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

// ── WASM function-handle callbacks ───────────────────────────────────────
//
// Exercises the full chain: a numbl function handle passed into a `.numbl.js`
// user function → registered with `wasm.callbacks.add` → handed to WASM as an
// integer id → invoked from inside WASM via the `env.numbl_cb_d` import →
// `callHandle` → interpreter → result back through WASM.
//
// `CB_WASM` is a hand-assembled standalone module (no emcc dependency):
//   import  env.numbl_cb_d : (i32, f64) -> f64
//   export  callcb(id: i32, x: f64) -> f64 { numbl_cb_d(id,x) + numbl_cb_d(id,x) }
// i.e. callcb returns 2 * handle(x).
describe("wasm function-handle callbacks", () => {
  // prettier-ignore
  const CB_WASM = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    // type section: one func type (i32, f64) -> f64
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7c, 0x01, 0x7c,
    // import section: env.numbl_cb_d : type 0
    0x02, 0x12, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x0a, 0x6e, 0x75,
    0x6d, 0x62, 0x6c, 0x5f, 0x63, 0x62, 0x5f, 0x64, 0x00, 0x00,
    // function section: one func of type 0
    0x03, 0x02, 0x01, 0x00,
    // export section: "callcb" -> func index 1
    0x07, 0x0a, 0x01, 0x06, 0x63, 0x61, 0x6c, 0x6c, 0x63, 0x62, 0x00, 0x01,
    // code section: callcb = numbl_cb_d(id,x) + numbl_cb_d(id,x)
    0x0a, 0x11, 0x01, 0x0f, 0x00,
    0x20, 0x00, 0x20, 0x01, 0x10, 0x00, // numbl_cb_d(id, x)
    0x20, 0x00, 0x20, 0x01, 0x10, 0x00, // numbl_cb_d(id, x)
    0xa0, 0x0b,                         // f64.add ; end
  ]);

  const DOUBLECB_SRC = `// wasm: cbmod
register({
  resolve: function () {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        var f = args[0];
        var x = toNumber(args[1]);
        var id = wasm.callbacks.add(function (xx) {
          return toNumber(callHandle(f, [xx]));
        });
        try {
          return wasm.exports.callcb(id, x);
        } finally {
          wasm.callbacks.remove(id);
        }
      },
    };
  },
});`;

  const wasmFile = { name: "cbmod.wasm", source: "", data: CB_WASM };

  it("invokes an anonymous handle from inside WASM", () => {
    const result = executeCode("r = doublecb(@(x) x + 10, 5);", {}, [
      { name: "doublecb.numbl.js", source: DOUBLECB_SRC },
      wasmFile,
    ]);
    // handle(5) = 15; callcb doubles it.
    expect(result.variableValues["r"]).toBe(30);
  });

  it("invokes a builtin handle from inside WASM", () => {
    const result = executeCode("r = doublecb(@sqrt, 16);", {}, [
      { name: "doublecb.numbl.js", source: DOUBLECB_SRC },
      wasmFile,
    ]);
    // sqrt(16) = 4; doubled = 8.
    expect(result.variableValues["r"]).toBe(8);
  });

  it("invokes a closure capturing a workspace variable", () => {
    const result = executeCode("k = 7; r = doublecb(@(x) k * x, 3);", {}, [
      { name: "doublecb.numbl.js", source: DOUBLECB_SRC },
      wasmFile,
    ]);
    // k*x = 21; doubled = 42.
    expect(result.variableValues["r"]).toBe(42);
  });

  it("throws when WASM calls back with an unregistered id", () => {
    const badSrc = `// wasm: cbmod
register({
  resolve: function () {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        return wasm.exports.callcb(999, toNumber(args[0]));
      },
    };
  },
});`;
    expect(() =>
      executeCode("r = badcb(1);", {}, [
        { name: "badcb.numbl.js", source: badSrc },
        wasmFile,
      ])
    ).toThrow(/no callback registered for id 999/);
  });
});

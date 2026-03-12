import { describe, it, expect } from "vitest";
import {
  RuntimeError,
  offsetToLine,
  offsetToColumn,
  extractSnippet,
  buildLineTable,
  offsetToLineFast,
} from "../numbl-core/runtime/error.js";
import { toNumber, toBool, toString } from "../numbl-core/runtime/convert.js";
import { displayValue } from "../numbl-core/runtime/display.js";
import { valuesAreEqual } from "../numbl-core/runtime/compare.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import {
  FloatXArray,
  type RuntimeValue,
  type RuntimeClassInstance,
  type RuntimeFunction,
} from "../numbl-core/runtime/types.js";

// ── RuntimeError ────────────────────────────────────────────────────

describe("RuntimeError", () => {
  it("withSpan sets span and file", () => {
    const err = new RuntimeError("test");
    err.withSpan({ file: "foo.m", start: 0, end: 5 });
    expect(err.span).toEqual({ file: "foo.m", start: 0, end: 5 });
    expect(err.file).toBe("foo.m");
  });

  it("withContext enriches with line/column/snippet", () => {
    const err = new RuntimeError("oops", { file: "foo.m", start: 6, end: 7 });
    const sources = new Map([["foo.m", "line1\nline2\nline3"]]);
    err.withContext(sources);
    expect(err.line).toBe(2);
    expect(err.column).toBe(1);
    expect(err.snippet).toBeDefined();
  });

  it("withContext does nothing without span", () => {
    const err = new RuntimeError("oops");
    const sources = new Map([["foo.m", "code"]]);
    err.withContext(sources);
    expect(err.line).toBeNull();
  });

  it("withContext does nothing if file not in sources", () => {
    const err = new RuntimeError("oops", {
      file: "bar.m",
      start: 0,
      end: 1,
    });
    const sources = new Map([["foo.m", "code"]]);
    err.withContext(sources);
    expect(err.line).toBeNull();
  });

  it("toString with file and line", () => {
    const err = new RuntimeError("bad");
    err.file = "test.m";
    err.line = 5;
    err.column = 3;
    const s = err.toString();
    expect(s).toContain("test.m:5:3");
    expect(s).toContain("bad");
  });

  it("toString with line only (no file)", () => {
    const err = new RuntimeError("bad");
    err.line = 5;
    const s = err.toString();
    expect(s).toContain("line 5");
  });

  it("toString with snippet", () => {
    const err = new RuntimeError("bad");
    err.snippet = "> 1 | x = 1;";
    const s = err.toString();
    expect(s).toContain("> 1 | x = 1;");
  });

  it("toString with callStack", () => {
    const err = new RuntimeError("bad");
    err.file = "main.m";
    err.line = 10;
    err.callStack = [
      { name: "main", callerFile: null, callerLine: 0 },
      { name: "helper", callerFile: "main.m", callerLine: 5 },
    ];
    const s = err.toString();
    expect(s).toContain("Call stack");
    expect(s).toContain("helper");
    expect(s).toContain("main");
  });

  it("toString callStack with callerFile and callerLine", () => {
    const err = new RuntimeError("bad");
    err.file = "inner.m";
    err.line = 3;
    err.callStack = [
      { name: "outer", callerFile: null, callerLine: 0 },
      { name: "middle", callerFile: "outer.m", callerLine: 10 },
      { name: "inner", callerFile: "middle.m", callerLine: 20 },
    ];
    const s = err.toString();
    expect(s).toContain("inner.m:3");
    expect(s).toContain("middle.m:20");
    expect(s).toContain("outer.m:10");
  });

  it("toString callStack with unknown locations", () => {
    const err = new RuntimeError("bad");
    err.callStack = [{ name: "fn", callerFile: null, callerLine: 0 }];
    const s = err.toString();
    expect(s).toContain("unknown");
  });

  it("toString callStack innermost without file/line", () => {
    const err = new RuntimeError("bad");
    err.callStack = [{ name: "fn", callerFile: null, callerLine: 0 }];
    const s = err.toString();
    expect(s).toContain("at fn (unknown)");
  });

  it("toString callStack innermost with line only", () => {
    const err = new RuntimeError("bad");
    err.line = 7;
    err.callStack = [{ name: "fn", callerFile: null, callerLine: 0 }];
    const s = err.toString();
    expect(s).toContain("at fn (line 7)");
  });

  it("toString callStack outer frame with callerLine only", () => {
    const err = new RuntimeError("bad");
    err.file = "test.m";
    err.line = 1;
    err.callStack = [
      { name: "outer", callerFile: null, callerLine: 0 },
      { name: "inner", callerFile: null, callerLine: 5 },
    ];
    const s = err.toString();
    expect(s).toContain("line 5");
  });
});

// ── Source location utilities ────────────────────────────────────────

describe("offsetToLine", () => {
  it("returns 1 for start of file", () => {
    expect(offsetToLine("abc\ndef", 0)).toBe(1);
  });

  it("returns 2 after first newline", () => {
    expect(offsetToLine("abc\ndef", 4)).toBe(2);
  });
});

describe("offsetToColumn", () => {
  it("returns correct column on first line", () => {
    expect(offsetToColumn("abcdef", 3)).toBe(4);
  });

  it("returns 1 at start of second line", () => {
    expect(offsetToColumn("abc\ndef", 4)).toBe(1);
  });
});

describe("extractSnippet", () => {
  it("produces snippet with context lines", () => {
    const source = "line1\nline2\nline3\nline4\nline5";
    const snippet = extractSnippet(source, 12); // in line3
    expect(snippet).toContain("line2");
    expect(snippet).toContain("line3");
    expect(snippet).toContain("line4");
    expect(snippet).toContain(">");
    expect(snippet).toContain("^");
  });
});

describe("buildLineTable / offsetToLineFast", () => {
  it("builds correct line table", () => {
    const table = buildLineTable("a\nb\nc");
    expect(table).toEqual([1, 3]);
  });

  it("offsetToLineFast returns correct line", () => {
    const table = buildLineTable("a\nb\nc");
    expect(offsetToLineFast(table, 0)).toBe(1);
    expect(offsetToLineFast(table, 2)).toBe(2);
    expect(offsetToLineFast(table, 4)).toBe(3);
  });
});

// ── convert.ts ──────────────────────────────────────────────────────

describe("toNumber", () => {
  it("converts char to number", () => {
    expect(toNumber(RTV.char("A"))).toBe(65);
  });

  it("throws for multi-char", () => {
    expect(() => toNumber(RTV.char("AB"))).toThrow("multi-char");
  });

  it("converts 1x1 tensor to number", () => {
    expect(toNumber(RTV.tensor(new FloatXArray([42]), [1, 1]))).toBe(42);
  });

  it("throws for non-scalar tensor", () => {
    expect(() => toNumber(RTV.tensor(new FloatXArray([1, 2]), [1, 2]))).toThrow(
      "non-scalar"
    );
  });

  it("converts complex with zero imag", () => {
    expect(toNumber(RTV.complex(5, 0))).toBe(5);
  });

  it("throws for complex with nonzero imag", () => {
    expect(() => toNumber(RTV.complex(1, 2))).toThrow("Complex");
  });

  it("converts string number", () => {
    const s: RuntimeValue = "42";
    expect(toNumber(s)).toBe(42);
  });

  it("throws for non-numeric string", () => {
    const s: RuntimeValue = "abc";
    expect(() => toNumber(s)).toThrow("Cannot convert");
  });
});

describe("toBool", () => {
  it("tensor with all nonzero is truthy", () => {
    expect(toBool(RTV.tensor(new FloatXArray([1, 2, 3]), [1, 3]))).toBe(true);
  });

  it("tensor with a zero is falsy", () => {
    expect(toBool(RTV.tensor(new FloatXArray([1, 0, 3]), [1, 3]))).toBe(false);
  });

  it("empty tensor is falsy", () => {
    expect(toBool(RTV.tensor(new FloatXArray([]), [0, 0]))).toBe(false);
  });

  it("complex tensor all nonzero is truthy", () => {
    const t = RTV.tensor(new FloatXArray([0, 0]), [1, 2]);
    t.imag = new FloatXArray([1, 1]);
    expect(toBool(t)).toBe(true);
  });

  it("complex tensor with both zero is falsy", () => {
    const t = RTV.tensor(new FloatXArray([0, 1]), [1, 2]);
    t.imag = new FloatXArray([0, 0]);
    expect(toBool(t)).toBe(false);
  });

  it("char is truthy", () => {
    expect(toBool(RTV.char("a"))).toBe(true);
  });

  it("empty char is falsy", () => {
    expect(toBool(RTV.char(""))).toBe(false);
  });

  it("complex number truthy", () => {
    expect(toBool(RTV.complex(0, 1))).toBe(true);
  });

  it("complex number zero is falsy", () => {
    expect(toBool(RTV.complex(0, 0))).toBe(false);
  });

  it("string truthy", () => {
    const s: RuntimeValue = "hello";
    expect(toBool(s)).toBe(true);
  });

  it("empty string falsy", () => {
    const s: RuntimeValue = "";
    expect(toBool(s)).toBe(false);
  });
});

describe("toString", () => {
  it("converts logical true", () => {
    const v: RuntimeValue = true;
    expect(toString(v)).toBe("1");
  });

  it("converts logical false", () => {
    const v: RuntimeValue = false;
    expect(toString(v)).toBe("0");
  });

  it("converts number", () => {
    expect(toString(42)).toBe("42");
  });

  it("converts char", () => {
    expect(toString(RTV.char("hello"))).toBe("hello");
  });

  it("throws for tensor", () => {
    expect(() => toString(RTV.tensor(new FloatXArray([1]), [1, 1]))).toThrow(
      "Cannot convert"
    );
  });
});

// ── display.ts ──────────────────────────────────────────────────────

describe("displayValue", () => {
  it("displays logical true as 1", () => {
    const v: RuntimeValue = true;
    expect(displayValue(v)).toBe("1");
  });

  it("displays logical false as 0", () => {
    const v: RuntimeValue = false;
    expect(displayValue(v)).toBe("0");
  });

  it("displays empty tensor as []", () => {
    expect(displayValue(RTV.tensor(new FloatXArray([]), [0, 0]))).toBe("[]");
  });

  it("displays scalar tensor", () => {
    expect(displayValue(RTV.tensor(new FloatXArray([42]), [1, 1]))).toBe("42");
  });

  it("displays complex scalar tensor", () => {
    const t = RTV.tensor(new FloatXArray([3]), [1, 1]);
    t.imag = new FloatXArray([4]);
    const s = displayValue(t);
    expect(s).toContain("3");
    expect(s).toContain("4");
  });

  it("displays cell", () => {
    const c = RTV.cell([1, RTV.char("hi")], [1, 2]);
    const s = displayValue(c);
    expect(s).toContain("1");
    expect(s).toContain("'hi'");
  });

  it("displays struct", () => {
    const st = RTV.struct(
      new Map([
        ["x", 1],
        ["y", 2],
      ])
    );
    const s = displayValue(st);
    expect(s).toContain("x:");
    expect(s).toContain("y:");
  });

  it("displays function handle", () => {
    const fn: RuntimeFunction = {
      kind: "function",
      name: "sin",
      captures: [],
      impl: "builtin",
    };
    expect(displayValue(fn)).toBe("@sin");
  });

  it("displays complex number", () => {
    expect(displayValue(RTV.complex(3, 4))).toContain("3");
    expect(displayValue(RTV.complex(3, 4))).toContain("4");
  });

  it("displays complex number with negative imag", () => {
    const s = displayValue(RTV.complex(3, -4));
    expect(s).toContain("3");
    expect(s).toContain("4");
  });

  it("displays pure imaginary", () => {
    const s = displayValue(RTV.complex(0, 5));
    expect(s).toContain("5");
  });

  it("displays real complex (imag=0)", () => {
    expect(displayValue(RTV.complex(3, 0))).toBe("3");
  });

  it("displays class instance", () => {
    const inst: RuntimeClassInstance = {
      kind: "class_instance",
      className: "MyClass",
      fields: new Map([["val", 42]]),
      isHandleClass: false,
    };
    const s = displayValue(inst);
    expect(s).toContain("MyClass");
    expect(s).toContain("val:");
  });

  it("displays 3D tensor page by page", () => {
    const data = new FloatXArray([1, 2, 3, 4, 5, 6, 7, 8]);
    const t = RTV.tensor(data, [2, 2, 2]);
    const s = displayValue(t);
    expect(s).toContain("(:,:,1)");
    expect(s).toContain("(:,:,2)");
  });

  it("displays multi-row char array", () => {
    const c = { kind: "char" as const, value: "abcdef", shape: [2, 3] };
    const s = displayValue(c);
    expect(s).toBe("abc\ndef");
  });
});

// ── compare.ts additional paths ─────────────────────────────────────

describe("valuesAreEqual additional", () => {
  it("compares different kind objects as false", () => {
    const a = RTV.char("x");
    const b = RTV.complex(1, 2);
    expect(valuesAreEqual(a, b)).toBe(false);
  });

  it("compares cell with different lengths", () => {
    const a = RTV.cell([1, 2], [1, 2]);
    const b = RTV.cell([1], [1, 1]);
    expect(valuesAreEqual(a, b)).toBe(false);
  });

  it("compares cells with different content", () => {
    const a = RTV.cell([1, 2], [1, 2]);
    const b = RTV.cell([1, 3], [1, 2]);
    expect(valuesAreEqual(a, b)).toBe(false);
  });

  it("compares tensors with different imag presence", () => {
    const a = RTV.tensor(new FloatXArray([1]), [1, 1]);
    const b = RTV.tensor(new FloatXArray([1]), [1, 1]);
    a.imag = new FloatXArray([0]);
    expect(valuesAreEqual(a, b)).toBe(false);
  });

  it("compares tensors with different imag values", () => {
    const a = RTV.tensor(new FloatXArray([1]), [1, 1]);
    const b = RTV.tensor(new FloatXArray([1]), [1, 1]);
    a.imag = new FloatXArray([1]);
    b.imag = new FloatXArray([2]);
    expect(valuesAreEqual(a, b)).toBe(false);
  });

  it("compares structs by reference", () => {
    const s1 = RTV.struct(new Map([["a", 1]]));
    const s2 = RTV.struct(new Map([["a", 1]]));
    expect(valuesAreEqual(s1, s2)).toBe(false);
    expect(valuesAreEqual(s1, s1)).toBe(true);
  });

  it("primitive vs object returns false", () => {
    expect(valuesAreEqual(42, RTV.char("x"))).toBe(false);
    const v: RuntimeValue = true;
    expect(valuesAreEqual(v, RTV.char("x"))).toBe(false);
  });
});

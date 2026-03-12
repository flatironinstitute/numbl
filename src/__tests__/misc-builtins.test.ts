import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor, isRuntimeCell } from "../numbl-core/runtime/types.js";

function num(code: string, varName: string): number {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "number") throw new Error(`${varName} is not a number`);
  return v;
}

function bool(code: string, varName: string): boolean {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "boolean") throw new Error(`${varName} is not a boolean`);
  return v;
}

function tensorData(code: string, varName: string): number[] {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return Array.from(v.data);
}

function tensorShape(code: string, varName: string): number[] {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return v.shape;
}

// ── logical() ─────────────────────────────────────────────────────────

describe("logical conversion", () => {
  it("logical(1) is true", () => {
    expect(bool("x = logical(1);", "x")).toBe(true);
  });

  it("logical(0) is false", () => {
    expect(bool("x = logical(0);", "x")).toBe(false);
  });

  it("logical(tensor) converts element-wise", () => {
    const result = executeCode("v = logical([0, 1, 2, 0]);");
    const v = result.variableValues["v"];
    expect(isRuntimeTensor(v)).toBe(true);
    if (isRuntimeTensor(v)) {
      expect(Array.from(v.data)).toEqual([0, 1, 1, 0]);
    }
  });
});

// ── cell() ────────────────────────────────────────────────────────────

describe("cell constructor", () => {
  it("cell(2,3) creates 2x3 cell array", () => {
    const result = executeCode("c = cell(2, 3);");
    const c = result.variableValues["c"];
    expect(c).toBeDefined();
    expect((c as { kind: string }).kind).toBe("cell");
    expect((c as { shape: number[] }).shape).toEqual([2, 3]);
  });

  it("cell(0) creates empty cell", () => {
    const result = executeCode("c = cell(0);");
    const c = result.variableValues["c"];
    expect(c).toBeDefined();
  });

  it("cell(3) creates 3x3 cell", () => {
    const result = executeCode("c = cell(3);");
    const c = result.variableValues["c"];
    expect((c as { shape: number[] }).shape).toEqual([3, 3]);
  });
});

// ── struct() ──────────────────────────────────────────────────────────

describe("struct constructor", () => {
  it("struct() creates empty struct", () => {
    const result = executeCode("s = struct();");
    const s = result.variableValues["s"];
    expect((s as { kind: string }).kind).toBe("struct");
  });

  it("struct with field-value pairs", () => {
    const result = executeCode("s = struct('x', 1, 'y', 2);");
    const s = result.variableValues["s"];
    expect((s as { kind: string }).kind).toBe("struct");
  });

  it("struct field access", () => {
    expect(num("s = struct('x', 42); v = s.x;", "v")).toBe(42);
  });

  it("struct field assignment", () => {
    const result = executeCode("s = struct(); s.name = 'Alice';");
    const s = result.variableValues["s"];
    expect(s).toBeDefined();
  });
});

// ── fieldnames ────────────────────────────────────────────────────────

describe("fieldnames", () => {
  it("returns cell array of field names", () => {
    const result = executeCode(
      "s = struct('a', 1, 'b', 2); f = fieldnames(s);"
    );
    const f = result.variableValues["f"];
    expect(isRuntimeCell(f)).toBe(true);
    if (isRuntimeCell(f)) {
      expect(f.data.length).toBe(2);
    }
  });
});

// ── isfield ───────────────────────────────────────────────────────────

describe("isfield", () => {
  it("returns true for existing field", () => {
    expect(bool("s = struct('x', 1); v = isfield(s, 'x');", "v")).toBe(true);
  });

  it("returns false for non-existing field", () => {
    expect(bool("s = struct('x', 1); v = isfield(s, 'y');", "v")).toBe(false);
  });
});

// ── rmfield ───────────────────────────────────────────────────────────

describe("rmfield", () => {
  it("removes a field from struct", () => {
    const result = executeCode(
      "s = struct('x', 1, 'y', 2); s2 = rmfield(s, 'x');"
    );
    const s2 = result.variableValues["s2"];
    expect((s2 as { kind: string }).kind).toBe("struct");
  });
});

// ── num2cell ──────────────────────────────────────────────────────────

describe("num2cell", () => {
  it("converts each element to a cell", () => {
    const result = executeCode("c = num2cell([1,2,3]);");
    const c = result.variableValues["c"];
    expect(isRuntimeCell(c)).toBe(true);
    if (isRuntimeCell(c)) {
      expect(c.data.length).toBe(3);
    }
  });

  it("num2cell of scalar returns cell", () => {
    const result = executeCode("c = num2cell(5);");
    const c = result.variableValues["c"];
    expect(isRuntimeCell(c)).toBe(true);
  });
});

// ── cell2mat ──────────────────────────────────────────────────────────

describe("cell2mat", () => {
  it("assembles cell array of tensors into matrix", () => {
    const shape = tensorShape("c = {[1,2]; [3,4]}; A = cell2mat(c);", "A");
    expect(shape).toEqual([2, 2]);
  });

  it("cell2mat of scalars returns row vector", () => {
    const data = tensorData("c = {1, 2, 3}; v = cell2mat(c);", "v");
    expect(data).toEqual([1, 2, 3]);
  });
});

// ── struct2cell ───────────────────────────────────────────────────────

describe("struct2cell", () => {
  it("converts struct fields to cell column", () => {
    const result = executeCode(
      "s = struct('a', 1, 'b', 2); c = struct2cell(s);"
    );
    const c = result.variableValues["c"];
    expect(isRuntimeCell(c)).toBe(true);
    if (isRuntimeCell(c)) {
      expect(c.data.length).toBe(2);
    }
  });
});

// ── cell2struct ───────────────────────────────────────────────────────

describe("cell2struct", () => {
  it("creates struct from cell and field names", () => {
    const result = executeCode(
      "c = {42; 'hello'}; s = cell2struct(c, {'x'; 'y'}, 1);"
    );
    const s = result.variableValues["s"];
    expect((s as { kind: string }).kind).toBe("struct");
  });
});

// ── assert ────────────────────────────────────────────────────────────

describe("assert", () => {
  it("assert passes for true condition", () => {
    const result = executeCode("assert(1 == 1); x = 1;");
    expect(result.variableValues["x"]).toBe(1);
  });

  it("assert throws for false condition", () => {
    expect(() => executeCode("assert(1 == 2);")).toThrow();
  });

  it("assert with custom message", () => {
    expect(() => executeCode("assert(false, 'custom error');")).toThrow();
  });
});

// ── error ─────────────────────────────────────────────────────────────

describe("error", () => {
  it("error throws with message", () => {
    expect(() => executeCode("error('something went wrong');")).toThrow();
  });

  it("error with format string", () => {
    expect(() => executeCode("error('value is %d', 42);")).toThrow();
  });

  it("error is caught by try/catch", () => {
    const result = executeCode(`
try
  error('oops');
catch e
  x = 99;
end
`);
    expect(result.variableValues["x"]).toBe(99);
  });
});

// ── true() / false() ─────────────────────────────────────────────────

describe("true/false constructors", () => {
  it("true() returns true", () => {
    expect(bool("x = true();", "x")).toBe(true);
  });

  it("false() returns false", () => {
    expect(bool("x = false();", "x")).toBe(false);
  });

  it("true(2, 3) creates 2x3 logical ones", () => {
    const shape = tensorShape("A = true(2, 3);", "A");
    expect(shape).toEqual([2, 3]);
  });

  it("false(2, 2) creates 2x2 logical zeros", () => {
    const data = tensorData("A = false(2, 2);", "A");
    expect(data).toEqual([0, 0, 0, 0]);
  });
});

// ── deal ──────────────────────────────────────────────────────────────

describe("deal", () => {
  it("deal with single input replicates", () => {
    const result = executeCode("[a, b, c] = deal(7);");
    expect(result.variableValues["a"]).toBe(7);
    expect(result.variableValues["b"]).toBe(7);
    expect(result.variableValues["c"]).toBe(7);
  });

  it("deal distributes multiple inputs", () => {
    const result = executeCode("[a, b] = deal(1, 2);");
    expect(result.variableValues["a"]).toBe(1);
    expect(result.variableValues["b"]).toBe(2);
  });
});

// ── func2str ──────────────────────────────────────────────────────────

describe("func2str", () => {
  it("converts anonymous function to string", () => {
    const result = executeCode("f = @(x) x^2; s = func2str(f);");
    const s = result.variableValues["s"];
    expect(
      typeof s === "string" || (s as { value: string }).value !== undefined
    ).toBe(true);
  });
});

// ── tic/toc ───────────────────────────────────────────────────────────

describe("tic/toc", () => {
  it("toc returns non-negative elapsed time", () => {
    const result = executeCode("tic; t = toc;");
    const t = result.variableValues["t"];
    expect(typeof t).toBe("number");
    expect((t as number) >= 0).toBe(true);
  });
});

// ── coordinate transforms ─────────────────────────────────────────────

describe("cart2pol", () => {
  it("cart2pol at (1,0) has theta=0, rho=1", () => {
    const result = executeCode("[th, rho] = cart2pol(1, 0);");
    expect(result.variableValues["th"]).toBeCloseTo(0);
    expect(result.variableValues["rho"]).toBeCloseTo(1);
  });

  it("cart2pol at (0,1) has theta=pi/2", () => {
    const result = executeCode("[th, rho] = cart2pol(0, 1);");
    expect(result.variableValues["th"]).toBeCloseTo(Math.PI / 2);
    expect(result.variableValues["rho"]).toBeCloseTo(1);
  });
});

describe("pol2cart", () => {
  it("pol2cart roundtrip", () => {
    const result = executeCode("[x, y] = pol2cart(0, 1);");
    expect(result.variableValues["x"]).toBeCloseTo(1);
    expect(result.variableValues["y"]).toBeCloseTo(0);
  });
});

describe("cart2sph", () => {
  it("cart2sph at (1,0,0) has az=0, el=0, r=1", () => {
    const result = executeCode("[az, el, r] = cart2sph(1, 0, 0);");
    expect(result.variableValues["az"]).toBeCloseTo(0);
    expect(result.variableValues["el"]).toBeCloseTo(0);
    expect(result.variableValues["r"]).toBeCloseTo(1);
  });
});

describe("sph2cart", () => {
  it("sph2cart roundtrip from cart2sph", () => {
    const result = executeCode(
      "[az, el, r] = cart2sph(1, 0, 0); [x, y, z] = sph2cart(az, el, r);"
    );
    expect(result.variableValues["x"]).toBeCloseTo(1);
    expect(result.variableValues["y"]).toBeCloseTo(0);
    expect(result.variableValues["z"]).toBeCloseTo(0);
  });
});

// ── eval ──────────────────────────────────────────────────────────────

describe("eval", () => {
  it("eval executes a string as code", () => {
    // eval doesn't propagate vars back to outer scope in compiled mode,
    // but should not throw
    expect(() => executeCode("eval('1 + 1;');")).not.toThrow();
  });
});

// ── nargin / nargout ──────────────────────────────────────────────────

describe("nargin / nargout", () => {
  it("nargin inside function returns argument count", () => {
    const result = executeCode(`
function y = my_fn(a, b)
  y = nargin;
end
x = my_fn(1, 2);
`);
    expect(result.variableValues["x"]).toBe(2);
  });
});

// ── mat2cell ──────────────────────────────────────────────────────────

describe("mat2cell", () => {
  it("splits matrix by row/col distribution", () => {
    const result = executeCode(
      "A = [1,2,3;4,5,6]; C = mat2cell(A, [1,1], [1,2]);"
    );
    const C = result.variableValues["C"];
    expect(isRuntimeCell(C)).toBe(true);
    if (isRuntimeCell(C)) {
      expect(C.shape).toEqual([2, 2]);
    }
  });
});

// ── cellfun ───────────────────────────────────────────────────────────

describe("cellfun", () => {
  it("cellfun applies function to each cell element", () => {
    const result = executeCode("c = {1, 4, 9}; v = cellfun(@sqrt, c);");
    const v = result.variableValues["v"];
    expect(isRuntimeTensor(v)).toBe(true);
    if (isRuntimeTensor(v)) {
      expect(v.data[0]).toBeCloseTo(1);
      expect(v.data[1]).toBeCloseTo(2);
      expect(v.data[2]).toBeCloseTo(3);
    }
  });
});

// ── namedargs2cell ────────────────────────────────────────────────────

describe("namedargs2cell", () => {
  it("converts struct fields to name-value cell", () => {
    const result = executeCode(
      "s = struct('a', 1, 'b', 2); c = namedargs2cell(s);"
    );
    const c = result.variableValues["c"];
    expect(isRuntimeCell(c)).toBe(true);
    if (isRuntimeCell(c)) {
      expect(c.data.length).toBe(4); // 2 fields * 2 (name + value)
    }
  });
});

// ── sprintf / fprintf ─────────────────────────────────────────────────

describe("sprintf", () => {
  it("sprintf formats a number", () => {
    const result = executeCode("s = sprintf('%d', 42);");
    const s = result.variableValues["s"];
    expect((s as { value: string }).value).toBe("42");
  });

  it("sprintf formats a float", () => {
    const result = executeCode("s = sprintf('%.2f', 3.14159);");
    const s = result.variableValues["s"];
    expect((s as { value: string }).value).toBe("3.14");
  });

  it("sprintf formats a string", () => {
    const result = executeCode("s = sprintf('%s world', 'hello');");
    const s = result.variableValues["s"];
    expect((s as { value: string }).value).toBe("hello world");
  });
});

// ── input validation functions ────────────────────────────────────────

describe("isstruct / iscell misc", () => {
  it("isstruct on struct is true", () => {
    expect(bool("s = struct('x', 1); v = isstruct(s);", "v")).toBe(true);
  });

  it("isstruct on number is false", () => {
    expect(bool("v = isstruct(42);", "v")).toBe(false);
  });
});

import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";

// Helper to extract a numeric variable
function num(code: string, varName: string): number {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "number") throw new Error(`${varName} is not a number`);
  return v;
}

// Helper to extract a boolean variable
function bool(code: string, varName: string): boolean {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "boolean") throw new Error(`${varName} is not a boolean`);
  return v;
}

// Helper to extract tensor data as array
function tensorData(code: string, varName: string): number[] {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return Array.from(v.data);
}

// Helper to extract tensor shape
function tensorShape(code: string, varName: string): number[] {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return v.shape;
}

// ── Math functions ─────────────────────────────────────────────────────

describe("math builtins", () => {
  it("sin(0) = 0", () => {
    expect(num("x = sin(0);", "x")).toBeCloseTo(0);
  });

  it("sin(pi/2) = 1", () => {
    expect(num("x = sin(pi/2);", "x")).toBeCloseTo(1);
  });

  it("cos(0) = 1", () => {
    expect(num("x = cos(0);", "x")).toBeCloseTo(1);
  });

  it("cos(pi) = -1", () => {
    expect(num("x = cos(pi);", "x")).toBeCloseTo(-1);
  });

  it("tan(pi/4) ≈ 1", () => {
    expect(num("x = tan(pi/4);", "x")).toBeCloseTo(1);
  });

  it("asin(1) = pi/2", () => {
    expect(num("x = asin(1);", "x")).toBeCloseTo(Math.PI / 2);
  });

  it("acos(1) = 0", () => {
    expect(num("x = acos(1);", "x")).toBeCloseTo(0);
  });

  it("atan(1) ≈ pi/4", () => {
    expect(num("x = atan(1);", "x")).toBeCloseTo(Math.PI / 4);
  });

  it("atan2(1, 1) ≈ pi/4", () => {
    expect(num("x = atan2(1, 1);", "x")).toBeCloseTo(Math.PI / 4);
  });

  it("sqrt(4) = 2", () => {
    expect(num("x = sqrt(4);", "x")).toBe(2);
  });

  it("sqrt(9) = 3", () => {
    expect(num("x = sqrt(9);", "x")).toBe(3);
  });

  it("abs(-5) = 5", () => {
    expect(num("x = abs(-5);", "x")).toBe(5);
  });

  it("abs(3) = 3", () => {
    expect(num("x = abs(3);", "x")).toBe(3);
  });

  it("abs on vector", () => {
    expect(tensorData("v = abs([-1, 2, -3]);", "v")).toEqual([1, 2, 3]);
  });

  it("floor(3.7) = 3", () => {
    expect(num("x = floor(3.7);", "x")).toBe(3);
  });

  it("floor(-1.2) = -2", () => {
    expect(num("x = floor(-1.2);", "x")).toBe(-2);
  });

  it("ceil(3.2) = 4", () => {
    expect(num("x = ceil(3.2);", "x")).toBe(4);
  });

  it("ceil(-1.7) = -1", () => {
    expect(num("x = ceil(-1.7);", "x")).toBe(-1);
  });

  it("round(3.5) = 4", () => {
    expect(num("x = round(3.5);", "x")).toBe(4);
  });

  it("round(3.4) = 3", () => {
    expect(num("x = round(3.4);", "x")).toBe(3);
  });

  it("fix(3.7) = 3", () => {
    expect(num("x = fix(3.7);", "x")).toBe(3);
  });

  it("fix(-3.7) = -3", () => {
    expect(num("x = fix(-3.7);", "x")).toBe(-3);
  });

  it("sign(-5) = -1", () => {
    expect(num("x = sign(-5);", "x")).toBe(-1);
  });

  it("sign(5) = 1", () => {
    expect(num("x = sign(5);", "x")).toBe(1);
  });

  it("sign(0) = 0", () => {
    expect(num("x = sign(0);", "x")).toBe(0);
  });

  it("exp(0) = 1", () => {
    expect(num("x = exp(0);", "x")).toBe(1);
  });

  it("exp(1) ≈ e", () => {
    expect(num("x = exp(1);", "x")).toBeCloseTo(Math.E);
  });

  it("log(1) = 0", () => {
    expect(num("x = log(1);", "x")).toBe(0);
  });

  it("log(e) = 1", () => {
    expect(num("x = log(exp(1));", "x")).toBeCloseTo(1);
  });

  it("log2(8) = 3", () => {
    expect(num("x = log2(8);", "x")).toBeCloseTo(3);
  });

  it("log10(100) = 2", () => {
    expect(num("x = log10(100);", "x")).toBeCloseTo(2);
  });

  it("mod(10, 3) = 1", () => {
    expect(num("x = mod(10, 3);", "x")).toBe(1);
  });

  it("mod(-1, 3) = 2 (positive result)", () => {
    expect(num("x = mod(-1, 3);", "x")).toBe(2);
  });

  it("rem(10, 3) = 1", () => {
    expect(num("x = rem(10, 3);", "x")).toBe(1);
  });

  it("rem(-10, 3) = -1 (sign follows dividend)", () => {
    expect(num("x = rem(-10, 3);", "x")).toBe(-1);
  });

  it("hypot(3, 4) = 5", () => {
    expect(num("x = hypot(3, 4);", "x")).toBeCloseTo(5);
  });

  it("power(2, 10) = 1024", () => {
    expect(num("x = power(2, 10);", "x")).toBe(1024);
  });

  it("sin on vector applies element-wise", () => {
    const data = tensorData("v = sin([0, pi/2, pi]);", "v");
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(1);
    expect(data[2]).toBeCloseTo(0);
  });
});

// ── Reduction builtins ─────────────────────────────────────────────────

describe("reduction builtins", () => {
  it("sum of row vector", () => {
    expect(num("x = sum([1, 2, 3, 4]);", "x")).toBe(10);
  });

  it("sum of empty = 0", () => {
    expect(num("x = sum([]);", "x")).toBe(0);
  });

  it("sum along dim 2 returns column vector", () => {
    const data = tensorData("A = [1,2;3,4]; s = sum(A, 2);", "s");
    expect(data).toEqual([3, 7]);
  });

  it("prod of vector", () => {
    expect(num("x = prod([1, 2, 3, 4]);", "x")).toBe(24);
  });

  it("prod of empty = 1", () => {
    expect(num("x = prod([]);", "x")).toBe(1);
  });

  it("min of vector", () => {
    expect(num("x = min([3, 1, 4, 1, 5]);", "x")).toBe(1);
  });

  it("max of vector", () => {
    expect(num("x = max([3, 1, 4, 1, 5]);", "x")).toBe(5);
  });

  it("min(a, b) returns element-wise minimum", () => {
    const data = tensorData("v = min([5, 2, 3], [1, 4, 3]);", "v");
    expect(data).toEqual([1, 2, 3]);
  });

  it("max(a, b) returns element-wise maximum", () => {
    const data = tensorData("v = max([5, 2, 3], [1, 4, 3]);", "v");
    expect(data).toEqual([5, 4, 3]);
  });

  it("mean of vector", () => {
    expect(num("x = mean([1, 2, 3, 4, 5]);", "x")).toBe(3);
  });

  it("mean of matrix along dim 1", () => {
    const data = tensorData("A = [1,3;2,4]; m = mean(A, 1);", "m");
    expect(data[0]).toBeCloseTo(1.5);
    expect(data[1]).toBeCloseTo(3.5);
  });

  it("std of vector (sample, N-1)", () => {
    // std([1, 3]) = sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2)
    expect(num("x = std([1, 3]);", "x")).toBeCloseTo(Math.sqrt(2));
  });

  it("var of vector (sample, N-1)", () => {
    // var([1, 3]) = ((1-2)^2 + (3-2)^2) / 1 = 2
    expect(num("x = var([1, 3]);", "x")).toBeCloseTo(2);
  });

  it("any returns true when any element is nonzero", () => {
    expect(bool("x = any([0, 0, 1]);", "x")).toBe(true);
  });

  it("any returns false when all zero", () => {
    expect(bool("x = any([0, 0, 0]);", "x")).toBe(false);
  });

  it("all returns true when all nonzero", () => {
    expect(bool("x = all([1, 2, 3]);", "x")).toBe(true);
  });

  it("all returns false when any is zero", () => {
    expect(bool("x = all([1, 0, 3]);", "x")).toBe(false);
  });

  it("find returns indices of nonzero elements", () => {
    const data = tensorData("idx = find([0, 3, 0, 5, 0]);", "idx");
    expect(data).toEqual([2, 4]); // 1-based indices
  });

  it("find with n returns first n indices", () => {
    const data = tensorData("idx = find([1,0,1,0,1], 2);", "idx");
    expect(data).toEqual([1, 3]);
  });

  it("sum('all') reduces to scalar", () => {
    expect(num("A = [1,2;3,4]; x = sum(A, 'all');", "x")).toBe(10);
  });

  it("any('all') over matrix", () => {
    expect(bool("A = [0,0;0,1]; x = any(A, 'all');", "x")).toBe(true);
  });

  it("all('all') over matrix", () => {
    expect(bool("A = [1,1;1,0]; x = all(A, 'all');", "x")).toBe(false);
  });

  it("xor", () => {
    expect(bool("x = xor(true, false);", "x")).toBe(true);
    expect(bool("x = xor(true, true);", "x")).toBe(false);
  });
});

// ── String builtins ────────────────────────────────────────────────────

describe("string builtins", () => {
  it("strcmp with equal strings", () => {
    expect(bool("x = strcmp('hello', 'hello');", "x")).toBe(true);
  });

  it("strcmp with different strings", () => {
    expect(bool("x = strcmp('hello', 'world');", "x")).toBe(false);
  });

  it("strcmp is case-sensitive", () => {
    expect(bool("x = strcmp('Hello', 'hello');", "x")).toBe(false);
  });

  it("strcmpi is case-insensitive", () => {
    expect(bool("x = strcmpi('Hello', 'hello');", "x")).toBe(true);
  });

  it("strncmp compares first n characters", () => {
    expect(bool("x = strncmp('hello', 'help', 3);", "x")).toBe(true);
    expect(bool("x = strncmp('hello', 'help', 4);", "x")).toBe(false);
  });

  it("lower converts to lowercase", () => {
    const result = executeCode("s = lower('HELLO');");
    const v = result.variableValues["s"];
    expect(v).toHaveProperty("value", "hello");
  });

  it("upper converts to uppercase", () => {
    const result = executeCode("s = upper('hello');");
    const v = result.variableValues["s"];
    expect(v).toHaveProperty("value", "HELLO");
  });

  it("strtrim removes whitespace", () => {
    const result = executeCode("s = strtrim('  hello  ');");
    const v = result.variableValues["s"];
    expect(v).toHaveProperty("value", "hello");
  });

  it("strcat concatenates strings", () => {
    const result = executeCode("s = strcat('hello', ' world');");
    const v = result.variableValues["s"];
    // char: strips trailing whitespace from each, but ' world' has leading space
    expect(v).toHaveProperty("value");
    expect((v as { value: string }).value).toContain("hello");
  });

  it("strlength returns string length", () => {
    expect(num("n = strlength('hello');", "n")).toBe(5);
  });

  it("num2str converts number to string", () => {
    const result = executeCode("s = num2str(42);");
    const v = result.variableValues["s"];
    expect(v).toHaveProperty("value");
    expect((v as { value: string }).value).toContain("42");
  });

  it("int2str rounds to integer string", () => {
    const result = executeCode("s = int2str(3.7);");
    const v = result.variableValues["s"];
    expect(v).toHaveProperty("value", "4");
  });

  it("char(65) produces 'A'", () => {
    const result = executeCode("s = char(65);");
    const v = result.variableValues["s"];
    expect(v).toHaveProperty("value", "A");
  });

  it("strfind finds substring positions", () => {
    const data = tensorData("idx = strfind('abcabc', 'bc');", "idx");
    expect(data).toEqual([2, 5]); // 1-based
  });

  it("strfind returns empty when not found", () => {
    const result = executeCode("idx = strfind('hello', 'xyz');");
    const v = result.variableValues["idx"];
    expect(isRuntimeTensor(v)).toBe(true);
    if (isRuntimeTensor(v)) expect(v.data.length).toBe(0);
  });
});

// ── Array manipulation builtins ────────────────────────────────────────

describe("array manipulation builtins", () => {
  it("reshape changes shape", () => {
    const shape = tensorShape("A = reshape(1:6, 2, 3);", "A");
    expect(shape).toEqual([2, 3]);
  });

  it("reshape preserves elements", () => {
    const data = tensorData("A = reshape([1,2,3,4], 2, 2);", "A");
    expect(data).toHaveLength(4);
  });

  it("reshape with auto dimension ([])", () => {
    const shape = tensorShape("A = reshape(1:6, [], 3);", "A");
    expect(shape).toEqual([2, 3]);
  });

  it("linspace creates evenly spaced vector", () => {
    const data = tensorData("v = linspace(0, 1, 5);", "v");
    expect(data).toHaveLength(5);
    expect(data[0]).toBeCloseTo(0);
    expect(data[4]).toBeCloseTo(1);
    expect(data[2]).toBeCloseTo(0.5);
  });

  it("linspace default is 100 points", () => {
    const shape = tensorShape("v = linspace(0, 1);", "v");
    expect(shape).toEqual([1, 100]);
  });

  it("diag of vector creates diagonal matrix", () => {
    const shape = tensorShape("A = diag([1,2,3]);", "A");
    expect(shape).toEqual([3, 3]);
  });

  it("diag of matrix extracts diagonal", () => {
    const data = tensorData("A = [1,2;3,4]; d = diag(A);", "d");
    expect(data[0]).toBe(1);
    expect(data[1]).toBe(4);
  });

  it("transpose function works", () => {
    const shape = tensorShape("A = [1,2,3]; B = transpose(A);", "B");
    expect(shape).toEqual([3, 1]);
  });

  it("colon function creates range", () => {
    const data = tensorData("v = colon(1, 5);", "v");
    expect(data).toEqual([1, 2, 3, 4, 5]);
  });

  it("colon with step", () => {
    const data = tensorData("v = colon(0, 2, 10);", "v");
    expect(data).toEqual([0, 2, 4, 6, 8, 10]);
  });

  it("horzcat via bracket syntax", () => {
    const shape = tensorShape("A = [[1,2], [3,4]];", "A");
    expect(shape).toEqual([1, 4]);
  });

  it("vertcat via bracket syntax", () => {
    const shape = tensorShape("A = [[1;2]; [3;4]];", "A");
    expect(shape).toEqual([4, 1]);
  });
});

// ── Introspection builtins ─────────────────────────────────────────────

describe("introspection builtins", () => {
  it("size returns [rows, cols]", () => {
    const data = tensorData("A = [1,2;3,4;5,6]; s = size(A);", "s");
    expect(data).toEqual([3, 2]);
  });

  it("size(A, 1) returns rows", () => {
    expect(num("A = [1,2;3,4]; s = size(A, 1);", "s")).toBe(2);
  });

  it("size(A, 2) returns cols", () => {
    expect(num("A = [1,2;3,4]; s = size(A, 2);", "s")).toBe(2);
  });

  it("size of scalar is [1, 1]", () => {
    const data = tensorData("s = size(42);", "s");
    expect(data).toEqual([1, 1]);
  });

  it("length returns max dimension", () => {
    expect(num("A = [1,2,3;4,5,6]; n = length(A);", "n")).toBe(3);
  });

  it("length of scalar is 1", () => {
    expect(num("n = length(42);", "n")).toBe(1);
  });

  it("numel counts all elements", () => {
    expect(num("A = [1,2;3,4;5,6]; n = numel(A);", "n")).toBe(6);
  });

  it("numel of scalar is 1", () => {
    expect(num("n = numel(5);", "n")).toBe(1);
  });

  it("isempty on empty matrix", () => {
    expect(bool("x = isempty([]);", "x")).toBe(true);
  });

  it("isempty on non-empty", () => {
    expect(bool("x = isempty([1,2]);", "x")).toBe(false);
  });

  it("isscalar on number", () => {
    expect(bool("x = isscalar(5);", "x")).toBe(true);
  });

  it("isscalar on matrix", () => {
    expect(bool("x = isscalar([1,2]);", "x")).toBe(false);
  });

  it("isvector on row vector", () => {
    expect(bool("x = isvector([1,2,3]);", "x")).toBe(true);
  });

  it("isvector on matrix", () => {
    expect(bool("x = isvector([1,2;3,4]);", "x")).toBe(false);
  });

  it("ismatrix on 2D array", () => {
    expect(bool("x = ismatrix([1,2;3,4]);", "x")).toBe(true);
  });

  it("isfloat on number", () => {
    expect(bool("x = isfloat(3.14);", "x")).toBe(true);
  });

  it("isnumeric on number", () => {
    expect(bool("x = isnumeric(42);", "x")).toBe(true);
  });

  it("isnumeric on string", () => {
    expect(bool("x = isnumeric('hello');", "x")).toBe(false);
  });

  it("islogical on boolean", () => {
    expect(bool("x = islogical(true);", "x")).toBe(true);
  });

  it("islogical on number", () => {
    expect(bool("x = islogical(1);", "x")).toBe(false);
  });

  it("ischar on char", () => {
    expect(bool("x = ischar('hello');", "x")).toBe(true);
  });

  it("ischar on number", () => {
    expect(bool("x = ischar(42);", "x")).toBe(false);
  });

  it("iscell on cell array", () => {
    expect(bool("x = iscell({1, 2, 3});", "x")).toBe(true);
  });

  it("iscell on matrix", () => {
    expect(bool("x = iscell([1, 2, 3]);", "x")).toBe(false);
  });

  it("class of number", () => {
    const result = executeCode("c = class(3.14);");
    const v = result.variableValues["c"];
    expect(v).toBe("double");
  });

  it("class of logical", () => {
    const result = executeCode("c = class(true);");
    const v = result.variableValues["c"];
    expect(v).toBe("logical");
  });

  it("class of char", () => {
    const result = executeCode("c = class('hi');");
    const v = result.variableValues["c"];
    expect(v).toBe("char");
  });

  it("isequal with equal values", () => {
    expect(bool("x = isequal([1,2,3], [1,2,3]);", "x")).toBe(true);
  });

  it("isequal with different values", () => {
    expect(bool("x = isequal([1,2,3], [1,2,4]);", "x")).toBe(false);
  });

  it("isequal with scalars", () => {
    expect(bool("x = isequal(5, 5);", "x")).toBe(true);
    expect(bool("x = isequal(5, 6);", "x")).toBe(false);
  });
});

// ── Control flow ───────────────────────────────────────────────────────

describe("control flow", () => {
  it("switch/case matches value", () => {
    const result = executeCode(`
x = 2;
switch x
  case 1
    y = 'one';
  case 2
    y = 'two';
  otherwise
    y = 'other';
end
`);
    const v = result.variableValues["y"];
    expect(v).toHaveProperty("value", "two");
  });

  it("switch/case falls through to otherwise", () => {
    const result = executeCode(`
x = 99;
switch x
  case 1
    y = 'one';
  otherwise
    y = 'other';
end
`);
    const v = result.variableValues["y"];
    expect(v).toHaveProperty("value", "other");
  });

  it("try/catch catches an error", () => {
    const result = executeCode(`
try
  error('oops');
  x = 1;
catch e
  x = 2;
end
`);
    expect(result.variableValues["x"]).toBe(2);
  });

  it("break exits loop early", () => {
    const result = executeCode(`
s = 0;
for i = 1:10
  if i > 3
    break;
  end
  s = s + i;
end
`);
    expect(result.variableValues["s"]).toBe(6); // 1+2+3
  });

  it("continue skips iteration", () => {
    const result = executeCode(`
s = 0;
for i = 1:5
  if mod(i, 2) == 0
    continue;
  end
  s = s + i;
end
`);
    expect(result.variableValues["s"]).toBe(9); // 1+3+5
  });

  it("nested for loops", () => {
    const result = executeCode(`
s = 0;
for i = 1:3
  for j = 1:3
    s = s + 1;
  end
end
`);
    expect(result.variableValues["s"]).toBe(9);
  });

  it("while with break", () => {
    const result = executeCode(`
x = 0;
while true
  x = x + 1;
  if x >= 5
    break;
  end
end
`);
    expect(result.variableValues["x"]).toBe(5);
  });

  it("return exits function early", () => {
    const result = executeCode(`
function r = first_positive(v)
  r = -1;
  for i = 1:length(v)
    if v(i) > 0
      r = v(i);
      return;
    end
  end
end
x = first_positive([-3, -1, 4, 2]);
`);
    expect(result.variableValues["x"]).toBe(4);
  });
});

// ── Arithmetic edge cases ──────────────────────────────────────────────

describe("arithmetic edge cases", () => {
  it("matrix multiplication", () => {
    const result = executeCode("A = [1,2;3,4]; B = [5;6]; C = A * B;");
    const C = result.variableValues["C"];
    expect(isRuntimeTensor(C)).toBe(true);
    if (isRuntimeTensor(C)) {
      expect(C.shape).toEqual([2, 1]);
      expect(C.data[0]).toBe(17); // 1*5+2*6
      expect(C.data[1]).toBe(39); // 3*5+4*6
    }
  });

  it("scalar + matrix broadcasts", () => {
    const data = tensorData("A = [1,2;3,4] + 10;", "A");
    expect(data).toEqual([11, 13, 12, 14]); // column-major
  });

  it("element-wise power .^", () => {
    const data = tensorData("v = [1,2,3] .^ 2;", "v");
    expect(data).toEqual([1, 4, 9]);
  });

  it("element-wise division ./", () => {
    const data = tensorData("v = [2,4,6] ./ 2;", "v");
    expect(data).toEqual([1, 2, 3]);
  });

  it("unary negation of vector", () => {
    const data = tensorData("v = -[1,2,3];", "v");
    expect(data).toEqual([-1, -2, -3]);
  });

  it("comparison returns logical vector", () => {
    const result = executeCode("v = [1,2,3,4,5] > 3;");
    const v = result.variableValues["v"];
    expect(isRuntimeTensor(v)).toBe(true);
    if (isRuntimeTensor(v)) {
      expect(Array.from(v.data)).toEqual([0, 0, 0, 1, 1]);
    }
  });

  it("logical AND short-circuits", () => {
    expect(bool("x = (2 > 1) && (3 > 2);", "x")).toBe(true);
    expect(bool("x = (2 > 1) && (1 > 2);", "x")).toBe(false);
  });

  it("logical OR", () => {
    expect(bool("x = false || true;", "x")).toBe(true);
    expect(bool("x = false || false;", "x")).toBe(false);
  });

  it("matrix left division A \\ b", () => {
    // Solve Ax = b for x: [2 0; 0 3] \ [4; 6] = [2; 2]
    const result = executeCode("A = [2,0;0,3]; b = [4;6]; x = A \\ b;");
    const x = result.variableValues["x"];
    expect(isRuntimeTensor(x)).toBe(true);
    if (isRuntimeTensor(x)) {
      expect(x.data[0]).toBeCloseTo(2);
      expect(x.data[1]).toBeCloseTo(2);
    }
  });
});

// ── Built-in array constructors ────────────────────────────────────────

describe("array constructors", () => {
  it("nan creates NaN matrix", () => {
    const result = executeCode("A = nan(2, 2);");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.shape).toEqual([2, 2]);
      expect(isNaN(A.data[0])).toBe(true);
    }
  });

  it("NaN() creates NaN matrix", () => {
    const result = executeCode("A = NaN(1, 3);");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.shape).toEqual([1, 3]);
    }
  });

  it("zeros with single argument creates square matrix", () => {
    const shape = tensorShape("A = zeros(3);", "A");
    expect(shape).toEqual([3, 3]);
  });

  it("ones with single argument creates square matrix", () => {
    const shape = tensorShape("A = ones(3);", "A");
    expect(shape).toEqual([3, 3]);
  });

  it("eye with single argument creates identity", () => {
    const result = executeCode("A = eye(2);");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.data[0]).toBe(1);
      expect(A.data[1]).toBe(0);
      expect(A.data[2]).toBe(0);
      expect(A.data[3]).toBe(1);
    }
  });
});

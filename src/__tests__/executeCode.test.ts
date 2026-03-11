import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";

describe("executeCode", () => {
  it("executes a simple assignment and returns variable", () => {
    const result = executeCode("x = 42;");
    expect(result.variableValues["x"]).toBe(42);
  });

  it("executes arithmetic", () => {
    const result = executeCode("x = 2 + 3;");
    expect(result.variableValues["x"]).toBe(5);
  });

  it("captures output from disp", () => {
    const result = executeCode("disp(42);");
    expect(result.output.join("")).toContain("42");
  });

  it("captures output from unsuppressed expression", () => {
    const result = executeCode("x = 10", { displayResults: true });
    expect(result.output.join("")).toContain("10");
  });

  it("suppresses output with semicolon", () => {
    const result = executeCode("x = 10;");
    expect(result.output.join("").trim()).toBe("");
  });

  it("supports matrix creation", () => {
    const result = executeCode("A = [1, 2; 3, 4];");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.shape).toEqual([2, 2]);
    }
  });

  it("supports string variables", () => {
    const result = executeCode("s = 'hello';");
    // Strings in MATLAB are char arrays
    expect(result.variableValues["s"]).toBeDefined();
  });

  it("supports if-else", () => {
    const result = executeCode("if true\n  x = 1;\nelse\n  x = 2;\nend");
    expect(result.variableValues["x"]).toBe(1);
  });

  it("supports for loop", () => {
    const result = executeCode("s = 0;\nfor i = 1:5\n  s = s + i;\nend");
    expect(result.variableValues["s"]).toBe(15);
  });

  it("supports while loop", () => {
    const result = executeCode("x = 10;\nwhile x > 5\n  x = x - 1;\nend");
    expect(result.variableValues["x"]).toBe(5);
  });

  it("supports function definitions and calls", () => {
    const code = `
function y = double_it(x)
  y = x * 2;
end
result = double_it(21);
`;
    const result = executeCode(code);
    expect(result.variableValues["result"]).toBe(42);
  });

  it("supports anonymous functions", () => {
    const result = executeCode("f = @(x) x^2;\ny = f(5);");
    expect(result.variableValues["y"]).toBe(25);
  });

  it("supports logical operations", () => {
    const result = executeCode("x = true && false;");
    expect(result.variableValues["x"]).toBe(false);
  });

  it("supports comparison operators", () => {
    const result = executeCode("x = 3 > 2;");
    expect(result.variableValues["x"]).toBe(true);
  });

  it("supports element-wise operations", () => {
    const result = executeCode("A = [1, 2, 3]; B = A .* 2;");
    const B = result.variableValues["B"];
    expect(isRuntimeTensor(B)).toBe(true);
    if (isRuntimeTensor(B)) {
      expect(Array.from(B.data)).toEqual([2, 4, 6]);
    }
  });

  it("supports transpose", () => {
    const result = executeCode("A = [1, 2, 3]; B = A';");
    const B = result.variableValues["B"];
    expect(isRuntimeTensor(B)).toBe(true);
    if (isRuntimeTensor(B)) {
      expect(B.shape).toEqual([3, 1]);
    }
  });

  it("supports built-in functions like zeros", () => {
    const result = executeCode("A = zeros(2, 3);");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.shape).toEqual([2, 3]);
      expect(Array.from(A.data)).toEqual([0, 0, 0, 0, 0, 0]);
    }
  });

  it("supports ones", () => {
    const result = executeCode("A = ones(1, 4);");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(Array.from(A.data)).toEqual([1, 1, 1, 1]);
    }
  });

  it("supports eye", () => {
    const result = executeCode("A = eye(3);");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.shape).toEqual([3, 3]);
      // diagonal should be 1
      expect(A.data[0]).toBe(1);
      expect(A.data[4]).toBe(1);
      expect(A.data[8]).toBe(1);
    }
  });

  it("supports size and length", () => {
    const result = executeCode("A = [1,2,3;4,5,6]; s = size(A, 1);");
    expect(result.variableValues["s"]).toBe(2);
  });

  it("supports initial variable values", () => {
    const result = executeCode("y = x + 1;", {
      initialVariableValues: { x: 10 },
    });
    expect(result.variableValues["y"]).toBe(11);
  });

  it("returns generated JS code", () => {
    const result = executeCode("x = 1;");
    expect(result.generatedJS).toBeTruthy();
    expect(typeof result.generatedJS).toBe("string");
  });
});

import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  builtin,
  classMethod,
} from "../numbl-core/interpreter/builtins/builtin.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import { dispatchClassOf } from "../numbl-core/runtime/kinds.js";
import { isRuntimeClassInstance } from "../numbl-core/runtime/types.js";

// A class implemented in TypeScript, with methods registered in the class
// method table: an operator (`plus`, `not`) and an ordinary function name.
const CLS = "TsProbeClass";

builtin({
  name: "tsProbeMake",
  nargin: 1,
  apply: args => {
    const obj = RTV.classInstance(CLS, ["v"]);
    obj.fields.set("v", args[0]);
    return obj;
  },
});

const tag = (s: string) => RTV.char(s);
classMethod(CLS, { name: "plus", nargin: 2, apply: () => tag("probe.plus") });
classMethod(CLS, { name: "not", nargin: 1, apply: () => tag("probe.not") });
classMethod(CLS, {
  name: "tsProbeDescribe",
  nargin: 1,
  apply: () => tag("probe.describe"),
});

const run = (code: string, files: { name: string; source: string }[] = []) =>
  executeCode(code, { optimization: "0" }, files).variableValues;

const str = (v: unknown) => (v as { value: string }).value;

describe("TypeScript class method table", () => {
  it("gives the class a dispatch class", () => {
    const vals = run("p = tsProbeMake(1);");
    expect(isRuntimeClassInstance(vals["p"] as never)).toBe(true);
    expect(dispatchClassOf(vals["p"])).toBe(CLS);
    expect(dispatchClassOf(3)).toBeNull();
    expect(dispatchClassOf(RTV.cell([], [0, 0]))).toBeNull();
  });

  it("is chosen by an operator and by a function call", () => {
    const vals = run(
      [
        "p = tsProbeMake(1);",
        "a = p + 1;",
        "b = 1 + p;",
        "c = plus(p, 2);",
        "d = ~p;",
        "e = not(p);",
        "f = tsProbeDescribe(p);",
        "g = p.tsProbeDescribe();",
        "h = ismethod(p, 'tsProbeDescribe');",
      ].join("\n")
    );
    expect(str(vals["a"])).toBe("probe.plus");
    expect(str(vals["b"])).toBe("probe.plus");
    expect(str(vals["c"])).toBe("probe.plus");
    expect(str(vals["d"])).toBe("probe.not");
    expect(str(vals["e"])).toBe("probe.not");
    expect(str(vals["f"])).toBe("probe.describe");
    expect(str(vals["g"])).toBe("probe.describe");
    expect(vals["h"]).toBe(true);
  });

  it("an operator it does not define raises MATLAB's error", () => {
    const vals = run(
      "p = tsProbeMake(1);\ntry\n  q = p * 2;\ncatch err\n  id = err.identifier;\nend"
    );
    expect(str(vals["id"])).toBe("MATLAB:math:mustBeNumericCharOrLogical");
  });

  it("a classdef object's method dominates", () => {
    const cd = [
      "classdef CdPlus",
      "  methods",
      "    function r = plus(a, b)",
      "      r = 'classdef.plus';",
      "    end",
      "  end",
      "end",
    ].join("\n");
    const vals = run(
      "p = tsProbeMake(1);\nc = CdPlus();\na = p + c;\nb = c + p;\nd = plus(p, c);",
      [{ name: "CdPlus.m", source: cd }]
    );
    expect(str(vals["a"])).toBe("classdef.plus");
    expect(str(vals["b"])).toBe("classdef.plus");
    expect(str(vals["d"])).toBe("classdef.plus");
  });
});

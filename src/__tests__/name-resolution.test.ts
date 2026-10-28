import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import type { RuntimeValue } from "../numbl-core/runtime/types.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";
import type { WorkspaceFile } from "../numbl-core/workspace/types.js";

function values(v: RuntimeValue | undefined): number[] {
  if (typeof v === "number") return [v];
  if (v !== undefined && isRuntimeTensor(v)) return Array.from(v.data);
  throw new Error("expected a numeric value");
}

// MATLAB's cellfun accepts only a few legacy names ('isempty', ...) as
// strings; numbl accepts any function name. Such a name resolves from the
// caller's scope, like feval('name').
describe("calls by name resolve from the caller's scope", () => {
  it("sees the local functions of a classdef file from its methods", () => {
    const cls: WorkspaceFile = {
      name: "/w/NrcClass.m",
      source: `classdef NrcClass
  methods
    function r = viaCellfun(obj)
      r = cellfun('nrc_loc', {1, 2});
    end
  end
end
function r = nrc_loc(x)
  r = 10 * x;
end`,
    };
    const result = executeCode(
      "o = NrcClass(); r = o.viaCellfun();",
      { optimization: "0" },
      [cls],
      "/w/script.m",
      ["/w"]
    );
    expect(values(result.variableValues["r"])).toEqual([10, 20]);
  });

  it("sees the private functions of a private function's folder", () => {
    const files: WorkspaceFile[] = [
      {
        name: "/w/private/nrc_priv.m",
        source: `function r = nrc_priv(x)
r = cellfun('nrc_priv2', {x});
end`,
      },
      {
        name: "/w/private/nrc_priv2.m",
        source: `function y = nrc_priv2(x)
y = x + 1000;
end`,
      },
    ];
    const result = executeCode(
      "r = nrc_priv(1);",
      { optimization: "0" },
      files,
      "/w/script.m",
      ["/w"]
    );
    expect(values(result.variableValues["r"])).toEqual([1001]);
  });
});

import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeChar } from "../numbl-core/runtime/types.js";
import type { WorkspaceFile } from "../numbl-core/workspace/types.js";

describe("classdef search path priority", () => {
  it("first classdef on search path wins when same name exists in multiple paths", () => {
    // Simulates two directories each containing Greeter_.m with different Tags.
    // path_a (first) has Tag='A', path_b (second) has Tag='B'.
    // The first-registered class should win.
    const fileA: WorkspaceFile = {
      name: "/fake/path_a/Greeter_.m",
      source: `classdef Greeter_ < handle
  properties
    Tag
  end
  methods
    function obj = Greeter_()
      obj.Tag = 'A';
    end
  end
end`,
    };

    const fileB: WorkspaceFile = {
      name: "/fake/path_b/Greeter_.m",
      source: `classdef Greeter_ < handle
  properties
    Tag
  end
  methods
    function obj = Greeter_()
      obj.Tag = 'B';
    end
  end
end`,
    };

    // fileA is listed first — it should win
    const result = executeCode(
      "g = Greeter_(); x = g.Tag;",
      {},
      [fileA, fileB],
      "script.m",
      ["/fake/path_a", "/fake/path_b"]
    );
    const x = result.variableValues["x"];
    expect(isRuntimeChar(x)).toBe(true);
    if (isRuntimeChar(x)) {
      expect(x.value).toBe("A");
    }
  });

  it("@ClassName methods without classdef do not shadow regular functions", () => {
    // Simulates @UnknownClass/greet.m (no classdef) alongside a regular greet.m.
    // The regular greet.m should be called, not the @-folder method.
    const regularGreet: WorkspaceFile = {
      name: "/fake/dir/greet.m",
      source: `function r = greet(name)
r = ['hello ' name];
`,
    };

    const atFolderGreet: WorkspaceFile = {
      name: "/fake/dir/@UnknownClass/greet.m",
      source: `function r = greet(obj)
r = 'wrong';
`,
    };

    const result = executeCode(
      "x = greet('world');",
      {},
      [regularGreet, atFolderGreet],
      "script.m",
      ["/fake/dir"]
    );
    const x = result.variableValues["x"];
    expect(isRuntimeChar(x)).toBe(true);
    if (isRuntimeChar(x)) {
      expect(x.value).toBe("hello world");
    }
  });
});

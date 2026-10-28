import { describe, it, expect } from "vitest";
import {
  callerPrivateScope,
  privateDirScope,
  relativeToSearchPath,
} from "../numbl-core/workspace/searchPaths.js";

describe("search-path helpers", () => {
  it("relativeToSearchPath uses the longest matching search path", () => {
    const paths = ["/a", "/a/b"];
    expect(relativeToSearchPath("/a/b/c.m", paths)).toBe("c.m");
    expect(relativeToSearchPath("/a/x/c.m", paths)).toBe("x/c.m");
    expect(relativeToSearchPath("/z/c.m", paths)).toBe("/z/c.m");
    expect(relativeToSearchPath("c.m", [])).toBe("c.m");
  });

  it("a private function is in the scope of its siblings and of its parent dir", () => {
    const paths = ["/w"];
    for (const [callee, callers] of [
      ["/w/private/h.m", ["/w/main.m", "/w/private/g.m"]],
      ["/w/+pkg/private/h.m", ["/w/+pkg/f.m", "/w/+pkg/private/g.m"]],
      ["/w/@Cls/private/h.m", ["/w/@Cls/method.m", "/w/@Cls/Cls.m"]],
    ] as const) {
      const parts = relativeToSearchPath(callee, paths).split("/");
      const scope = privateDirScope(parts);
      for (const caller of callers) {
        expect(callerPrivateScope(caller, paths)).toBe(scope);
      }
    }
    expect(callerPrivateScope("/w/other/f.m", paths)).toBe("other/");
    expect(privateDirScope(["+pkg", "f.m"])).toBeNull();
  });
});

import { describe, it, expect } from "vitest";
import { globToRegExp } from "../numbl-core/helpers/glob.js";

describe("globToRegExp", () => {
  it("matches single names", () => {
    const re = globToRegExp("a*.m");
    expect(re.test("abc.m")).toBe(true);
    expect(re.test("a.m")).toBe(true);
    expect(re.test("abc.mat")).toBe(false);
    expect(re.test("xa.m")).toBe(false);
    expect(globToRegExp("f?.txt").test("f1.txt")).toBe(true);
    expect(globToRegExp("f?.txt").test("f12.txt")).toBe(false);
    expect(globToRegExp("a+b(1).m").test("a+b(1).m")).toBe(true);
  });

  it("matches paths", () => {
    const re = globToRegExp("**/*.mex*", { paths: true });
    expect(re.test("f.mexa64")).toBe(true);
    expect(re.test("sub/dir/f.mexa64")).toBe(true);
    expect(re.test("sub/f.m")).toBe(false);
    expect(globToRegExp("*.m", { paths: true }).test("a/b.m")).toBe(false);
    expect(globToRegExp("**", { paths: true }).test("a/b.m")).toBe(true);
    expect(globToRegExp("a?c", { paths: true }).test("a/c")).toBe(false);
  });

  it("agrees with the former placeholder-based path conversion", () => {
    const old = (pattern: string) => {
      const DIRS = "\u0000";
      const ANY = "\u0001";
      return new RegExp(
        "^" +
          pattern
            .replace(/[.+^${}()|[\]\\]/g, "\\$&")
            .replace(/\*\*\//g, DIRS)
            .replace(/\*\*/g, ANY)
            .replace(/\*/g, "[^/]*")
            .replace(/\?/g, "[^/]")
            .replace(new RegExp(DIRS, "g"), "(?:.*/)?")
            .replace(new RegExp(ANY, "g"), ".*") +
          "$"
      ).source;
    };
    for (const p of [
      "**/*.m",
      "**",
      "**/sub/**/x?.m",
      "a/**",
      "*.m*",
      "**/**/a",
    ]) {
      expect(globToRegExp(p, { paths: true }).source).toBe(old(p));
    }
  });
});

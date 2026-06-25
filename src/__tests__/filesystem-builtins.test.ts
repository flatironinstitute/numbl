import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { VirtualFileSystem } from "../vfs/VirtualFileSystem.js";

const enc = (s: string) => new TextEncoder().encode(s);
const dec = (b: Uint8Array) => new TextDecoder().decode(b);

describe("VirtualFileSystem.copyfile", () => {
  it("copies a file, leaving the source in place", () => {
    const vfs = new VirtualFileSystem();
    vfs.writeFile("/a.txt", enc("hello"));
    expect(vfs.copyfile("/a.txt", "/b.txt")).toBe(true);
    expect(dec(vfs.readFile("/b.txt"))).toBe("hello");
    expect(dec(vfs.readFile("/a.txt"))).toBe("hello"); // source untouched
  });

  it("copies into an existing directory keeping the basename", () => {
    const vfs = new VirtualFileSystem();
    vfs.writeFile("/a.txt", enc("x"));
    vfs.mkdir("/d");
    expect(vfs.copyfile("/a.txt", "/d")).toBe(true);
    expect(dec(vfs.readFile("/d/a.txt"))).toBe("x");
  });

  it("copies a directory tree", () => {
    const vfs = new VirtualFileSystem();
    vfs.mkdir("/src");
    vfs.writeFile("/src/one.txt", enc("1"));
    vfs.writeFile("/src/sub/two.txt", enc("2"));
    expect(vfs.copyfile("/src", "/dst")).toBe(true);
    expect(dec(vfs.readFile("/dst/one.txt"))).toBe("1");
    expect(dec(vfs.readFile("/dst/sub/two.txt"))).toBe("2");
    expect(dec(vfs.readFile("/src/one.txt"))).toBe("1"); // source untouched
  });

  it("returns false for a missing source", () => {
    const vfs = new VirtualFileSystem();
    expect(vfs.copyfile("/nope.txt", "/x.txt")).toBe(false);
  });
});

describe("VirtualFileSystem.fileattrib", () => {
  it("resolves a file's path + attributes", () => {
    const vfs = new VirtualFileSystem();
    vfs.writeFile("/a.txt", enc("hi"));
    const fa = vfs.fileattrib("/a.txt");
    expect(fa).not.toBeNull();
    expect(fa!.Name).toContain("a.txt");
    expect(fa!.directory).toBe(false);
  });

  it("reports a directory", () => {
    const vfs = new VirtualFileSystem();
    vfs.mkdir("/d");
    const fa = vfs.fileattrib("/d");
    expect(fa).not.toBeNull();
    expect(fa!.directory).toBe(true);
  });

  it("returns null for a missing path", () => {
    const vfs = new VirtualFileSystem();
    expect(vfs.fileattrib("/missing")).toBeNull();
  });
});

describe("maxNumCompThreads builtin", () => {
  it("returns 1 (numbl is single-threaded)", () => {
    const result = executeCode("n = maxNumCompThreads;");
    expect(result.variableValues["n"]).toBe(1);
  });

  it("treats maxNumCompThreads(N) as a no-op returning 1", () => {
    const result = executeCode("n = maxNumCompThreads(8);");
    expect(result.variableValues["n"]).toBe(1);
  });
});

import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import type { FileIOAdapter } from "../numbl-core/fileIOAdapter.js";
import { VirtualFileSystem } from "../vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "../vfs/BrowserFileIOAdapter.js";

const enc = (s: string) => new TextEncoder().encode(s);
const dec = (b: Uint8Array) => new TextDecoder().decode(b);

const run = (code: string, fileIO: FileIOAdapter) =>
  executeCode(code, { fileIO }).output.join("");

describe("FileTable", () => {
  it("keeps fids open across runs that share an adapter", () => {
    const vfs = new VirtualFileSystem();
    const io = new BrowserFileIOAdapter(vfs);
    run("fid = fopen('/a.txt', 'w'); fprintf(fid, 'one\\n');", io);
    // Left open, but flushed when the run ended.
    expect(dec(vfs.readFile("/a.txt"))).toBe("one\n");
    const out = run(
      "fprintf(3, 'two\\n'); disp(fopen('all')); fclose(3); disp(isempty(fopen('all')))",
      io
    );
    expect(out.replace(/\s+/g, " ").trim()).toBe("3 1");
    expect(dec(vfs.readFile("/a.txt"))).toBe("one\ntwo\n");
  });

  it("does not modify a host buffer it was handed by readFile", () => {
    const vfs = new VirtualFileSystem();
    const original = enc("abc");
    vfs.writeFile("/a.txt", original);
    run(
      "fid = fopen('/a.txt', 'r+'); fwrite(fid, 'X'); fclose(fid);",
      new BrowserFileIOAdapter(vfs)
    );
    expect(dec(original)).toBe("abc");
    expect(dec(vfs.readFile("/a.txt"))).toBe("Xbc");
  });

  it("works with hosts that only implement the legacy stream methods", () => {
    const files = new Map<string, Uint8Array>();
    const open = new Map<
      number,
      { path: string; pos: number; write: boolean }
    >();
    let next = 3;
    const legacy: FileIOAdapter = {
      fopen(path, perm) {
        if (perm.startsWith("r") && !files.has(path)) return -1;
        if (perm.startsWith("w")) files.set(path, new Uint8Array(0));
        open.set(next, { path, pos: 0, write: perm.startsWith("w") });
        return next++;
      },
      fclose(fid) {
        open.delete(fid as number);
        return 0;
      },
      freadBytes(fid, count) {
        const e = open.get(fid)!;
        const data = files.get(e.path)!;
        const out = data.slice(e.pos, e.pos + count);
        e.pos += out.length;
        return out;
      },
      fwriteBytes(fid, data) {
        const e = open.get(fid)!;
        const old = files.get(e.path)!;
        const merged = new Uint8Array(old.length + data.length);
        merged.set(old);
        merged.set(data, old.length);
        files.set(e.path, merged);
        return data.length;
      },
    };
    const out = run(
      "fid = fopen('a.txt', 'w'); fprintf(fid, 'hi\\nthere\\n'); fclose(fid);" +
        "fid = fopen('a.txt'); disp(fgetl(fid)); disp(fgetl(fid)); fclose(fid);" +
        "disp(fileread('a.txt'))",
      legacy
    );
    expect(out).toBe("hi\nthere\nhi\nthere\n\n");
    expect(dec(files.get("a.txt")!)).toBe("hi\nthere\n");
  });
});

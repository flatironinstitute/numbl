/**
 * Browser implementation of FileIOAdapter using an in-memory VirtualFileSystem.
 * Mirrors NodeFileIOAdapter patterns from cli-fileio.ts. Stream semantics
 * (fids, positions, EOF) live in the core FileTable.
 */

import { unzipSync, zipSync } from "fflate";
import type {
  FileIOAdapter,
  WebOptions,
  FileAttribInfo,
} from "../numbl-core/fileIOAdapter.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { VirtualFileSystem, type VfsChanges } from "./VirtualFileSystem.js";
import { globToRegExp } from "../numbl-core/helpers/glob.js";
import { corsProxiedUrl } from "../browser/mip.js";

const TEXT_DECODER = new TextDecoder("utf-8");

export class BrowserFileIOAdapter implements FileIOAdapter {
  constructor(private vfs: VirtualFileSystem) {
    // tempdir() always exists, as on a real system.
    vfs.mkdir(this.tempdir());
  }

  readFile(path: string): Uint8Array {
    if (this.vfs.exists(path) !== "file")
      throw Object.assign(new Error(`File not found: ${path}`), {
        code: "ENOENT",
      });
    return this.vfs.readFile(path);
  }

  writeFile(path: string, data: Uint8Array): void {
    this.vfs.writeFile(path, data);
  }

  setFileMode(path: string, mode: { write?: boolean }): boolean {
    if (this.vfs.exists(path) === null) return false;
    if (mode.write !== undefined) this.vfs.setWritable(path, mode.write);
    return true;
  }

  scanDirectory(dirPath: string): WorkspaceFile[] {
    const norm = this.vfs.normalizePath(dirPath);
    const prefix = norm === "/" ? "/" : norm + "/";
    const results: WorkspaceFile[] = [];

    // Mirrors cli-scan.ts: only top-level .m/.numbl.js/.wasm files, plus
    // files inside @class/, +pkg/, and private/ subdirectories (recursively).
    // Files are stored with their absolute VFS path in `name` so that
    // search-path prefix matching in the code index works correctly.
    for (const filePath of this.vfs.allFiles()) {
      if (!filePath.startsWith(prefix)) continue;
      const relativePath = filePath.slice(prefix.length);
      if (!relativePath) continue;

      // Walk path segments. Allow any depth as long as every intermediate
      // segment is a special directory (@…, +…, or "private").
      const segments = relativePath.split("/");
      const dirs = segments.slice(0, -1);
      const allSpecial = dirs.every(
        d => d.startsWith("@") || d.startsWith("+") || d === "private"
      );
      if (!allSpecial) continue;

      if (
        !relativePath.endsWith(".m") &&
        !relativePath.endsWith(".numbl.js") &&
        !relativePath.endsWith(".wasm")
      ) {
        continue;
      }

      try {
        const content = this.vfs.readFile(filePath);
        if (relativePath.endsWith(".wasm")) {
          results.push({
            name: filePath,
            source: "",
            data: new Uint8Array(content),
          });
        } else {
          results.push({
            name: filePath,
            source: TEXT_DECODER.decode(content),
          });
        }
      } catch {
        // skip unreadable files
      }
    }
    return results;
  }

  resolvePath(dirPath: string): string {
    return this.vfs.normalizePath(dirPath);
  }

  existsPath(path: string): "file" | "dir" | null {
    return this.vfs.exists(path);
  }

  mkdir(dirPath: string): boolean {
    return this.vfs.mkdir(dirPath);
  }

  tempdir(): string {
    return "/tmp";
  }

  userpath(): string {
    return "/system";
  }

  deleteFile(pattern: string): void {
    const norm = this.vfs.normalizePath(pattern);
    if (norm.includes("*") || norm.includes("?")) {
      // Glob matching
      const lastSlash = norm.lastIndexOf("/");
      const dir = norm.slice(0, lastSlash);
      const globPat = norm.slice(lastSlash + 1);
      const re = globToRegExp(globPat);
      const entries = this.vfs.listDir(dir);
      for (const entry of entries) {
        if (!entry.isdir && re.test(entry.name)) {
          this.vfs.deleteFile(dir + "/" + entry.name);
        }
      }
    } else {
      this.vfs.deleteFile(norm);
    }
  }

  rmdir(dirPath: string, recursive: boolean): boolean {
    return this.vfs.rmdir(dirPath, recursive);
  }

  movefile(source: string, destination: string, force: boolean): boolean {
    if (!force && this.replacesReadOnly(source, destination)) return false;
    return this.vfs.movefile(source, destination);
  }

  copyfile(source: string, destination: string, force: boolean): boolean {
    if (!force && this.replacesReadOnly(source, destination)) return false;
    return this.vfs.copyfile(source, destination);
  }

  /** Whether moving/copying source to destination would overwrite a
   *  read-only file, which MATLAB only does with the 'f' option. */
  private replacesReadOnly(source: string, destination: string): boolean {
    const src = this.vfs.normalizePath(source);
    let dst = this.vfs.normalizePath(destination);
    if (this.vfs.exists(dst) === "dir")
      dst =
        (dst === "/" ? "" : dst) + "/" + src.slice(src.lastIndexOf("/") + 1);
    return this.vfs.exists(dst) === "file" && !this.vfs.isWritable(dst);
  }

  fileattrib(path: string): FileAttribInfo | null {
    return this.vfs.fileattrib(path);
  }

  listDir(dirPath: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    const norm = this.vfs.normalizePath(dirPath);

    // Handle ** recursive search
    if (dirPath.includes("**")) {
      return this.listDirRecursive(dirPath);
    }

    // Handle glob wildcards
    if (dirPath.includes("*") || dirPath.includes("?")) {
      return this.listDirGlob(dirPath);
    }

    const type = this.vfs.exists(dirPath);
    if (type === "file") {
      const lastSlash = norm.lastIndexOf("/");
      const name = lastSlash >= 0 ? norm.slice(lastSlash + 1) : norm;
      const folder = lastSlash >= 0 ? norm.slice(0, lastSlash) : "/";
      return [
        {
          name,
          folder,
          bytes: this.vfs.fileSize(dirPath),
          isdir: false,
          mtimeMs: Date.now(),
        },
      ];
    }

    return this.vfs.listDir(dirPath);
  }

  websave(url: string, filename: string, options?: WebOptions): void {
    url = corsProxiedUrl(url);
    const method = options?.requestMethod?.toUpperCase() || "GET";
    const xhr = new XMLHttpRequest();
    xhr.open(method, url, false);
    if (options?.timeout) xhr.timeout = Math.round(options.timeout * 1000);
    if (options?.username && options?.password) {
      xhr.setRequestHeader(
        "Authorization",
        "Basic " + btoa(`${options.username}:${options.password}`)
      );
    }
    if (options?.keyName && options?.keyValue) {
      xhr.setRequestHeader(options.keyName, options.keyValue);
    }
    xhr.responseType = "arraybuffer";
    xhr.send();
    if (xhr.status < 200 || xhr.status >= 300) {
      throw new Error(`websave: HTTP ${xhr.status} for ${url}`);
    }
    const data = new Uint8Array(xhr.response as ArrayBuffer);
    this.vfs.writeFile(filename, data);
  }

  webread(url: string, options?: WebOptions): string {
    url = corsProxiedUrl(url);
    const method = options?.requestMethod?.toUpperCase() || "GET";
    const xhr = new XMLHttpRequest();
    xhr.open(method, url, false);
    if (options?.timeout) xhr.timeout = Math.round(options.timeout * 1000);
    if (options?.username && options?.password) {
      xhr.setRequestHeader(
        "Authorization",
        "Basic " + btoa(`${options.username}:${options.password}`)
      );
    }
    if (options?.keyName && options?.keyValue) {
      xhr.setRequestHeader(options.keyName, options.keyValue);
    }
    xhr.send();
    if (xhr.status < 200 || xhr.status >= 300) {
      throw new Error(`webread: HTTP ${xhr.status} for ${url}`);
    }
    return xhr.responseText;
  }

  unzip(zipfilename: string, outputfolder: string): string[] {
    zipfilename = corsProxiedUrl(zipfilename);

    const zipData = this.vfs.readFile(zipfilename);
    this.vfs.mkdir(outputfolder);

    const files = unzipSync(zipData);
    const extracted: string[] = [];

    for (const [entryName, fileData] of Object.entries(files)) {
      if (entryName.endsWith("/")) {
        this.vfs.mkdir(outputfolder + "/" + entryName);
        continue;
      }
      const outPath = outputfolder + "/" + entryName;
      const lastSlash = outPath.lastIndexOf("/");
      if (lastSlash > 0) {
        this.vfs.mkdir(outPath.slice(0, lastSlash));
      }
      this.vfs.writeFile(outPath, fileData);
      extracted.push(this.vfs.normalizePath(outPath));
    }

    return extracted;
  }

  zip(zipfilename: string, filenames: string[], rootDir: string): string[] {
    const root = this.vfs.normalizePath(rootDir);
    const dest = this.vfs.normalizePath(zipfilename);
    const entries: Record<string, Uint8Array> = {};
    const names: string[] = [];

    const toEntryName = (abs: string): string => {
      const prefix = root === "/" ? "/" : root + "/";
      // Files outside rootDir are stored under their basename.
      if (!abs.startsWith(prefix)) return abs.slice(abs.lastIndexOf("/") + 1);
      return abs.slice(prefix.length);
    };
    const addFile = (abs: string) => {
      const name = toEntryName(abs);
      entries[name] = this.vfs.readFile(abs);
      names.push(name);
    };
    const addDir = (absDir: string) => {
      for (const entry of this.vfs.listDir(absDir)) {
        if (entry.name === "." || entry.name === "..") continue;
        const abs = absDir + "/" + entry.name;
        if (entry.isdir) addDir(abs);
        else addFile(abs);
      }
    };
    const addPath = (abs: string) => {
      const type = this.vfs.exists(abs);
      if (type === null) throw new Error(`zip: cannot find '${abs}'`);
      if (type === "dir") addDir(abs);
      else addFile(abs);
    };

    for (const f of filenames) {
      const p = f.replace(/\\/g, "/");
      const abs = p.startsWith("/")
        ? this.vfs.normalizePath(p)
        : this.vfs.normalizePath(root + "/" + p);
      const last = p.split("/").pop() ?? p;
      if (last.includes("*") || last.includes("?")) {
        // Simple glob in the final path component
        const lastSlash = abs.lastIndexOf("/");
        const dir = lastSlash > 0 ? abs.slice(0, lastSlash) : "/";
        const re = globToRegExp(last);
        if (this.vfs.exists(dir) !== "dir") continue;
        for (const entry of this.vfs.listDir(dir)) {
          if (entry.name === "." || entry.name === "..") continue;
          if (re.test(entry.name)) addPath(dir + "/" + entry.name);
        }
      } else {
        addPath(abs);
      }
    }

    this.vfs.writeFile(dest, zipSync(entries));
    return names;
  }

  /** Get the VFS changes for syncing back to the main thread. Open files
   *  are flushed by executeCode when each run ends. */
  getChanges(): VfsChanges {
    return this.vfs.getChanges();
  }

  private listDirGlob(pattern: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    const norm = this.vfs.normalizePath(pattern);
    const lastSlash = norm.lastIndexOf("/");
    const dir = lastSlash >= 0 ? norm.slice(0, lastSlash) : "/";
    const globPat = lastSlash >= 0 ? norm.slice(lastSlash + 1) : norm;
    const re = globToRegExp(globPat);
    const entries = this.vfs.listDir(dir);
    return entries.filter(
      e => e.name !== "." && e.name !== ".." && re.test(e.name)
    );
  }

  private listDirRecursive(pattern: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    // ** means recursive. Extract the base directory (everything before **)
    // and the pattern from ** onward, which is matched against each entry's
    // path relative to the base (so `base/**/*.mex*` only returns entries
    // whose name matches `*.mex*`, at any depth including the base itself).
    const norm = pattern.replace(/\\/g, "/");
    const idx = norm.indexOf("**");
    let baseDir = norm.slice(0, idx);
    if (baseDir.endsWith("/")) baseDir = baseDir.slice(0, -1);
    if (!baseDir) baseDir = ".";
    const absBase = this.vfs.normalizePath(baseDir);
    if (this.vfs.exists(absBase) !== "dir") return [];

    const suffix = norm.slice(idx);
    const re = globToRegExp(suffix, { paths: true });

    const results: {
      name: string;
      folder: string;
      bytes: number;
      isdir: boolean;
      mtimeMs: number;
    }[] = [];

    const walkDir = (dir: string, rel: string) => {
      const relPrefix = rel ? rel + "/" : "";
      for (const entry of this.vfs.listDir(dir)) {
        if (re.test(relPrefix + entry.name)) results.push(entry);
        if (entry.isdir && entry.name !== "." && entry.name !== "..") {
          walkDir(entry.folder + "/" + entry.name, relPrefix + entry.name);
        }
      }
    };

    walkDir(absBase, "");
    return results;
  }
}

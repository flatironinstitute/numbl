/**
 * Browser implementation of FileIOAdapter using an in-memory VirtualFileSystem.
 * Mirrors NodeFileIOAdapter patterns from cli-fileio.ts.
 */

import { unzipSync } from "fflate";
import type { FileIOAdapter, WebOptions } from "../numbl-core/fileIOAdapter.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { VirtualFileSystem, type VfsChanges } from "./VirtualFileSystem.js";

interface VFSOpenFile {
  path: string;
  permission: string;
  lastError: string;
  buffer: string; // text-mode read buffer
  eof: boolean;
  pos: number; // byte position
  data: Uint8Array; // file contents snapshot (for reading) or growing buffer (for writing)
  dataLen: number; // actual used length in data (may be < data.length due to pre-allocation)
  dirty: boolean;
}

const TEXT_DECODER = new TextDecoder("utf-8");
const TEXT_ENCODER = new TextEncoder();
const READ_CHUNK_SIZE = 8192;

export class BrowserFileIOAdapter implements FileIOAdapter {
  private nextFid = 3; // 0=stdin, 1=stdout, 2=stderr reserved
  private openFiles = new Map<number, VFSOpenFile>();

  constructor(private vfs: VirtualFileSystem) {}

  fopen(filename: string, permission: string): number {
    // Strip 'b' suffix for binary mode (handled the same way internally)
    const perm = permission.replace(/b/g, "");
    const path = this.vfs.normalizePath(filename);

    try {
      let data: Uint8Array;
      let pos = 0;

      if (perm === "r" || perm === "r+") {
        if (this.vfs.exists(filename) !== "file") return -1;
        data = new Uint8Array(this.vfs.readFile(filename));
      } else if (perm === "w" || perm === "w+") {
        data = new Uint8Array(0);
      } else if (perm === "a" || perm === "a+") {
        if (this.vfs.exists(filename) === "file") {
          data = new Uint8Array(this.vfs.readFile(filename));
          pos = data.length;
        } else {
          data = new Uint8Array(0);
        }
      } else {
        return -1;
      }

      const fid = this.nextFid++;
      this.openFiles.set(fid, {
        path,
        permission: perm,
        lastError: "",
        buffer: "",
        eof: false,
        pos,
        data,
        dataLen: data.length,
        dirty: perm === "w" || perm === "a", // w starts dirty (truncated or new)
      });
      return fid;
    } catch {
      return -1;
    }
  }

  fclose(fidOrAll: number | "all"): number {
    if (fidOrAll === "all") {
      for (const [, entry] of this.openFiles) {
        this.flushEntry(entry);
      }
      this.openFiles.clear();
      return 0;
    }
    const entry = this.openFiles.get(fidOrAll);
    if (!entry) return -1;
    this.flushEntry(entry);
    this.openFiles.delete(fidOrAll);
    return 0;
  }

  fgetl(fid: number): string | number {
    const entry = this.getEntry(fid);
    if (!entry) return -1;
    return this.readLine(entry, false);
  }

  fgets(fid: number): string | number {
    const entry = this.getEntry(fid);
    if (!entry) return -1;
    return this.readLine(entry, true);
  }

  fileread(filename: string): string {
    const data = this.vfs.readFile(filename);
    return TEXT_DECODER.decode(data);
  }

  feof(fid: number): number {
    const entry = this.getEntry(fid);
    if (!entry) return 1;
    if (entry.buffer.length > 0) return 0;
    if (entry.eof) return 1;
    // Check if we're at the end
    if (entry.pos >= entry.dataLen) {
      entry.eof = true;
      return 1;
    }
    return 0;
  }

  ferror(fid: number): string {
    const entry = this.getEntry(fid);
    if (!entry) return "Invalid file identifier";
    return entry.lastError;
  }

  fwrite(fid: number, text: string): void {
    const entry = this.getEntry(fid);
    if (!entry) throw new Error(`Invalid file identifier: ${fid}`);
    const bytes = TEXT_ENCODER.encode(text);
    this.writeBytes(entry, bytes);
    entry.lastError = "";
  }

  freadBytes(fid: number, count: number): Uint8Array {
    const entry = this.getEntry(fid);
    if (!entry) throw new Error(`Invalid file identifier: ${fid}`);
    // Clear text buffer when switching to binary mode
    if (entry.buffer.length > 0) {
      entry.buffer = "";
    }
    const available = entry.dataLen - entry.pos;
    const toRead = Math.min(count, available);
    if (toRead <= 0) {
      entry.eof = true;
      return new Uint8Array(0);
    }
    const result = new Uint8Array(toRead);
    result.set(entry.data.subarray(entry.pos, entry.pos + toRead));
    entry.pos += toRead;
    if (entry.pos >= entry.dataLen) entry.eof = true;
    entry.lastError = "";
    return result;
  }

  fwriteBytes(fid: number, data: Uint8Array): number {
    const entry = this.getEntry(fid);
    if (!entry) throw new Error(`Invalid file identifier: ${fid}`);
    this.writeBytes(entry, data);
    entry.lastError = "";
    return data.length;
  }

  fseek(fid: number, offset: number, origin: number): number {
    const entry = this.getEntry(fid);
    if (!entry) return -1;
    entry.buffer = "";
    entry.eof = false;
    if (origin === -1) {
      // SEEK_SET (bof)
      entry.pos = offset;
    } else if (origin === 0) {
      // SEEK_CUR (cof)
      entry.pos += offset;
    } else {
      // SEEK_END (eof)
      entry.pos = entry.dataLen + offset;
    }
    return 0;
  }

  ftell(fid: number): number {
    const entry = this.getEntry(fid);
    if (!entry) return -1;
    return entry.pos;
  }

  scanDirectory(dirPath: string): WorkspaceFile[] {
    const norm = this.vfs.normalizePath(dirPath);
    const prefix = norm === "/" ? "/" : norm + "/";
    const results: WorkspaceFile[] = [];

    for (const filePath of this.vfs.allFiles()) {
      if (!filePath.startsWith(prefix)) continue;
      const relativePath = filePath.slice(prefix.length);
      if (
        relativePath.endsWith(".m") ||
        relativePath.endsWith(".js") ||
        relativePath.endsWith(".wasm")
      ) {
        try {
          const content = this.vfs.readFile(filePath);
          if (relativePath.endsWith(".wasm")) {
            results.push({
              name: relativePath,
              source: "",
              data: new Uint8Array(content),
            });
          } else {
            results.push({
              name: relativePath,
              source: TEXT_DECODER.decode(content),
            });
          }
        } catch {
          // skip unreadable files
        }
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

  deleteFile(pattern: string): void {
    const norm = this.vfs.normalizePath(pattern);
    if (norm.includes("*") || norm.includes("?")) {
      // Glob matching
      const lastSlash = norm.lastIndexOf("/");
      const dir = norm.slice(0, lastSlash);
      const globPat = norm.slice(lastSlash + 1);
      const re = new RegExp(
        "^" +
          globPat
            .replace(/[.+^${}()|[\]\\]/g, "\\$&")
            .replace(/\*/g, ".*")
            .replace(/\?/g, ".") +
          "$"
      );
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

  movefile(source: string, destination: string): boolean {
    return this.vfs.movefile(source, destination);
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
    url = filterUrl(url);
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
    url = filterUrl(url);
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
    zipfilename = filterUrl(zipfilename);

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

  /** Get the VFS changes for syncing back to the main thread. */
  getChanges(): VfsChanges {
    // Flush all open files first
    for (const [, entry] of this.openFiles) {
      this.flushEntry(entry);
    }
    return this.vfs.getChanges();
  }

  private getEntry(fid: number): VFSOpenFile | undefined {
    return this.openFiles.get(fid);
  }

  private flushEntry(entry: VFSOpenFile): void {
    if (entry.dirty) {
      const finalData =
        entry.dataLen === entry.data.length
          ? entry.data
          : entry.data.subarray(0, entry.dataLen);
      this.vfs.writeFile(entry.path, finalData);
      entry.dirty = false;
    }
  }

  private writeBytes(entry: VFSOpenFile, bytes: Uint8Array): void {
    const needed = entry.pos + bytes.length;
    if (needed > entry.data.length) {
      // Grow the buffer
      const newSize = Math.max(needed, entry.data.length * 2, 256);
      const newData = new Uint8Array(newSize);
      newData.set(entry.data.subarray(0, entry.dataLen));
      entry.data = newData;
    }
    entry.data.set(bytes, entry.pos);
    entry.pos += bytes.length;
    if (entry.pos > entry.dataLen) {
      entry.dataLen = entry.pos;
    }
    entry.dirty = true;
  }

  private readLine(entry: VFSOpenFile, keepNewline: boolean): string | number {
    // Fill text buffer from binary data
    while (!entry.eof) {
      const nlIdx = entry.buffer.indexOf("\n");
      if (nlIdx !== -1) break;

      const available = entry.dataLen - entry.pos;
      if (available <= 0) {
        entry.eof = true;
        break;
      }
      const toRead = Math.min(READ_CHUNK_SIZE, available);
      const chunk = entry.data.subarray(entry.pos, entry.pos + toRead);
      entry.pos += toRead;
      entry.buffer += TEXT_DECODER.decode(chunk, { stream: true });
    }

    if (entry.buffer.length === 0) {
      return -1;
    }

    const nlIdx = entry.buffer.indexOf("\n");
    if (nlIdx !== -1) {
      const line = keepNewline
        ? entry.buffer.slice(0, nlIdx + 1)
        : entry.buffer.slice(0, nlIdx);
      entry.buffer = entry.buffer.slice(nlIdx + 1);
      return line;
    }

    // No newline found but we have data (EOF mid-line)
    const line = entry.buffer;
    entry.buffer = "";
    return line;
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
    const re = new RegExp(
      "^" +
        globPat
          .replace(/[.+^${}()|[\]\\]/g, "\\$&")
          .replace(/\*/g, ".*")
          .replace(/\?/g, ".") +
        "$"
    );
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
    const idx = pattern.indexOf("**");
    let baseDir = pattern.slice(0, idx);
    if (baseDir.endsWith("/")) baseDir = baseDir.slice(0, -1);
    if (!baseDir) baseDir = ".";

    const results: {
      name: string;
      folder: string;
      bytes: number;
      isdir: boolean;
      mtimeMs: number;
    }[] = [];

    const walkDir = (dir: string) => {
      const entries = this.vfs.listDir(dir);
      results.push(...entries);
      for (const entry of entries) {
        if (entry.isdir && entry.name !== "." && entry.name !== "..") {
          walkDir(entry.folder + "/" + entry.name);
        }
      }
    };

    walkDir(this.vfs.normalizePath(baseDir));
    return results;
  }
}

function filterUrl(url: string): string {
  // If the url is of the form
  // https://github.com/*/releases/download/*
  // then we need to route through a CORS proxy since GitHub doesn't send CORS headers on release assets.
  if (/^https:\/\/github\.com\/.+\/releases\/download\/.+/.test(url)) {
    url = url.replace(
      "https://github.com/",
      "https://mip-cors-proxy.figurl.workers.dev/gh/"
    );
  }
  return url;
}

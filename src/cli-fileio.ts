/* eslint-disable @typescript-eslint/no-unused-vars */
/**
 * Node.js implementation of FileIOAdapter using synchronous fs operations.
 */

import {
  openSync,
  closeSync,
  readSync,
  writeSync,
  readFileSync,
  writeFileSync,
  statSync,
  fstatSync,
  mkdirSync,
  unlinkSync,
  readdirSync,
  rmSync,
  rmdirSync,
} from "fs";
import { unzipSync } from "fflate";
import { execFileSync } from "child_process";
import { homedir, tmpdir } from "os";
import { join, resolve, dirname, basename } from "path";
import type { FileIOAdapter, WebOptions } from "./numbl-core/fileIOAdapter.js";
import { scanMFiles } from "./cli-scan.js";

function expandTilde(filepath: string): string {
  if (filepath.startsWith("~/")) {
    return join(homedir(), filepath.slice(2));
  }
  return filepath;
}

interface OpenFile {
  fd: number;
  permission: string;
  lastError: string;
  buffer: string;
  eof: boolean;
  /** Byte offset tracked for binary I/O (fread/fwrite/fseek/ftell). -1 means not tracking (text mode). */
  pos: number;
}

function permissionToFlags(permission: string): string {
  switch (permission) {
    case "r":
      return "r";
    case "w":
      return "w";
    case "a":
      return "a";
    case "r+":
      return "r+";
    case "w+":
      return "w+";
    case "a+":
      return "a+";
    default:
      return permission; // pass through for binary modes like 'rb'
  }
}

const READ_CHUNK_SIZE = 8192;

/** Build fetch init options from WebOptions. */
function buildFetchOptions(options?: WebOptions): Record<string, unknown> {
  const init: Record<string, unknown> = {};
  if (options?.requestMethod) {
    init.method = options.requestMethod.toUpperCase();
  }
  const headers: Record<string, string> = {};
  if (options?.username && options?.password) {
    headers["Authorization"] =
      "Basic " +
      Buffer.from(`${options.username}:${options.password}`).toString("base64");
  }
  if (options?.keyName && options?.keyValue) {
    headers[options.keyName] = options.keyValue;
  }
  if (Object.keys(headers).length > 0) init.headers = headers;
  return init;
}

export class NodeFileIOAdapter implements FileIOAdapter {
  private nextFid = 3; // 0=stdin, 1=stdout, 2=stderr reserved
  private openFiles = new Map<number, OpenFile>();

  fopen(filename: string, permission: string): number {
    try {
      const flags = permissionToFlags(permission);
      const fd = openSync(expandTilde(filename), flags);
      const fid = this.nextFid++;
      this.openFiles.set(fid, {
        fd,
        permission,
        lastError: "",
        buffer: "",
        eof: false,
        pos: 0,
      });
      return fid;
    } catch (e) {
      return -1;
    }
  }

  fclose(fidOrAll: number | "all"): number {
    if (fidOrAll === "all") {
      for (const [, entry] of this.openFiles) {
        try {
          closeSync(entry.fd);
        } catch {
          // ignore close errors during close-all
        }
      }
      this.openFiles.clear();
      return 0;
    }
    const entry = this.openFiles.get(fidOrAll);
    if (!entry) return -1;
    try {
      closeSync(entry.fd);
    } catch {
      return -1;
    }
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
    return readFileSync(expandTilde(filename), "utf-8");
  }

  feof(fid: number): number {
    const entry = this.getEntry(fid);
    if (!entry) return 1;
    if (entry.buffer.length > 0) return 0;
    if (entry.eof) return 1;
    // Buffer is empty and eof not yet detected — try reading to check
    const buf = Buffer.alloc(1);
    try {
      const bytesRead = readSync(entry.fd, buf, 0, 1, null);
      if (bytesRead === 0) {
        entry.eof = true;
        return 1;
      }
      // Got data — put it back in the buffer
      entry.buffer = buf.toString("utf-8", 0, bytesRead);
      return 0;
    } catch {
      entry.eof = true;
      return 1;
    }
  }

  ferror(fid: number): string {
    const entry = this.getEntry(fid);
    if (!entry) return "Invalid file identifier";
    return entry.lastError;
  }

  fwrite(fid: number, text: string): void {
    const entry = this.getEntry(fid);
    if (!entry) throw new Error(`Invalid file identifier: ${fid}`);
    try {
      writeSync(entry.fd, text);
      entry.lastError = "";
    } catch (e) {
      entry.lastError = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  freadBytes(fid: number, count: number): Uint8Array {
    const entry = this.getEntry(fid);
    if (!entry) throw new Error(`Invalid file identifier: ${fid}`);
    // Clear any text buffer (switching to binary mode)
    if (entry.buffer.length > 0) {
      entry.buffer = "";
    }
    const buf = Buffer.alloc(count);
    try {
      const bytesRead = readSync(entry.fd, buf, 0, count, entry.pos);
      entry.pos += bytesRead;
      if (bytesRead === 0) entry.eof = true;
      else if (bytesRead < count) entry.eof = true;
      entry.lastError = "";
      return new Uint8Array(buf.buffer, buf.byteOffset, bytesRead);
    } catch (e) {
      entry.lastError = e instanceof Error ? e.message : String(e);
      entry.eof = true;
      return new Uint8Array(0);
    }
  }

  fwriteBytes(fid: number, data: Uint8Array): number {
    const entry = this.getEntry(fid);
    if (!entry) throw new Error(`Invalid file identifier: ${fid}`);
    try {
      const written = writeSync(entry.fd, data, 0, data.length, entry.pos);
      entry.pos += written;
      entry.lastError = "";
      return written;
    } catch (e) {
      entry.lastError = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  fseek(fid: number, offset: number, origin: number): number {
    const entry = this.getEntry(fid);
    if (!entry) return -1;
    entry.buffer = "";
    entry.eof = false;
    try {
      if (origin === -1) {
        // SEEK_SET (bof)
        entry.pos = offset;
      } else if (origin === 0) {
        // SEEK_CUR (cof)
        entry.pos += offset;
      } else {
        // SEEK_END (eof)
        const size = fstatSync(entry.fd).size;
        entry.pos = size + offset;
      }
      return 0;
    } catch {
      return -1;
    }
  }

  ftell(fid: number): number {
    const entry = this.getEntry(fid);
    if (!entry) return -1;
    return entry.pos;
  }

  scanDirectory(dirPath: string) {
    return scanMFiles(expandTilde(dirPath));
  }

  resolvePath(dirPath: string) {
    return resolve(process.cwd(), expandTilde(dirPath));
  }

  existsPath(path: string): "file" | "dir" | null {
    try {
      const s = statSync(expandTilde(path));
      return s.isDirectory() ? "dir" : "file";
    } catch {
      return null;
    }
  }

  mkdir(dirPath: string): boolean {
    try {
      mkdirSync(expandTilde(dirPath), { recursive: true });
      return true;
    } catch {
      return false;
    }
  }

  tempdir(): string {
    return tmpdir();
  }

  websave(url: string, filename: string, options?: WebOptions): void {
    const dest = expandTilde(filename);
    const timeoutMs = options?.timeout
      ? Math.round(options.timeout * 1000)
      : 30000;
    // Use a child node process with fetch to perform a synchronous download
    const fetchOpts = buildFetchOptions(options);
    const script = `
      const ac = new AbortController();
      const t = setTimeout(() => ac.abort(), ${timeoutMs});
      fetch(${JSON.stringify(url)}, { ...${JSON.stringify(fetchOpts)}, signal: ac.signal })
        .then(r => { clearTimeout(t); if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); })
        .then(buf => require('fs').writeFileSync(${JSON.stringify(dest)}, Buffer.from(buf)))
        .catch(e => { process.stderr.write(e.message + '\\n'); process.exit(1); });
    `;
    try {
      execFileSync(process.execPath, ["-e", script], {
        stdio: ["ignore", "ignore", "pipe"],
        timeout: timeoutMs + 5000,
      });
    } catch (e: unknown) {
      const msg =
        e instanceof Error && "stderr" in e
          ? (e as { stderr: Buffer }).stderr.toString().trim()
          : String(e);
      throw new Error(`websave: failed to download ${url}: ${msg}`);
    }
  }

  webread(url: string, options?: WebOptions): string {
    const timeoutMs = options?.timeout
      ? Math.round(options.timeout * 1000)
      : 30000;
    // Use a child node process with fetch to perform a synchronous download
    const fetchOpts = buildFetchOptions(options);
    const script = `
      const ac = new AbortController();
      const t = setTimeout(() => ac.abort(), ${timeoutMs});
      fetch(${JSON.stringify(url)}, { ...${JSON.stringify(fetchOpts)}, signal: ac.signal })
        .then(r => { clearTimeout(t); if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
        .then(t => process.stdout.write(t))
        .catch(e => { process.stderr.write(e.message + '\\n'); process.exit(1); });
    `;
    try {
      const result = execFileSync(process.execPath, ["-e", script], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: timeoutMs + 5000,
        maxBuffer: 50 * 1024 * 1024,
      });
      return result.toString();
    } catch (e: unknown) {
      const msg =
        e instanceof Error && "stderr" in e
          ? (e as { stderr: Buffer }).stderr.toString().trim()
          : String(e);
      throw new Error(`webread: failed to fetch ${url}: ${msg}`);
    }
  }

  rmdir(dirPath: string, recursive: boolean): boolean {
    try {
      const p = expandTilde(dirPath);
      if (recursive) {
        rmSync(p, { recursive: true });
      } else {
        rmdirSync(p);
      }
      return true;
    } catch {
      return false;
    }
  }

  deleteFile(pattern: string): void {
    const p = expandTilde(pattern);
    // Check for glob wildcards
    if (p.includes("*") || p.includes("?")) {
      const lastSep = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
      const dir = lastSep >= 0 ? p.slice(0, lastSep) : ".";
      const globPat = lastSep >= 0 ? p.slice(lastSep + 1) : p;
      // Convert glob to regex: * -> .*, ? -> .
      const re = new RegExp(
        "^" +
          globPat
            .replace(/[.+^${}()|[\]\\]/g, "\\$&")
            .replace(/\*/g, ".*")
            .replace(/\?/g, ".") +
          "$"
      );
      let entries: string[];
      try {
        entries = readdirSync(dir);
      } catch {
        return; // directory doesn't exist — nothing to delete
      }
      for (const entry of entries) {
        if (re.test(entry)) {
          try {
            unlinkSync(join(dir, entry));
          } catch {
            // skip files that can't be deleted
          }
        }
      }
    } else {
      try {
        unlinkSync(p);
      } catch {
        // MATLAB silently warns on nonexistent files; we just ignore
      }
    }
  }

  unzip(zipfilename: string, outputfolder: string): string[] {
    const src = expandTilde(zipfilename);
    const dest = resolve(expandTilde(outputfolder));
    mkdirSync(dest, { recursive: true });

    const buf = readFileSync(src);
    const files = unzipSync(
      new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
    );

    const extracted: string[] = [];
    for (const [entryName, data] of Object.entries(files)) {
      if (entryName.endsWith("/")) {
        mkdirSync(join(dest, entryName), { recursive: true });
        continue;
      }
      const outPath = join(dest, entryName);
      mkdirSync(dirname(outPath), { recursive: true });
      writeFileSync(outPath, data);
      extracted.push(outPath);
    }

    return extracted;
  }

  listDir(dirPath: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    const p = expandTilde(dirPath);

    // Check if path contains ** (recursive search)
    if (p.includes("**")) {
      return this.listDirRecursive(p);
    }

    // Check for glob wildcards in the filename part
    if (p.includes("*") || p.includes("?")) {
      return this.listDirGlob(p);
    }

    // Plain path — check if it's a directory or a file
    try {
      const s = statSync(p);
      if (s.isDirectory()) {
        // List contents of directory, including . and ..
        const absDir = resolve(p);
        const results: {
          name: string;
          folder: string;
          bytes: number;
          isdir: boolean;
          mtimeMs: number;
        }[] = [];
        // Add . and ..
        results.push({
          name: ".",
          folder: absDir,
          bytes: 0,
          isdir: true,
          mtimeMs: s.mtimeMs,
        });
        try {
          const parentStat = statSync(resolve(absDir, ".."));
          results.push({
            name: "..",
            folder: absDir,
            bytes: 0,
            isdir: true,
            mtimeMs: parentStat.mtimeMs,
          });
        } catch {
          results.push({
            name: "..",
            folder: absDir,
            bytes: 0,
            isdir: true,
            mtimeMs: 0,
          });
        }
        const entries = readdirSync(absDir);
        for (const entry of entries) {
          try {
            const es = statSync(join(absDir, entry));
            results.push({
              name: entry,
              folder: absDir,
              bytes: es.isDirectory() ? 0 : es.size,
              isdir: es.isDirectory(),
              mtimeMs: es.mtimeMs,
            });
          } catch {
            // skip entries we can't stat
          }
        }
        return results;
      } else {
        // Single file match
        const absPath = resolve(p);
        return [
          {
            name: basename(absPath),
            folder: dirname(absPath),
            bytes: s.size,
            isdir: false,
            mtimeMs: s.mtimeMs,
          },
        ];
      }
    } catch {
      // Path doesn't exist — return empty
      return [];
    }
  }

  private listDirGlob(pattern: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    const lastSep = Math.max(
      pattern.lastIndexOf("/"),
      pattern.lastIndexOf("\\")
    );
    const dir = lastSep >= 0 ? pattern.slice(0, lastSep) : ".";
    const globPat = lastSep >= 0 ? pattern.slice(lastSep + 1) : pattern;
    const re = new RegExp(
      "^" +
        globPat
          .replace(/[.+^${}()|[\]\\]/g, "\\$&")
          .replace(/\*/g, ".*")
          .replace(/\?/g, ".") +
        "$"
    );
    const absDir = resolve(dir);
    const results: {
      name: string;
      folder: string;
      bytes: number;
      isdir: boolean;
      mtimeMs: number;
    }[] = [];
    let entries: string[];
    try {
      entries = readdirSync(absDir);
    } catch {
      return [];
    }
    for (const entry of entries) {
      if (re.test(entry)) {
        try {
          const es = statSync(join(absDir, entry));
          results.push({
            name: entry,
            folder: absDir,
            bytes: es.isDirectory() ? 0 : es.size,
            isdir: es.isDirectory(),
            mtimeMs: es.mtimeMs,
          });
        } catch {
          // skip
        }
      }
    }
    return results;
  }

  private listDirRecursive(pattern: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    // ** means recursive. Extract the base directory (everything before **)
    const idx = pattern.indexOf("**");
    let baseDir = pattern.slice(0, idx);
    if (baseDir.endsWith("/") || baseDir.endsWith("\\")) {
      baseDir = baseDir.slice(0, -1);
    }
    if (!baseDir) baseDir = ".";
    const absBase = resolve(baseDir);

    const results: {
      name: string;
      folder: string;
      bytes: number;
      isdir: boolean;
      mtimeMs: number;
    }[] = [];

    const walkDir = (dir: string) => {
      // Add . and .. for this directory
      try {
        const ds = statSync(dir);
        results.push({
          name: ".",
          folder: dir,
          bytes: 0,
          isdir: true,
          mtimeMs: ds.mtimeMs,
        });
      } catch {
        results.push({
          name: ".",
          folder: dir,
          bytes: 0,
          isdir: true,
          mtimeMs: 0,
        });
      }
      try {
        const ps = statSync(resolve(dir, ".."));
        results.push({
          name: "..",
          folder: dir,
          bytes: 0,
          isdir: true,
          mtimeMs: ps.mtimeMs,
        });
      } catch {
        results.push({
          name: "..",
          folder: dir,
          bytes: 0,
          isdir: true,
          mtimeMs: 0,
        });
      }

      let entries: string[];
      try {
        entries = readdirSync(dir);
      } catch {
        return;
      }
      const subdirs: string[] = [];
      for (const entry of entries) {
        try {
          const es = statSync(join(dir, entry));
          results.push({
            name: entry,
            folder: dir,
            bytes: es.isDirectory() ? 0 : es.size,
            isdir: es.isDirectory(),
            mtimeMs: es.mtimeMs,
          });
          if (es.isDirectory()) subdirs.push(entry);
        } catch {
          // skip
        }
      }
      for (const sub of subdirs) {
        walkDir(join(dir, sub));
      }
    };

    walkDir(absBase);
    return results;
  }

  private getEntry(fid: number): OpenFile | undefined {
    return this.openFiles.get(fid);
  }

  /**
   * Read one line from the file's buffer, refilling from the fd as needed.
   * @param keepNewline - if true, keep the trailing newline (fgets); if false, strip it (fgetl)
   */
  private readLine(entry: OpenFile, keepNewline: boolean): string | number {
    // Fill buffer until we find a newline or hit EOF
    while (!entry.eof) {
      const nlIdx = entry.buffer.indexOf("\n");
      if (nlIdx !== -1) break; // we have a complete line

      const buf = Buffer.alloc(READ_CHUNK_SIZE);
      let bytesRead: number;
      try {
        bytesRead = readSync(entry.fd, buf, 0, READ_CHUNK_SIZE, null);
        entry.lastError = "";
      } catch (e) {
        entry.lastError = e instanceof Error ? e.message : String(e);
        entry.eof = true;
        break;
      }
      if (bytesRead === 0) {
        entry.eof = true;
        break;
      }
      entry.buffer += buf.toString("utf-8", 0, bytesRead);
    }

    // No data available at all
    if (entry.buffer.length === 0) {
      return -1;
    }

    const nlIdx = entry.buffer.indexOf("\n");
    if (nlIdx !== -1) {
      // Found a newline
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
}

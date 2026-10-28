/**
 * Node.js implementation of FileIOAdapter using synchronous fs operations.
 * Stream semantics (fids, positions, EOF) live in the core FileTable; this
 * adapter only supplies byte-level filesystem access.
 */

import {
  openSync,
  closeSync,
  readSync,
  writeSync,
  fstatSync,
  readFileSync,
  writeFileSync,
  statSync,
  accessSync,
  constants as fsConstants,
  mkdirSync,
  unlinkSync,
  readdirSync,
  rmSync,
  rmdirSync,
  renameSync,
  chmodSync,
  realpathSync,
} from "fs";
import { unzipSync, zipSync } from "fflate";
import { execFileSync } from "child_process";
import { homedir, tmpdir } from "os";
import { join, resolve, relative, dirname, basename } from "path";
import type {
  FileHandleIO,
  FileIOAdapter,
  WebOptions,
} from "./numbl-core/fileIOAdapter.js";
import { scanMFiles } from "./cli-scan.js";
import { globToRegExp } from "./numbl-core/helpers/glob.js";

function expandTilde(filepath: string): string {
  if (filepath.startsWith("~/")) {
    return join(homedir(), filepath.slice(2));
  }
  return filepath;
}

function isWritable(path: string): boolean {
  try {
    accessSync(path, fsConstants.W_OK);
    return true;
  } catch {
    return false;
  }
}

function copyDirRecursive(src: string, dst: string): void {
  mkdirSync(dst, { recursive: true });
  for (const entry of readdirSync(src, { withFileTypes: true })) {
    const s = join(src, entry.name);
    const d = join(dst, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(s, d);
    } else {
      writeFileSync(d, readFileSync(s));
    }
  }
}

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

/** Random access through file descriptors. */
const nodeHandles: FileHandleIO = {
  openHandle: (path, mode) => openSync(expandTilde(path), mode),
  readAt(fd, pos, len) {
    const buf = new Uint8Array(len);
    let got = 0;
    while (got < len) {
      const n = readSync(fd, buf, got, len - got, pos + got);
      if (n === 0) break;
      got += n;
    }
    return got < len ? buf.slice(0, got) : buf;
  },
  writeAt(fd, pos, bytes) {
    let done = 0;
    while (done < bytes.length) {
      done += writeSync(fd, bytes, done, bytes.length - done, pos + done);
    }
  },
  size: fd => fstatSync(fd).size,
  close: fd => closeSync(fd),
};

export class NodeFileIOAdapter implements FileIOAdapter {
  handles = nodeHandles;

  readFile(path: string): Uint8Array {
    const buf = readFileSync(expandTilde(path));
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  writeFile(path: string, data: Uint8Array): void {
    writeFileSync(expandTilde(path), data);
  }

  setFileMode(
    path: string,
    mode: { write?: boolean; execute?: boolean }
  ): boolean {
    try {
      const p = expandTilde(path);
      let bits = statSync(p).mode & 0o7777;
      if (mode.write !== undefined)
        bits = mode.write ? bits | 0o200 : bits & ~0o200;
      if (mode.execute !== undefined)
        bits = mode.execute ? bits | 0o100 : bits & ~0o100;
      chmodSync(p, bits);
      return true;
    } catch {
      return false;
    }
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

  userpath(): string {
    return join(homedir(), ".numbl");
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

  movefile(source: string, destination: string, force: boolean): boolean {
    try {
      const src = expandTilde(source);
      const srcStat = statSync(src);

      // Resolve destination: if it's an existing directory and source is a
      // file (or source is a folder being moved into an existing folder),
      // append the basename of source.
      let dst = expandTilde(destination);
      let dstType: "file" | "dir" | null = null;
      try {
        const s = statSync(dst);
        dstType = s.isDirectory() ? "dir" : "file";
      } catch {
        dstType = null;
      }
      if (dstType === "dir") {
        dst = join(dst, basename(src));
        try {
          dstType = statSync(dst).isDirectory() ? "dir" : "file";
        } catch {
          dstType = null;
        }
      }

      // A read-only destination file is only replaced with force (rename
      // itself would ignore the file's permissions).
      if (dstType === "file") {
        if (force) {
          try {
            chmodSync(dst, 0o666);
          } catch {
            // ignore
          }
        } else if (!isWritable(dst)) {
          return false;
        }
      }

      // If source is a folder, destination must be a folder (per MATLAB).
      if (srcStat.isDirectory() && dstType === "file") {
        return false;
      }

      // Ensure parent directory exists.
      const parent = dirname(dst);
      if (parent && parent !== "." && parent !== "/") {
        try {
          mkdirSync(parent, { recursive: true });
        } catch {
          // ignore
        }
      }

      try {
        renameSync(src, dst);
        return true;
      } catch {
        // Cross-device or other rename failure — fall back to copy+delete.
        if (srcStat.isDirectory()) {
          copyDirRecursive(src, dst);
          rmSync(src, { recursive: true });
        } else {
          writeFileSync(dst, readFileSync(src));
          unlinkSync(src);
        }
        return true;
      }
    } catch {
      return false;
    }
  }

  copyfile(source: string, destination: string, force: boolean): boolean {
    try {
      const src = expandTilde(source);
      const srcStat = statSync(src);

      // If destination is an existing directory, copy INTO it keeping the
      // source basename (MATLAB semantics); otherwise destination names the
      // target file/folder directly.
      let dst = expandTilde(destination);
      try {
        if (statSync(dst).isDirectory()) dst = join(dst, basename(src));
      } catch {
        // destination does not exist: it is the target name
      }

      if (srcStat.isDirectory()) {
        copyDirRecursive(src, dst);
        return true;
      }

      const parent = dirname(dst);
      if (parent && parent !== "." && parent !== "/") {
        mkdirSync(parent, { recursive: true });
      }
      // If destination exists and force is set, ensure it is writable.
      if (force) {
        try {
          chmodSync(dst, 0o666);
        } catch {
          // ignore (does not exist yet, or chmod unsupported)
        }
      }
      writeFileSync(dst, readFileSync(src));
      return true;
    } catch {
      return false;
    }
  }

  fileattrib(path: string): {
    Name: string;
    directory: boolean;
    UserRead: boolean;
    UserWrite: boolean;
    UserExecute: boolean;
  } | null {
    try {
      const p = expandTilde(path);
      const st = statSync(p); // throws if the path does not exist
      const mode = st.mode;
      return {
        Name: realpathSync(p),
        directory: st.isDirectory(),
        UserRead: (mode & 0o400) !== 0,
        UserWrite: (mode & 0o200) !== 0,
        UserExecute: (mode & 0o100) !== 0,
      };
    } catch {
      return null;
    }
  }

  deleteFile(pattern: string): void {
    const p = expandTilde(pattern);
    // Check for glob wildcards
    if (p.includes("*") || p.includes("?")) {
      const lastSep = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
      const dir = lastSep >= 0 ? p.slice(0, lastSep) : ".";
      const globPat = lastSep >= 0 ? p.slice(lastSep + 1) : p;
      const re = globToRegExp(globPat);
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

  zip(zipfilename: string, filenames: string[], rootDir: string): string[] {
    const root = resolve(expandTilde(rootDir));
    const dest = resolve(expandTilde(zipfilename));
    const entries: Record<string, Uint8Array> = {};
    const names: string[] = [];

    const toEntryName = (abs: string): string => {
      const rel = relative(root, abs);
      // Files outside rootDir are stored under their basename.
      if (rel.startsWith("..")) return basename(abs);
      return rel.split("\\").join("/");
    };
    const addFile = (abs: string) => {
      const name = toEntryName(abs);
      const buf = readFileSync(abs);
      entries[name] = new Uint8Array(
        buf.buffer,
        buf.byteOffset,
        buf.byteLength
      );
      names.push(name);
    };
    const addDir = (absDir: string) => {
      for (const entry of readdirSync(absDir)) {
        const abs = join(absDir, entry);
        let st;
        try {
          st = statSync(abs);
        } catch {
          continue; // skip unreadable entries (e.g. dangling symlinks)
        }
        if (st.isDirectory()) addDir(abs);
        else addFile(abs);
      }
    };
    const addPath = (abs: string) => {
      const st = statSync(abs); // throws if missing
      if (st.isDirectory()) addDir(abs);
      else addFile(abs);
    };

    for (const f of filenames) {
      const p = expandTilde(f);
      const abs = resolve(root, p);
      const last = basename(p);
      if (last.includes("*") || last.includes("?")) {
        // Simple glob in the final path component
        const dir = dirname(abs);
        const re = globToRegExp(last);
        let dirEntries: string[];
        try {
          dirEntries = readdirSync(dir);
        } catch {
          continue;
        }
        for (const entry of dirEntries) {
          if (re.test(entry)) addPath(join(dir, entry));
        }
      } else {
        addPath(abs);
      }
    }

    writeFileSync(dest, zipSync(entries));
    return names;
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
    const re = globToRegExp(globPat);
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
    // and the pattern from ** onward, which is matched against each entry's
    // path relative to the base (so `base/**/*.mex*` only returns entries
    // whose name matches `*.mex*`, at any depth including the base itself).
    const norm = pattern.replace(/\\/g, "/");
    const idx = norm.indexOf("**");
    let baseDir = norm.slice(0, idx);
    if (baseDir.endsWith("/")) {
      baseDir = baseDir.slice(0, -1);
    }
    if (!baseDir) baseDir = ".";
    const absBase = resolve(baseDir);
    try {
      if (!statSync(absBase).isDirectory()) return [];
    } catch {
      return []; // base does not exist — no matches
    }

    const suffix = norm.slice(idx);
    const re = globToRegExp(suffix, { paths: true });

    const results: {
      name: string;
      folder: string;
      bytes: number;
      isdir: boolean;
      mtimeMs: number;
    }[] = [];

    const pushIfMatch = (
      rel: string,
      name: string,
      folder: string,
      bytes: number,
      isdir: boolean,
      mtimeMs: number
    ) => {
      if (re.test(rel)) results.push({ name, folder, bytes, isdir, mtimeMs });
    };

    const walkDir = (dir: string, rel: string) => {
      // . and .. entries for this directory (returned when they match the
      // pattern, e.g. `**` or `**/*`, mirroring MATLAB's dir listings)
      const relPrefix = rel ? rel + "/" : "";
      let dirMtime = 0;
      try {
        dirMtime = statSync(dir).mtimeMs;
      } catch {
        // keep 0
      }
      pushIfMatch(relPrefix + ".", ".", dir, 0, true, dirMtime);
      let parentMtime = 0;
      try {
        parentMtime = statSync(resolve(dir, "..")).mtimeMs;
      } catch {
        // keep 0
      }
      pushIfMatch(relPrefix + "..", "..", dir, 0, true, parentMtime);

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
          pushIfMatch(
            relPrefix + entry,
            entry,
            dir,
            es.isDirectory() ? 0 : es.size,
            es.isDirectory(),
            es.mtimeMs
          );
          if (es.isDirectory()) subdirs.push(entry);
        } catch {
          // skip
        }
      }
      for (const sub of subdirs) {
        walkDir(join(dir, sub), relPrefix + sub);
      }
    };

    walkDir(absBase, "");
    return results;
  }
}

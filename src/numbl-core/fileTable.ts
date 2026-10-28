/**
 * MATLAB file-identifier semantics (fopen/fclose/fgetl/fread/fseek/...),
 * implemented once on top of a host's whole-file byte primitives.
 *
 * Every open file is held in memory. All fids on the same path share one
 * buffer, so a reader sees another fid's writes immediately, as in MATLAB.
 * Dirty buffers are written back to the host on fclose, before any other
 * adapter call (see `fileIOView`), and at the end of each execution.
 */

import type { FileIOAdapter } from "./fileIOAdapter.js";
import { RuntimeError } from "./runtime/error.js";
import {
  DEFAULT_TEXT_ENCODING,
  lookupTextEncoding,
  type TextEncoding,
} from "./helpers/textCodec.js";

/** The host operations the table needs. */
interface FilePrimitives {
  readFile(path: string): Uint8Array;
  writeFile(path: string, data: Uint8Array): void;
  existsPath(path: string): "file" | "dir" | null;
  resolvePath(path: string): string;
  isWritable(path: string): boolean;
}

/** In-memory contents of one open path, shared by all fids on it. */
interface SharedFile {
  path: string;
  data: Uint8Array;
  size: number;
  /** False while `data` may alias a buffer the host still holds. */
  owned: boolean;
  dirty: boolean;
  refs: number;
}

interface Stream {
  name: string;
  permission: string;
  canRead: boolean;
  canWrite: boolean;
  append: boolean;
  machinefmt: string;
  encoding: TextEncoding;
  pos: number;
  eof: boolean;
  lastError: string;
  file: SharedFile;
}

const INVALID_FID =
  "Invalid file identifier. Use fopen to generate a valid file identifier.";
const BAD_FD = "Bad file descriptor";
const NO_SUCH_FILE = "No such file or directory";

// fopen(fid) reports these names with literal double quotes.
const STD_STREAMS: Record<number, { name: string; permission: string }> = {
  0: { name: '"stdin"', permission: "r" },
  1: { name: '"stdout"', permission: "a" },
  2: { name: '"stderr"', permission: "a" },
};

const MACHINE_FORMATS: Record<string, string> = {
  n: "ieee-le.l64",
  native: "ieee-le.l64",
  a: "ieee-le.l64",
  "ieee-le.l64": "ieee-le.l64",
  l: "ieee-le",
  "ieee-le": "ieee-le",
  b: "ieee-be",
  "ieee-be": "ieee-be",
  s: "ieee-be.l64",
  "ieee-be.l64": "ieee-be.l64",
};

const ERRNO_MESSAGES: Record<string, string> = {
  ENOENT: NO_SUCH_FILE,
  EACCES: "Permission denied",
  EPERM: "Operation not permitted",
  EISDIR: "Is a directory",
  ENOTDIR: "Not a directory",
  EROFS: "Read-only file system",
  ENOSPC: "No space left on device",
};

function osMessage(e: unknown): string {
  const code = (e as { code?: unknown } | null)?.code;
  if (typeof code === "string" && ERRNO_MESSAGES[code])
    return ERRNO_MESSAGES[code];
  return e instanceof Error ? e.message : String(e);
}

function invalidFid(message = INVALID_FID): RuntimeError {
  const err = new RuntimeError(message);
  err.identifier = "MATLAB:FileIO:InvalidFid";
  return err;
}

function parentDir(path: string): string {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  if (i < 0) return "";
  return i === 0 ? path.slice(0, 1) : path.slice(0, i);
}

/** Parse an fopen permission. Returns the mode letter and whether '+' was
 *  given, plus the permission as fopen(fid) reports it ('t' dropped). */
function parsePermission(perm: string): {
  mode: "r" | "w" | "a";
  plus: boolean;
  reported: string;
} {
  if ((perm.match(/[rwa]/gi) ?? []).length > 1) {
    throw new RuntimeError(
      "The file mode for fopen must contain exactly one of the modes 'r', 'w' or 'a'."
    );
  }
  const m = /^([rwaWA])(\+?)([tb]?)(\+?)$/.exec(perm);
  if (!m || (m[2] && m[4])) {
    const err = new RuntimeError("Invalid permission.");
    err.identifier = "MATLAB:badpermission_mx";
    throw err;
  }
  return {
    mode: m[1].toLowerCase() as "r" | "w" | "a",
    plus: !!(m[2] || m[4]),
    reported: perm.replace("t", ""),
  };
}

function primitivesFor(io: FileIOAdapter): FilePrimitives {
  const readFile = io.readFile
    ? (p: string) => io.readFile!(p)
    : (p: string) => legacyReadFile(io, p);
  return {
    readFile,
    writeFile: io.writeFile
      ? (p, d) => io.writeFile!(p, d)
      : (p, d) => legacyWriteFile(io, p, d),
    existsPath: io.existsPath
      ? p => io.existsPath!(p)
      : p => {
          try {
            readFile(p);
            return "file";
          } catch {
            return null;
          }
        },
    resolvePath: io.resolvePath ? p => io.resolvePath!(p) : p => p,
    isWritable: p => io.fileattrib?.(p)?.UserWrite ?? true,
  };
}

function enoent(): Error {
  return Object.assign(new Error(NO_SUCH_FILE), { code: "ENOENT" });
}

/** readFile for hosts that only implement the legacy stream methods. */
function legacyReadFile(io: FileIOAdapter, path: string): Uint8Array {
  if (io.fopen && io.freadBytes && io.fclose) {
    const fid = io.fopen(path, "rb");
    if (fid < 0) throw enoent();
    try {
      const chunks: Uint8Array[] = [];
      let total = 0;
      for (;;) {
        const chunk = io.freadBytes(fid, 1 << 16);
        if (chunk.length === 0) break;
        chunks.push(chunk.slice());
        total += chunk.length;
      }
      const out = new Uint8Array(total);
      let off = 0;
      for (const c of chunks) {
        out.set(c, off);
        off += c.length;
      }
      return out;
    } finally {
      io.fclose(fid);
    }
  }
  if (io.fileread) return new TextEncoder().encode(io.fileread(path));
  throw new Error("File I/O is not available in this environment");
}

/** writeFile for hosts that only implement the legacy stream methods. */
function legacyWriteFile(
  io: FileIOAdapter,
  path: string,
  data: Uint8Array
): void {
  if (!io.fopen || !io.fclose)
    throw new Error("File I/O is not available in this environment");
  const fid = io.fopen(path, "wb");
  if (fid < 0) throw enoent();
  try {
    if (io.fwriteBytes) io.fwriteBytes(fid, data);
    else if (io.fwrite) io.fwrite(fid, new TextDecoder().decode(data));
  } finally {
    io.fclose(fid);
  }
}

export class FileTable {
  private readonly fs: FilePrimitives;
  private readonly streams = new Map<number, Stream>();
  private readonly shared = new Map<string, SharedFile>();

  constructor(io: FileIOAdapter) {
    this.fs = primitivesFor(io);
  }

  // ── Opening and closing ───────────────────────────────────────────

  /** fopen(filename, permission, machinefmt, encoding). Returns the fid,
   *  or -1 and an OS-style message. Throws for invalid arguments. */
  open(
    filename: string,
    permission = "r",
    machinefmt = "n",
    encoding?: string
  ): { fid: number; message: string } {
    const { mode, plus, reported } = parsePermission(permission);
    const fmt = MACHINE_FORMATS[machinefmt.toLowerCase()];
    if (!fmt) throw new RuntimeError("Invalid machine format.");
    let enc = DEFAULT_TEXT_ENCODING;
    if (encoding !== undefined && encoding !== "") {
      const found = lookupTextEncoding(encoding);
      if (!found)
        throw new RuntimeError(`fopen: unsupported encoding '${encoding}'`);
      enc = found;
    }

    const fail = (message: string) => ({ fid: -1, message });
    if (filename === "") return fail(NO_SUCH_FILE);
    const path = this.fs.resolvePath(filename);
    const kind = this.fs.existsPath(path);
    if (kind === "dir") return fail("Is a directory");
    if (kind === null && mode === "r") return fail(NO_SUCH_FILE);
    if (kind === null) {
      const parent = parentDir(path);
      if (parent && this.fs.existsPath(parent) !== "dir")
        return fail(NO_SUCH_FILE);
    }
    const writes = mode !== "r" || plus;
    if (kind === "file" && writes && !this.fs.isWritable(path)) {
      return fail(
        mode === "r"
          ? "Unable to open file. File must have read permission."
          : "Permission denied"
      );
    }

    let file = this.shared.get(path);
    try {
      if (!file) {
        const data =
          kind === "file" && mode !== "w"
            ? this.fs.readFile(path)
            : new Uint8Array(0);
        file = {
          path,
          data,
          size: data.length,
          owned: false,
          dirty: false,
          refs: 0,
        };
      }
      // 'w' truncates and 'a' creates the file at open time, as C does.
      if (mode === "w" || kind === null) {
        file.size = 0;
        this.fs.writeFile(path, new Uint8Array(0));
        file.dirty = false;
      }
    } catch (e) {
      return fail(osMessage(e));
    }
    this.shared.set(path, file);
    file.refs++;

    let fid = 3;
    while (this.streams.has(fid)) fid++;
    this.streams.set(fid, {
      name: path,
      permission: reported,
      canRead: mode === "r" || plus,
      canWrite: writes,
      append: mode === "a",
      machinefmt: fmt,
      encoding: enc,
      pos: mode === "a" && !plus ? file.size : 0,
      eof: false,
      lastError: "",
      file,
    });
    return { fid, message: "" };
  }

  /** fclose(fid). Returns 0; throws for a fid that is not open. */
  close(fid: number): number {
    const s = this.streams.get(fid);
    if (!s) {
      throw invalidFid(
        fid >= 0
          ? "Invalid file identifier.  Use fopen to generate a valid file identifier."
          : INVALID_FID
      );
    }
    this.streams.delete(fid);
    this.release(s.file);
    return 0;
  }

  /** fclose('all'). */
  closeAll(): number {
    for (const fid of [...this.streams.keys()]) this.close(fid);
    return 0;
  }

  /** Open fids in ascending order (fopen('all') / openedFiles). */
  openFids(): number[] {
    return [...this.streams.keys()].sort((a, b) => a - b);
  }

  /** [filename, permission, machinefmt, encoding] = fopen(fid); null when
   *  fid is not open. */
  info(fid: number): {
    name: string;
    permission: string;
    machinefmt: string;
    encoding: string;
  } | null {
    const std = STD_STREAMS[fid];
    if (std) return { ...std, machinefmt: "ieee-le.l64", encoding: "UTF-16" };
    const s = this.streams.get(fid);
    if (!s) return null;
    return {
      name: s.name,
      permission: s.permission,
      machinefmt: s.machinefmt,
      encoding: s.encoding.name,
    };
  }

  // ── Reading ───────────────────────────────────────────────────────

  /**
   * fgetl/fgets. A line ends at LF, CRLF or a lone CR. Returns -1 at end
   * of file. Like MATLAB, the EOF flag is set as soon as a line read
   * reaches the end of the file, and once set it sticks until a seek.
   */
  readLine(fid: number, keepNewline: boolean, maxChars?: number): string | -1 {
    const s = this.readable(fid);
    if (!s) return "";
    const { data, size } = s.file;
    if (s.eof || s.pos >= size) {
      s.eof = true;
      return -1;
    }
    let lineEnd = s.pos;
    while (lineEnd < size && data[lineEnd] !== 10 && data[lineEnd] !== 13)
      lineEnd++;
    let termEnd = lineEnd;
    if (lineEnd < size) {
      termEnd =
        data[lineEnd] === 13 && lineEnd + 1 < size && data[lineEnd + 1] === 10
          ? lineEnd + 2
          : lineEnd + 1;
    }
    let text: string;
    if (maxChars !== undefined) {
      const r = s.encoding.decode(data, s.pos, termEnd, maxChars);
      text = r.text;
      s.pos = r.end;
    } else {
      text = s.encoding.decode(
        data,
        s.pos,
        keepNewline ? termEnd : lineEnd
      ).text;
      s.pos = termEnd;
    }
    if (s.pos >= size) s.eof = true;
    return text;
  }

  /** Read up to `count` bytes (Infinity reads to the end). Asking for more
   *  bytes than remain sets the EOF flag. */
  readBytes(fid: number, count: number): Uint8Array {
    const s = this.readable(fid);
    if (!s || s.eof) return new Uint8Array(0);
    const avail = s.file.size - s.pos;
    const n = Math.max(0, Math.min(count, avail));
    if (count > avail) s.eof = true;
    const out = s.file.data.slice(s.pos, s.pos + n);
    s.pos += n;
    return out;
  }

  /** Read up to `count` characters decoded with the fid's encoding (fread
   *  with 'char' precision). */
  readChars(fid: number, count: number): string {
    const s = this.readable(fid);
    if (!s || s.eof) return "";
    const r = s.encoding.decode(s.file.data, s.pos, s.file.size, count);
    if (r.text.length < count) s.eof = true;
    s.pos = r.end;
    return r.text;
  }

  /** The rest of the file as text, without moving (for fscanf). */
  peekText(fid: number): string {
    const s = this.readable(fid);
    if (!s || s.eof) return "";
    return s.encoding.decode(s.file.data, s.pos, s.file.size).text;
  }

  /** Move past `nchars` characters of `peekText`; sets the EOF flag when
   *  that reaches the end of the file. */
  advanceChars(fid: number, nchars: number): void {
    const s = this.readable(fid);
    if (!s || s.eof) return;
    s.pos = s.encoding.decode(s.file.data, s.pos, s.file.size, nchars).end;
    if (s.pos >= s.file.size) s.eof = true;
  }

  // ── Writing ───────────────────────────────────────────────────────

  /** Write bytes at the current position ('a' modes: at the end). Returns
   *  the byte count, or 0 (setting ferror) if fid is not writable. */
  writeBytes(fid: number, bytes: Uint8Array): number {
    const s = this.stream(fid);
    if (!s.canWrite) {
      s.lastError = BAD_FD;
      return 0;
    }
    const f = s.file;
    if (s.append) s.pos = f.size;
    const end = s.pos + bytes.length;
    if (!f.owned || end > f.data.length) {
      const grown = new Uint8Array(
        Math.max(end, f.owned ? f.data.length * 2 : f.size, 256)
      );
      grown.set(f.data.subarray(0, f.size));
      f.data = grown;
      f.owned = true;
    }
    f.data.set(bytes, s.pos);
    s.pos = end;
    if (end > f.size) f.size = end;
    f.dirty = true;
    s.lastError = "";
    return bytes.length;
  }

  /** Advance `n` bytes before a write (fwrite's skip), zero-filling past
   *  the end of the file. */
  skipForWrite(fid: number, n: number): void {
    const s = this.stream(fid);
    if (!s.canWrite) return;
    const target = s.pos + n;
    if (target <= s.file.size) {
      s.pos = target;
    } else {
      s.pos = s.file.size;
      this.writeBytes(fid, new Uint8Array(target - s.file.size));
    }
  }

  /** Write text encoded with the fid's encoding (fprintf). */
  writeText(fid: number, text: string): number {
    return this.writeBytes(fid, this.encodeText(fid, text));
  }

  /** Encode text with the fid's encoding. */
  encodeText(fid: number, text: string): Uint8Array {
    return this.stream(fid).encoding.encode(text);
  }

  // ── Position and status ───────────────────────────────────────────

  /** fseek. origin: -1 (bof), 0 (cof), 1 (eof). Seeking before the start
   *  or past the end fails with -1 and leaves the position unchanged. */
  seek(fid: number, offset: number, origin: number): number {
    if (STD_STREAMS[fid]) return -1;
    const s = this.stream(fid);
    const base = origin === -1 ? 0 : origin === 0 ? s.pos : s.file.size;
    const target = base + offset;
    if (!Number.isFinite(target) || target < 0 || target > s.file.size)
      return -1;
    s.pos = Math.floor(target);
    s.eof = false;
    return 0;
  }

  tell(fid: number): number {
    if (STD_STREAMS[fid]) return -1;
    return this.stream(fid).pos;
  }

  atEof(fid: number): boolean {
    if (STD_STREAMS[fid]) return false;
    return this.stream(fid).eof;
  }

  /** [message, errnum] = ferror(fid[, 'clear']). */
  error(fid: number, clear = false): { message: string; errnum: number } {
    if (STD_STREAMS[fid]) return { message: "", errnum: 0 };
    const s = this.stream(fid);
    const message = s.lastError;
    if (clear) s.lastError = "";
    return { message, errnum: message === BAD_FD ? 9 : message ? -1 : 0 };
  }

  /** The machine format fid was opened with ('ieee-le', 'ieee-be', ...). */
  machineFormat(fid: number): string {
    return STD_STREAMS[fid] ? "ieee-le.l64" : this.stream(fid).machinefmt;
  }

  // ── Whole files and flushing ──────────────────────────────────────

  /** Write every modified open file back to the host. */
  flush(): void {
    for (const f of this.shared.values()) this.flushFile(f);
  }

  /** Read a whole file (fileread), seeing unflushed writes. */
  readFileBytes(path: string): Uint8Array {
    this.flush();
    return this.fs.readFile(path);
  }

  // ── Internals ─────────────────────────────────────────────────────

  private stream(fid: number): Stream {
    const s = this.streams.get(fid);
    if (!s) throw invalidFid();
    return s;
  }

  /** The stream for a read; null (with ferror set) if it is write-only. */
  private readable(fid: number): Stream | null {
    if (STD_STREAMS[fid]) {
      throw new RuntimeError(
        "Operation is not implemented for requested file identifier."
      );
    }
    const s = this.stream(fid);
    if (!s.canRead) {
      s.lastError = BAD_FD;
      return null;
    }
    return s;
  }

  private flushFile(f: SharedFile): void {
    if (!f.dirty) return;
    f.dirty = false;
    // The host may keep this buffer, so copy before the next write.
    f.owned = false;
    f.data = f.data.subarray(0, f.size);
    this.fs.writeFile(f.path, f.data);
  }

  private release(f: SharedFile): void {
    if (--f.refs > 0) return;
    this.shared.delete(f.path);
    this.flushFile(f);
  }
}

const tables = new WeakMap<FileIOAdapter, FileTable>();
const views = new WeakMap<FileIOAdapter, FileIOAdapter>();

/** The FileTable for an adapter. One table per adapter, so open fids live
 *  as long as the host keeps passing the same adapter (e.g. a REPL). */
export function fileTableFor(io: FileIOAdapter): FileTable {
  let t = tables.get(io);
  if (!t) {
    t = new FileTable(io);
    tables.set(io, t);
  }
  return t;
}

/** The adapter as builtins should see it: every call first flushes the
 *  adapter's FileTable, so dir/copyfile/fileread/... see what fids wrote. */
export function fileIOView(io: FileIOAdapter): FileIOAdapter {
  let v = views.get(io);
  if (!v) {
    const table = fileTableFor(io);
    v = new Proxy(io, {
      get(target, prop, receiver) {
        const value = Reflect.get(target, prop, receiver);
        if (typeof value !== "function") return value;
        return (...args: unknown[]) => {
          table.flush();
          return value.apply(target, args);
        };
      },
    });
    views.set(io, v);
  }
  return v;
}

/**
 * Byte storage behind one open path, shared by every fid on that path (see
 * FileTable). Two implementations:
 *
 * - HandleStore: random access through the host's `FileHandleIO` (Node).
 *   Memory stays bounded: a small read cache and a small write buffer.
 * - MemoryStore: the whole file in memory, written back with writeFile
 *   (hosts without random access, e.g. the browser VFS).
 */

import type { FileHandleIO } from "./fileIOAdapter.js";

export interface FileStore {
  readonly path: string;
  /** Number of fids sharing this store. */
  refs: number;
  size(): number;
  /** Up to `len` bytes at `pos` (fewer only at end of file). The caller
   *  owns the returned array. */
  read(pos: number, len: number): Uint8Array;
  /** Some bytes at `pos` (at least one unless at end of file, at most
   *  `len`). Only valid until the next store operation. */
  view(pos: number, len: number): Uint8Array;
  write(pos: number, bytes: Uint8Array): void;
  /** Drop all contents after the host file was truncated or recreated
   *  (fopen 'w' on an open path). Pending writes are discarded. */
  reset(): void;
  /** Push buffered writes to the host and forget cached reads, so the
   *  next access sees what other host calls did to the file. */
  sync(): void;
  /** Ensure writes are possible (a store opened for reading only). */
  makeWritable(): void;
  /** sync() and release host resources. */
  close(): void;
}

// ── Whole file in memory ────────────────────────────────────────────

export class MemoryStore implements FileStore {
  refs = 0;
  private data: Uint8Array;
  private length: number;
  /** False while `data` may alias a buffer the host still holds. */
  private owned = false;
  private dirty = false;

  constructor(
    readonly path: string,
    initial: Uint8Array,
    private readonly writeFile: (path: string, data: Uint8Array) => void
  ) {
    this.data = initial;
    this.length = initial.length;
  }

  size(): number {
    return this.length;
  }

  read(pos: number, len: number): Uint8Array {
    return this.data.slice(pos, Math.min(this.length, pos + len));
  }

  view(pos: number, len: number): Uint8Array {
    return this.data.subarray(pos, Math.min(this.length, pos + len));
  }

  write(pos: number, bytes: Uint8Array): void {
    const end = pos + bytes.length;
    if (!this.owned || end > this.data.length) {
      const grown = new Uint8Array(
        Math.max(end, this.owned ? this.data.length * 2 : this.length, 256)
      );
      grown.set(this.data.subarray(0, this.length));
      this.data = grown;
      this.owned = true;
    }
    this.data.set(bytes, pos);
    if (end > this.length) this.length = end;
    this.dirty = true;
  }

  reset(): void {
    this.length = 0;
    this.dirty = false;
  }

  sync(): void {
    if (!this.dirty) return;
    this.dirty = false;
    // The host may keep this buffer, so copy before the next write.
    this.owned = false;
    this.data = this.data.subarray(0, this.length);
    this.writeFile(this.path, this.data);
  }

  makeWritable(): void {}

  close(): void {
    this.sync();
  }
}

// ── Random access through a host handle ─────────────────────────────

/** Read cache block size, and the largest write that is buffered. */
const BLOCK = 64 * 1024;

export class HandleStore implements FileStore {
  refs = 0;
  private handle: number;
  private writable: boolean;
  /** Host file size, or -1 if it must be asked again. */
  private hostSize = -1;
  private cache: Uint8Array | null = null;
  private cacheStart = 0;
  private readonly pending = new Uint8Array(BLOCK);
  private pendingStart = 0;
  private pendingLen = 0;

  constructor(
    readonly path: string,
    private readonly io: FileHandleIO,
    writable: boolean
  ) {
    this.writable = writable;
    this.handle = io.openHandle(path, writable ? "r+" : "r");
  }

  size(): number {
    if (this.hostSize < 0) this.hostSize = this.io.size(this.handle);
    return Math.max(this.hostSize, this.pendingStart + this.pendingLen);
  }

  read(pos: number, len: number): Uint8Array {
    this.flushPending();
    len = Math.max(0, Math.min(len, this.size() - pos));
    if (len >= BLOCK) return this.io.readAt(this.handle, pos, len);
    const out = new Uint8Array(len);
    let got = 0;
    while (got < len) {
      const v = this.view(pos + got, len - got);
      if (v.length === 0) break;
      out.set(v, got);
      got += v.length;
    }
    return got < len ? out.slice(0, got) : out;
  }

  view(pos: number, len: number): Uint8Array {
    this.flushPending();
    const c = this.cache;
    if (!c || pos < this.cacheStart || pos >= this.cacheStart + c.length) {
      if (pos >= this.size()) return new Uint8Array(0);
      this.cache = this.io.readAt(this.handle, pos, BLOCK);
      this.cacheStart = pos;
    }
    const off = pos - this.cacheStart;
    return this.cache!.subarray(off, Math.min(this.cache!.length, off + len));
  }

  write(pos: number, bytes: Uint8Array): void {
    const end = pos + bytes.length;
    const c = this.cache;
    if (c && pos < this.cacheStart + c.length && end > this.cacheStart)
      this.cache = null;
    if (
      this.pendingLen > 0 &&
      pos === this.pendingStart + this.pendingLen &&
      this.pendingLen + bytes.length <= BLOCK
    ) {
      this.pending.set(bytes, this.pendingLen);
      this.pendingLen += bytes.length;
      return;
    }
    this.flushPending();
    if (bytes.length >= BLOCK) {
      this.writeThrough(pos, bytes);
    } else {
      this.pending.set(bytes);
      this.pendingStart = pos;
      this.pendingLen = bytes.length;
    }
  }

  reset(): void {
    this.pendingLen = 0;
    this.cache = null;
    this.hostSize = -1;
    // The host may have replaced the file, so reopen it.
    this.io.close(this.handle);
    this.handle = this.io.openHandle(this.path, this.writable ? "r+" : "r");
  }

  sync(): void {
    const wrote = this.pendingLen > 0;
    this.flushPending();
    if (wrote) this.io.flush?.(this.handle);
    this.cache = null;
    this.hostSize = -1;
  }

  makeWritable(): void {
    if (this.writable) return;
    this.io.close(this.handle);
    this.handle = this.io.openHandle(this.path, "r+");
    this.writable = true;
    this.cache = null;
  }

  close(): void {
    this.sync();
    this.io.close(this.handle);
  }

  private flushPending(): void {
    if (this.pendingLen === 0) return;
    const len = this.pendingLen;
    this.pendingLen = 0;
    this.writeThrough(this.pendingStart, this.pending.subarray(0, len));
  }

  private writeThrough(pos: number, bytes: Uint8Array): void {
    this.io.writeAt(this.handle, pos, bytes);
    if (this.hostSize >= 0)
      this.hostSize = Math.max(this.hostSize, pos + bytes.length);
  }
}

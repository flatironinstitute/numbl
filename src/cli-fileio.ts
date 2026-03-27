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
  statSync,
  mkdirSync,
} from "fs";
import { homedir } from "os";
import { join, resolve } from "path";
import type { FileIOAdapter } from "./numbl-core/fileIOAdapter.js";
import { scanMFiles } from "./cli.js";

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

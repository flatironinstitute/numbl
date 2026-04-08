/**
 * Browser implementation of SystemAdapter using in-memory state.
 *
 * The current working directory is delegated to the VFS so that
 * `cd` (system cwd) and relative-path resolution inside the VFS stay
 * in sync. Without this, project files written at "folder1/foo.m"
 * would land under VFS.cwd ("/project") while `cd folder1` would
 * resolve against the system adapter's own root, producing "/folder1"
 * — a directory that contains nothing.
 */

import type { SystemAdapter } from "../numbl-core/systemAdapter.js";
import type { VirtualFileSystem } from "./VirtualFileSystem.js";

export class BrowserSystemAdapter implements SystemAdapter {
  private vars = new Map<string, string>();
  private vfs: VirtualFileSystem | null = null;
  // Used only when no VFS is attached (e.g. tests that don't need files).
  private fallbackCwd = "/";

  constructor(vfs?: VirtualFileSystem) {
    if (vfs) this.vfs = vfs;
  }

  /** Attach (or replace) the VFS used as the cwd source of truth. */
  setVfs(vfs: VirtualFileSystem): void {
    this.vfs = vfs;
  }

  getEnv(name: string): string | undefined {
    return this.vars.get(name);
  }

  getAllEnv(): Record<string, string> {
    const result: Record<string, string> = {};
    for (const [k, v] of this.vars) {
      result[k] = v;
    }
    return result;
  }

  setEnv(name: string, value: string): void {
    this.vars.set(name, value);
  }

  cwd(): string {
    return this.vfs ? this.vfs.getCwd() : this.fallbackCwd;
  }

  chdir(dir: string): void {
    if (this.vfs) {
      this.vfs.setCwd(dir);
    } else {
      // Fallback used only when no VFS is attached.
      this.fallbackCwd = dir.startsWith("/")
        ? dir
        : this.fallbackCwd.replace(/\/$/, "") + "/" + dir;
    }
  }

  platform(): string {
    return "linux";
  }

  arch(): string {
    return "x64";
  }
}

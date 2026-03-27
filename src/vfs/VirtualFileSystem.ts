/**
 * In-memory virtual file system for browser execution.
 * Pure TypeScript — no browser or Node.js APIs.
 * All file content stored as Uint8Array internally.
 */

export interface VFSFile {
  content: Uint8Array;
  mtimeMs: number;
}

export interface VfsChanges {
  created: { path: string; content: Uint8Array }[];
  modified: { path: string; content: Uint8Array }[];
  deleted: string[];
}

export class VirtualFileSystem {
  private files = new Map<string, VFSFile>();
  private directories = new Set<string>();
  private cwd = "/project";

  // Change tracking
  private createdFiles = new Set<string>();
  private modifiedFiles = new Set<string>();
  private deletedFiles = new Set<string>();

  /** Clear change tracking. Call after populating the VFS with initial files. */
  clearChangeTracking(): void {
    this.createdFiles.clear();
    this.modifiedFiles.clear();
    this.deletedFiles.clear();
  }

  /** Normalize a path to absolute form. */
  normalizePath(p: string): string {
    // Make absolute
    if (!p.startsWith("/")) {
      p = this.cwd + "/" + p;
    }
    // Resolve . and ..
    const parts = p.split("/");
    const resolved: string[] = [];
    for (const part of parts) {
      if (part === "" || part === ".") continue;
      if (part === "..") {
        if (resolved.length > 0) resolved.pop();
      } else {
        resolved.push(part);
      }
    }
    return "/" + resolved.join("/");
  }

  getCwd(): string {
    return this.cwd;
  }

  setCwd(dir: string): void {
    this.cwd = this.normalizePath(dir);
  }

  readFile(path: string): Uint8Array {
    const norm = this.normalizePath(path);
    const file = this.files.get(norm);
    if (!file) throw new Error(`File not found: ${path}`);
    return file.content;
  }

  writeFile(path: string, content: Uint8Array): void {
    const norm = this.normalizePath(path);
    const existed = this.files.has(norm);
    this.files.set(norm, { content, mtimeMs: Date.now() });

    // Ensure parent directories exist implicitly
    this.ensureParentDirs(norm);

    // Track changes
    if (this.deletedFiles.has(norm)) {
      // Was deleted earlier, now re-created
      this.deletedFiles.delete(norm);
      this.createdFiles.add(norm);
    } else if (!existed) {
      this.createdFiles.add(norm);
    } else if (!this.createdFiles.has(norm)) {
      this.modifiedFiles.add(norm);
    }
    // If it was already in createdFiles, no change needed
  }

  deleteFile(path: string): boolean {
    const norm = this.normalizePath(path);
    if (!this.files.has(norm)) return false;
    this.files.delete(norm);

    if (this.createdFiles.has(norm)) {
      // Created then deleted in same session — no net change
      this.createdFiles.delete(norm);
      this.modifiedFiles.delete(norm);
    } else {
      this.modifiedFiles.delete(norm);
      this.deletedFiles.add(norm);
    }
    return true;
  }

  exists(path: string): "file" | "dir" | null {
    const norm = this.normalizePath(path);
    if (this.files.has(norm)) return "file";
    if (this.directories.has(norm)) return "dir";
    // Check if any file has this as a prefix (implicit directory)
    const prefix = norm + "/";
    for (const key of this.files.keys()) {
      if (key.startsWith(prefix)) return "dir";
    }
    return null;
  }

  fileSize(path: string): number {
    const norm = this.normalizePath(path);
    const file = this.files.get(norm);
    return file ? file.content.length : 0;
  }

  mkdir(dirPath: string): boolean {
    const norm = this.normalizePath(dirPath);
    this.directories.add(norm);
    // Ensure parents too
    this.ensureParentDirs(norm + "/placeholder");
    return true;
  }

  rmdir(dirPath: string, recursive: boolean): boolean {
    const norm = this.normalizePath(dirPath);
    if (recursive) {
      const prefix = norm + "/";
      for (const key of [...this.files.keys()]) {
        if (key.startsWith(prefix)) {
          this.deleteFile(key);
        }
      }
      for (const dir of [...this.directories]) {
        if (dir === norm || dir.startsWith(prefix)) {
          this.directories.delete(dir);
        }
      }
    } else {
      // Non-recursive: only remove if empty
      const prefix = norm + "/";
      for (const key of this.files.keys()) {
        if (key.startsWith(prefix)) return false; // not empty
      }
      this.directories.delete(norm);
    }
    return true;
  }

  /** List entries in a directory. */
  listDir(dirPath: string): {
    name: string;
    folder: string;
    bytes: number;
    isdir: boolean;
    mtimeMs: number;
  }[] {
    const norm = this.normalizePath(dirPath);
    const prefix = norm === "/" ? "/" : norm + "/";
    const results: {
      name: string;
      folder: string;
      bytes: number;
      isdir: boolean;
      mtimeMs: number;
    }[] = [];

    // Add . and ..
    const now = Date.now();
    results.push({
      name: ".",
      folder: norm,
      bytes: 0,
      isdir: true,
      mtimeMs: now,
    });
    results.push({
      name: "..",
      folder: norm,
      bytes: 0,
      isdir: true,
      mtimeMs: now,
    });

    const seen = new Set<string>();

    for (const [path, file] of this.files) {
      if (!path.startsWith(prefix)) continue;
      const rest = path.slice(prefix.length);
      const slashIdx = rest.indexOf("/");
      if (slashIdx === -1) {
        // Direct child file
        results.push({
          name: rest,
          folder: norm,
          bytes: file.content.length,
          isdir: false,
          mtimeMs: file.mtimeMs,
        });
      } else {
        // Subdirectory (implicit from file path)
        const dirName = rest.slice(0, slashIdx);
        if (!seen.has(dirName)) {
          seen.add(dirName);
          results.push({
            name: dirName,
            folder: norm,
            bytes: 0,
            isdir: true,
            mtimeMs: now,
          });
        }
      }
    }

    // Also include explicitly created empty directories
    for (const dir of this.directories) {
      if (!dir.startsWith(prefix)) continue;
      const rest = dir.slice(prefix.length);
      if (!rest.includes("/") && rest.length > 0 && !seen.has(rest)) {
        seen.add(rest);
        results.push({
          name: rest,
          folder: norm,
          bytes: 0,
          isdir: true,
          mtimeMs: now,
        });
      }
    }

    return results;
  }

  /** List all file paths in the VFS. */
  allFiles(): string[] {
    return [...this.files.keys()];
  }

  /** Get the changes since the VFS was created. */
  getChanges(): VfsChanges {
    const created: { path: string; content: Uint8Array }[] = [];
    const modified: { path: string; content: Uint8Array }[] = [];

    for (const path of this.createdFiles) {
      const file = this.files.get(path);
      if (file) created.push({ path, content: file.content });
    }
    for (const path of this.modifiedFiles) {
      const file = this.files.get(path);
      if (file) modified.push({ path, content: file.content });
    }

    return {
      created,
      modified,
      deleted: [...this.deletedFiles],
    };
  }

  private ensureParentDirs(path: string): void {
    const parts = path.split("/").filter(Boolean);
    for (let i = 1; i < parts.length; i++) {
      const dir = "/" + parts.slice(0, i).join("/");
      this.directories.add(dir);
    }
  }
}

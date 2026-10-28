/**
 * The files visible to a run: the caller's workspace files, the implicit cwd,
 * the search path and the bundled stdlib. Keeps them in sync with addpath,
 * rmpath and cd, and registers them with its `CodeIndex` for function
 * resolution.
 */

import { CodeIndex } from "./codeIndex.js";
import type { FileIOAdapter } from "../fileIOAdapter.js";
import type { SystemAdapter } from "../systemAdapter.js";
import type { WorkspaceFile, NativeBridge } from "./types.js";
import { parseMFile } from "../parser/index.js";
import { SyntaxError } from "../parser/errors.js";
import {
  isNumblJsFile,
  loadJsUserFunctions,
  type LoadedJsUserFunction,
} from "../jsUserFunctions.js";

/** Search-path entry for bundled shim files; always last. */
export const SHIM_SEARCH_PATH = "__numbl_shims__";

const withSep = (p: string) => (p.endsWith("/") ? p : p + "/");
const isUnder = (name: string, dir: string) =>
  name === dir || name.startsWith(withSep(dir));

export class Workspace {
  /** Functions and classes defined by the files, by name. */
  readonly code: CodeIndex;
  readonly mFiles: WorkspaceFile[] = [];
  readonly jsFiles: WorkspaceFile[] = [];
  readonly wasmFiles: WorkspaceFile[] = [];
  /** Loaded .numbl.js functions, sorted by search-path priority on rebuild. */
  jsFunctions: LoadedJsUserFunction[] = [];
  /** Names of the bundled stdlib and shim files. */
  readonly bundledFileNames = new Set<string>();
  /** The current directory, searched first unless it is an explicit path. */
  implicitCwd: string | null = null;
  /** True once addpath, rmpath or cd changed the workspace. */
  modified = false;
  /** Notified when a .m file enters or leaves the workspace after startup. */
  onFileAdded?: (f: WorkspaceFile) => void;
  onFileRemoved?: (name: string) => void;

  constructor(
    mainFileName: string,
    mainSource: string,
    private readonly fileIO: FileIOAdapter | undefined,
    private readonly nativeBridge: NativeBridge | undefined
  ) {
    this.code = new CodeIndex(mainFileName, mainSource);
  }

  get searchPaths(): string[] {
    return this.code.searchPaths;
  }

  private resolve(dir: string): string {
    return this.fileIO?.resolvePath?.(dir) ?? dir;
  }

  /** Parse a .m file into the shared AST cache. A file that fails to parse
   *  is skipped with a warning rather than aborting the run. */
  private parse(f: WorkspaceFile): boolean {
    try {
      this.code.fileASTCache.set(f.name, parseMFile(f.source, f.name));
      return true;
    } catch (e) {
      if (e instanceof SyntaxError) {
        console.warn(
          `Warning: skipping ${f.name} (syntax error at line ${e.line ?? "?"})`
        );
      } else {
        console.warn(`Warning: skipping ${f.name} (parse error)`);
      }
      return false;
    }
  }

  /** Initial file set: the caller's files, the implicit cwd, the explicit
   *  search paths (files already provided are not re-added) and the bundled
   *  stdlib and shims. All .numbl.js files load as one batch so library
   *  files (`_name.numbl.js`) are visible across directories. */
  load(
    files: WorkspaceFile[],
    searchPaths: string[],
    implicitCwd: string | null | undefined,
    system: SystemAdapter | undefined,
    bundled: WorkspaceFile[]
  ): void {
    const add = (f: WorkspaceFile) => {
      if (f.name.endsWith(".m")) {
        if (!this.mFiles.some(e => e.name === f.name)) this.mFiles.push(f);
      } else if (isNumblJsFile(f.name)) {
        if (!this.jsFiles.some(e => e.name === f.name)) this.jsFiles.push(f);
      } else if (f.name.endsWith(".wasm")) {
        if (!this.wasmFiles.some(e => e.name === f.name))
          this.wasmFiles.push(f);
      }
    };
    for (const f of files) add(f);

    const scan = this.fileIO?.scanDirectory?.bind(this.fileIO);
    if (implicitCwd !== null && system && scan) {
      try {
        const cwd = this.resolve(implicitCwd ?? system.cwd());
        if (!searchPaths.includes(cwd)) {
          this.implicitCwd = cwd;
          // The caller may already have sent this directory's files (the
          // REPL persists them between commands).
          if (!this.mFiles.some(f => isUnder(f.name, cwd))) {
            for (const f of scan(cwd)) add(f);
          }
        }
      } catch {
        // cwd inaccessible: run without an implicit cwd
        this.implicitCwd = null;
      }
    }
    if (scan) {
      for (const dir of searchPaths) {
        try {
          for (const f of scan(this.resolve(dir))) add(f);
        } catch {
          // a missing directory is tolerated, as for addpath and cd
        }
      }
    }

    this.jsFunctions = loadJsUserFunctions(
      this.jsFiles,
      this.wasmFiles,
      this.nativeBridge
    );

    for (const f of bundled) {
      this.mFiles.push(f);
      this.bundledFileNames.add(f.name);
    }
    this.searchPaths.splice(0, this.searchPaths.length, ...searchPaths);
    this.searchPaths.push(SHIM_SEARCH_PATH);
    if (this.implicitCwd !== null) this.searchPaths.unshift(this.implicitCwd);

    for (let i = this.mFiles.length - 1; i >= 0; i--) {
      if (!this.parse(this.mFiles[i])) this.mFiles.splice(i, 1);
    }
  }

  /** Register the current files and .numbl.js functions with the code
   *  index. */
  register(): void {
    if (this.mFiles.length > 0) this.code.registerWorkspaceFiles(this.mFiles);
    for (const e of this.jsFunctions) {
      this.code.registerJsUserFunction(e.name, e.fileName, e.builtin, e.help);
    }
  }

  /** Re-sort the files by search-path priority, re-register them and
   *  rebuild the function index. */
  rebuild(): void {
    const paths = this.searchPaths;
    // A file belongs to its LONGEST matching search path: with /a as the cwd
    // and /a/b on the path, /a/b/c.m belongs to /a/b.
    const priorityOf = (name: string): number => {
      let bestIdx = paths.length;
      let bestLen = -1;
      for (let i = 0; i < paths.length; i++) {
        if (isUnder(name, paths[i]) && paths[i].length > bestLen) {
          bestLen = paths[i].length;
          bestIdx = i;
        }
      }
      return bestIdx;
    };
    this.mFiles.sort((a, b) => priorityOf(a.name) - priorityOf(b.name));
    this.jsFunctions.sort(
      (a, b) => priorityOf(a.fileName) - priorityOf(b.fileName)
    );
    this.code.clearWorkspaceRegistrations();
    this.register();
    this.modified = true;
    this.code.buildFunctionIndex();
  }

  /** Scan a directory: parse its .m files not already known, and (re)load
   *  its .numbl.js functions. Files are always reloaded because files sent
   *  earlier (e.g. by the IDE) may have lacked their wasm/native binaries. */
  private scanDir(dir: string, tolerateMissing = false): void {
    if (!this.fileIO?.scanDirectory) return;
    let files: WorkspaceFile[];
    try {
      files = this.fileIO.scanDirectory(dir);
    } catch (e) {
      if (tolerateMissing) return;
      throw e;
    }
    const newJs: WorkspaceFile[] = [];
    const newWasm: WorkspaceFile[] = [];
    for (const f of files) {
      if (f.name.endsWith(".m")) {
        if (this.code.fileASTCache.has(f.name) || !this.parse(f)) continue;
        this.mFiles.push(f);
        this.onFileAdded?.(f);
      } else if (isNumblJsFile(f.name)) {
        newJs.push(f);
        if (!this.jsFiles.some(e => e.name === f.name)) this.jsFiles.push(f);
      } else if (f.name.endsWith(".wasm")) {
        newWasm.push(f);
        if (!this.wasmFiles.some(e => e.name === f.name))
          this.wasmFiles.push(f);
      }
    }
    if (newJs.length === 0) return;
    for (const entry of loadJsUserFunctions(
      newJs,
      newWasm,
      this.nativeBridge
    )) {
      const i = this.jsFunctions.findIndex(e => e.fileName === entry.fileName);
      if (i >= 0) this.jsFunctions[i] = entry;
      else this.jsFunctions.push(entry);
    }
  }

  /** Drop the files under `dir`, except those for which `keep` is true. */
  private removeFilesUnder(dir: string, keep: (name: string) => boolean): void {
    const drop = (name: string) => isUnder(name, dir) && !keep(name);
    for (let i = this.mFiles.length - 1; i >= 0; i--) {
      const name = this.mFiles[i].name;
      if (!drop(name)) continue;
      this.code.fileASTCache.delete(name);
      this.onFileRemoved?.(name);
      this.mFiles.splice(i, 1);
    }
    this.jsFunctions = this.jsFunctions.filter(e => !drop(e.fileName));
    for (const list of [this.jsFiles, this.wasmFiles]) {
      for (let i = list.length - 1; i >= 0; i--) {
        if (drop(list[i].name)) list.splice(i, 1);
      }
    }
  }

  /** addpath. Returns false if `dir` is already on the path. */
  addPath(dir: string, position: "begin" | "end"): boolean {
    const absDir = this.resolve(dir);
    const paths = this.searchPaths;
    if (paths.includes(absDir)) return false;
    if (position === "begin") {
      // The implicit cwd stays first.
      const cwdFirst =
        this.implicitCwd !== null && paths[0] === this.implicitCwd;
      paths.splice(cwdFirst ? 1 : 0, 0, absDir);
    } else {
      const shimIdx = paths.indexOf(SHIM_SEARCH_PATH);
      if (shimIdx >= 0) paths.splice(shimIdx, 0, absDir);
      else paths.push(absDir);
    }
    this.scanDir(absDir);
    return true;
  }

  /** rmpath. */
  removePath(dir: string): void {
    const absDir = this.resolve(dir);
    const idx = this.searchPaths.indexOf(absDir);
    if (idx >= 0) this.searchPaths.splice(idx, 1);
    this.removeFilesUnder(absDir, () => false);
  }

  /** cd: make `newCwd` the implicit cwd. Returns false if it already is. */
  changeCwd(newCwd: string): boolean {
    const absCwd = this.resolve(newCwd);
    const old = this.implicitCwd;
    if (old === absCwd) return false;

    if (old !== null) {
      // Files in explicit search paths below the old cwd belong to those
      // paths and stay (addpath('/a/b/lib'); cd('/a/b'); cd elsewhere keeps
      // /a/b/lib visible).
      const deeper = this.searchPaths.filter(
        p => p !== old && p !== SHIM_SEARCH_PATH && isUnder(p, old)
      );
      this.removeFilesUnder(old, name => deeper.some(p => isUnder(name, p)));
      const oldIdx = this.searchPaths.indexOf(old);
      if (oldIdx >= 0) this.searchPaths.splice(oldIdx, 1);
      this.implicitCwd = null;
    }

    // An explicit search path needs no implicit entry or scan.
    if (this.searchPaths.includes(absCwd)) return true;

    this.searchPaths.unshift(absCwd);
    this.implicitCwd = absCwd;
    // An inaccessible cwd keeps its entry but contributes no files.
    this.scanDir(absCwd, true);
    return true;
  }

  /** Search paths to report back to callers (no implicit cwd or shims). */
  explicitSearchPaths(): string[] {
    return this.searchPaths.filter(
      p => p !== SHIM_SEARCH_PATH && p !== this.implicitCwd
    );
  }

  /** Files to report back to callers (no bundled stdlib or shims). */
  userFiles(): WorkspaceFile[] {
    return [
      ...this.mFiles.filter(f => !this.bundledFileNames.has(f.name)),
      ...this.jsFiles,
      ...this.wasmFiles,
    ];
  }
}

/**
 * Code index: every function and class a run can call, by name.
 *
 * Holds the parsed ASTs of all known files, maps workspace files to function
 * names (top-level, `+pkg` namespaces, `@Class` folders, `private/`
 * directories), the main file's local functions and classes, the JS user
 * functions, and the `ClassTable`. From these it builds the `FunctionIndex`
 * that `resolveFunction` consults. The `Workspace` keeps it in sync with the
 * visible file set.
 */

import type { AbstractSyntaxTree, Stmt } from "../parser/index.js";
import type { IBuiltin } from "../interpreter/builtins/index.js";
import type { WorkspaceFile } from "./types.js";
import { extractClassInfo, makeOldStyleClassInfo } from "./classInfo.js";
import { ClassTable } from "./classTable.js";
import { buildFunctionIndex, type FunctionIndex } from "./functionIndex.js";
import {
  callerPrivateScope,
  privateDirScope,
  relativeToSearchPath,
} from "./searchPaths.js";

type FunctionStmt = Stmt & { type: "Function" };

/** A workspace file defining a function. */
export interface FunctionFile {
  fileName: string;
  source: string;
}

/** True if `source` starts with `classdef` after leading whitespace and
 *  `%` comment lines. */
function startsWithClassdef(source: string): boolean {
  let t = source.trimStart();
  while (t.startsWith("%")) {
    const nl = t.indexOf("\n");
    if (nl < 0) return false;
    t = t.slice(nl + 1).trimStart();
  }
  return t.startsWith("classdef");
}

export class CodeIndex {
  /** Parsed ASTs, keyed by file name. Every registered file is parsed
   *  before it is registered. */
  readonly fileASTCache = new Map<string, AbstractSyntaxTree>();
  /** Ordered absolute search paths, mutated in place by the `Workspace`.
   *  Empty means file names are used as given. */
  readonly searchPaths: string[] = [];
  /** Workspace function files: function name → file. */
  readonly filesByFuncName = new Map<string, FunctionFile>();
  /** JS user functions (.numbl.js): function name → file and builtin. */
  readonly jsUserFunctionsByName = new Map<
    string,
    { fileName: string; builtin: IBuiltin; help?: string }
  >();
  /** Private function files: scope directory → function name → file. */
  readonly privateFilesByDir = new Map<string, Map<string, FunctionFile>>();
  /** Absolute file name → resolved function name (e.g. "pkg.func"). */
  readonly fileToFuncName = new Map<string, string>();
  /** Local functions of the main file, in definition order. */
  readonly mainLocalFunctions = new Map<string, FunctionStmt>();
  readonly classes = new ClassTable();

  private index: FunctionIndex | null = null;

  constructor(
    readonly mainFileName: string,
    private readonly mainSource: string
  ) {}

  /** Path of a file relative to its search path. */
  getRelativePath(absolutePath: string): string {
    return relativeToSearchPath(absolutePath, this.searchPaths);
  }

  /** Look up a pre-parsed AST by file name. Throws if not cached. */
  getCachedAST(fileName: string): AbstractSyntaxTree {
    const cached = this.fileASTCache.get(fileName);
    if (!cached) {
      throw new Error(
        `FileASTCache miss: no cached AST for "${fileName}". All files must be pre-parsed.`
      );
    }
    return cached;
  }

  // ── Main file ─────────────────────────────────────────────────────

  /** Cache the main file's AST and register its local functions and
   *  classes. */
  setMainFile(ast: AbstractSyntaxTree): void {
    this.fileASTCache.set(this.mainFileName, ast);
    for (const stmt of ast.body) {
      if (stmt.type === "Function") {
        this.mainLocalFunctions.set(stmt.name, stmt);
      } else if (stmt.type === "ClassDef") {
        this.classes.addLocalClass(
          extractClassInfo(stmt, stmt.name, this.mainFileName, this.mainSource)
        );
      }
    }
  }

  // ── Workspace files ───────────────────────────────────────────────

  /**
   * Register parsed `.m` workspace files as functions (top-level and +pkg
   * namespaces), classes (classdef files and @ClassName folders) and private
   * functions. Earlier files win over later ones with the same name.
   */
  registerWorkspaceFiles(files: WorkspaceFile[]): void {
    // First pass: collect @folder groups
    const classFolderGroups = new Map<
      string,
      { classDefFile?: WorkspaceFile; methodFiles: WorkspaceFile[] }
    >();

    for (const file of files) {
      const parts = this.getRelativePath(file.name).split("/");

      // Files under private/ (including @ClassName/private/) are private
      // helper functions scoped to the directory containing private/.
      const privateScope = privateDirScope(parts);
      if (privateScope !== null) {
        const baseName = parts[parts.length - 1].replace(/\.m$/, "");
        let dirMap = this.privateFilesByDir.get(privateScope);
        if (!dirMap) {
          dirMap = new Map();
          this.privateFilesByDir.set(privateScope, dirMap);
        }
        dirMap.set(baseName, { fileName: file.name, source: file.source });
        continue;
      }

      // Check for @ClassName folder
      const atIdx = parts.findIndex(p => p.startsWith("@"));
      if (atIdx >= 0) {
        // Build qualified class name including namespace prefix
        // e.g. "+geometry/@Point/Point.m" → "geometry.Point"
        const nsDirs = parts.slice(0, atIdx).filter(d => d.startsWith("+"));
        const baseClassName = parts[atIdx].slice(1); // remove "@"
        const className =
          nsDirs.length > 0
            ? [...nsDirs.map(d => d.slice(1)), baseClassName].join(".")
            : baseClassName;
        const baseName = parts[parts.length - 1].replace(/\.m$/, "");

        let group = classFolderGroups.get(className);
        if (!group) {
          group = { methodFiles: [] };
          classFolderGroups.set(className, group);
        }
        if (baseName === baseClassName) {
          group.classDefFile = file;
        } else {
          group.methodFiles.push(file);
        }
        continue;
      }

      // Check namespace parts: all directories must be +prefixed or it must be top-level
      const dirs = parts.slice(0, -1);
      const isNamespace = dirs.length > 0 && dirs.every(d => d.startsWith("+"));
      if (dirs.length > 0 && !isNamespace) continue; // skip non-namespace subdirs

      const baseName = parts[parts.length - 1].replace(/\.m$/, "");
      // +pkg/+sub/func.m → "pkg.sub.func"
      const funcName = isNamespace
        ? [...dirs.map(d => d.slice(1)), baseName].join(".")
        : baseName;

      this.fileToFuncName.set(file.name, funcName);

      if (startsWithClassdef(file.source)) {
        this.registerWorkspaceClass(funcName, file);
      } else if (!this.filesByFuncName.has(funcName)) {
        this.filesByFuncName.set(funcName, {
          fileName: file.name,
          source: file.source,
        });
      }
    }

    // Second pass: register @folder class groups
    for (const [className, group] of classFolderGroups) {
      // The @Name/Name.m file lands in classDefFile by name match, but for an
      // old-style (pre-classdef) class it is a plain constructor function, not
      // a classdef. Detect that and route to the old-style path.
      const hasRealClassdef =
        !!group.classDefFile && startsWithClassdef(group.classDefFile.source);

      if (!hasRealClassdef) {
        // Old-style (pre-classdef) class. The constructor is @Name/Name.m
        // (the basename-matched file, which here sits in classDefFile); the
        // rest are external methods dispatched on instances.
        const dotIdx = className.lastIndexOf(".");
        const baseName = dotIdx >= 0 ? className.slice(dotIdx + 1) : className;
        const methodBase = (f: WorkspaceFile) =>
          f.name.split("/").pop()!.replace(/\.m$/, "");
        const ctorFile =
          group.classDefFile ??
          group.methodFiles.find(f => methodBase(f) === baseName);
        if (!ctorFile) {
          // No constructor file (e.g. only helper methods): cannot construct
          // instances, so skip — same rationale as the classdef path (would
          // shadow same-named functions from other directories).
          continue;
        }
        const methodFiles = group.methodFiles
          .filter(f => f !== ctorFile)
          .map(f => ({
            name: methodBase(f),
            fileName: f.name,
            source: f.source,
          }));
        this.classes.addWorkspaceClass(
          makeOldStyleClassInfo(
            className,
            baseName,
            { fileName: ctorFile.name, source: ctorFile.source },
            methodFiles
          )
        );
        continue;
      }

      // Register the class from the classdef file
      this.registerWorkspaceClass(className, group.classDefFile!);

      // Attach external method files to the class info
      const info = this.classes.workspaceClasses.get(className);
      if (info) {
        for (const mf of group.methodFiles) {
          const methodName = mf.name.split("/").pop()!.replace(/\.m$/, "");
          info.externalMethodFiles.set(methodName, {
            fileName: mf.name,
            source: mf.source,
          });
          // Only add to methodNames if not already declared as static in the classdef
          if (!info.staticMethodNames.has(methodName)) {
            info.methodNames.add(methodName);
          }
        }
      }
    }
  }

  /** Register a workspace file whose source starts with `classdef`. A file
   *  without a ClassDef statement registers as a function instead. */
  private registerWorkspaceClass(
    qualifiedName: string,
    file: WorkspaceFile
  ): void {
    const ast = this.getCachedAST(file.name);
    const classDef = ast.body.find(
      (s): s is Stmt & { type: "ClassDef" } => s.type === "ClassDef"
    );
    if (!classDef) {
      this.filesByFuncName.set(qualifiedName, {
        fileName: file.name,
        source: file.source,
      });
      return;
    }
    this.classes.addWorkspaceClass(
      extractClassInfo(classDef, qualifiedName, file.name, file.source)
    );
  }

  /** Register a JS user function (.numbl.js). First wins, so search-path
   *  priority is honored. */
  registerJsUserFunction(
    funcName: string,
    fileName: string,
    builtin: IBuiltin,
    help?: string
  ): void {
    if (this.jsUserFunctionsByName.has(funcName)) return;
    this.jsUserFunctionsByName.set(funcName, { fileName, builtin, help });
  }

  /** Clear workspace registrations so they can be rebuilt after
   *  addpath/rmpath/cd. The main file's functions and classes stay. */
  clearWorkspaceRegistrations(): void {
    this.filesByFuncName.clear();
    this.jsUserFunctionsByName.clear();
    this.classes.clearWorkspaceClasses();
    this.privateFilesByDir.clear();
    this.fileToFuncName.clear();
    this.index = null;
  }

  /** The private function `funcName` visible from `callerFileName`. */
  getPrivateFileEntry(
    callerFileName: string,
    funcName: string
  ): FunctionFile | null {
    const dir = callerPrivateScope(callerFileName, this.searchPaths);
    return this.privateFilesByDir.get(dir)?.get(funcName) ?? null;
  }

  // ── Function index ────────────────────────────────────────────────

  /** Build (or rebuild) the function index from the current
   *  registrations. */
  buildFunctionIndex(): FunctionIndex {
    this.index = buildFunctionIndex(this);
    return this.index;
  }

  /** The function index; `buildFunctionIndex()` must have been called. */
  get functionIndex(): FunctionIndex {
    if (!this.index) {
      throw new Error(
        "FunctionIndex not built yet — call buildFunctionIndex() first"
      );
    }
    return this.index;
  }
}

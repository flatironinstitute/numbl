/**
 * On-demand lowering context.
 *
 * Purpose-built for the execution pipeline. Manages variable scopes,
 * identifier resolution, and on-demand function lowering. Simpler than
 * LowerContextForFileObj (no workspace/cross-file context), but covers
 * the same core responsibilities for a single file.
 */

import { type AbstractSyntaxTree, type Stmt } from "../parser/index.js";
import { BUILTIN_CONSTANTS } from "../lowering/constants.js";
import { type ItemType } from "../lowering/itemTypes.js";
import { type IRVariable } from "../lowering/loweringTypes.js";
import { type IRStmt } from "../lowering/nodes.js";
import { VarId } from "../lowering/varId.js";
import { getAllBuiltinNames } from "../builtins";
import type { WorkspaceFile } from "../../numbl-core/workspace/index.js";
import { type ClassInfo, extractClassInfo } from "./classInfo.js";
import { computeSpecKey } from "./specKey.js";
import { lowerFunction } from "./lowerStmt.js";

// Re-export ClassInfo for consumers
export type { ClassInfo } from "./classInfo.js";

/** Cache of parsed file ASTs, keyed by fileName. Built once at the outset. */
export type FileASTCache = Map<string, AbstractSyntaxTree>;

// ── Shared registry ──────────────────────────────────────────────────────
// Holds workspace-level data shared by reference between the main context
// and all child contexts (workspace file contexts, class file contexts).
// One object to copy instead of four separate fields.

interface WorkspaceRegistry {
  /** Raw workspace files: functionName → { fileName, source } */
  filesByFuncName: Map<string, { fileName: string; source: string }>;
  /** Per-workspace-file lowering contexts (created on demand) */
  fileContexts: Map<string, LoweringContext>;
  /** Workspace classes: qualifiedName → ClassInfo */
  classesByName: Map<string, ClassInfo>;
  /** Local classes (classdef in same file): className → ClassInfo */
  localClassesByName: Map<string, ClassInfo>;
  /** Private function files: parentDir → funcName → { fileName, source } */
  privateFilesByDir: Map<
    string,
    Map<string, { fileName: string; source: string }>
  >;
  /** Shared function index (built once, accessible from all contexts) */
  functionIndex: FunctionIndex | null;
  /** Cached parsed ASTs: fileName → AST (shared across all contexts) */
  fileASTCache: FileASTCache;
  /** Ordered search paths (absolute). Empty = backward-compat mode. */
  searchPaths: string[];
  /** Absolute fileName → resolved function name (e.g. "pkg.func") */
  fileToFuncName: Map<string, string>;
}

function createWorkspaceRegistry(): WorkspaceRegistry {
  return {
    filesByFuncName: new Map(),
    fileContexts: new Map(),
    classesByName: new Map(),
    localClassesByName: new Map(),
    privateFilesByDir: new Map(),
    functionIndex: null,
    fileASTCache: new Map(),
    searchPaths: [],
    fileToFuncName: new Map(),
  };
}

// ── Function Index ──────────────────────────────────────────────────────
// A complete, upfront index mapping function names to where they exist.
// Built once after workspace files are registered; enables definitive
// (single-target) resolution without trial-and-error compilation.

export interface FunctionIndex {
  /** Built-in function names */
  builtins: Set<string>;

  /** Main script file name (used to scope mainLocalFunctions) */
  mainFileName: string;

  /** Main script local function names */
  mainLocalFunctions: Set<string>;

  /** Primary workspace functions (filename-derived names) */
  workspaceFunctions: Set<string>;

  /** JS user functions (resolved at workspace-function priority, not builtin) */
  jsUserFunctions: Set<string>;

  /** Workspace classes (classdef files) */
  workspaceClasses: Set<string>;

  /** Local subfunctions per workspace file: primaryFuncName → Set<subfuncName> */
  workspaceFileSubfunctions: Map<string, Set<string>>;

  /** Local subfunctions per class file: className → Set<subfuncName> */
  classFileSubfunctions: Map<string, Set<string>>;

  /** Class instance methods: className → Set<methodName> (flattened through inheritance) */
  classInstanceMethods: Map<string, Set<string>>;

  /** Class static methods: className → Set<methodName> (flattened through inheritance) */
  classStaticMethods: Map<string, Set<string>>;

  /** Class constructor names: className → constructorName */
  classConstructors: Map<string, string>;

  /** Private functions: parentDir → Set<funcName> */
  privateFunctions: Map<string, Set<string>>;

  /** Local subfunctions per private file: "dir/funcName" → Set<subfuncName> */
  privateFileSubfunctions: Map<string, Set<string>>;

  /** InferiorClasses: superiorClassName → Set<inferiorClassName> */
  classInferiorClasses: Map<string, Set<string>>;

  /** Ordered search paths for relative path computation */
  searchPaths: string[];

  /** Absolute fileName → resolved function name (e.g. "pkg.func") */
  fileToFuncName: Map<string, string>;
}

// ── Scope ───────────────────────────────────────────────────────────────

interface Scope {
  parent: number | null;
  bindings: Map<string, IRVariable>;
}

// ── Context ─────────────────────────────────────────────────────────────

export class LoweringContext {
  // Scope chain
  public scopes: Scope[] = [
    { parent: null, bindings: new Map<string, IRVariable>() },
  ];

  // Variable ID generation (unique across all functions in this file)
  private nextVarCounters: Record<string, number> = {};

  // All variables created during lowering
  readonly allVariables: IRVariable[] = [];

  // Local function ASTs (parsed but not yet lowered)
  private localFunctionASTs = new Map<string, Stmt & { type: "Function" }>();

  // Lowered local functions (populated on demand)
  private loweredFunctions = new Map<string, IRStmt & { type: "Function" }>();

  // The function currently being lowered (for recursion detection)
  private loweringInProgress = new Set<string>();

  // Specialized lowered functions: specKey → lowered IR
  private specializedFunctions = new Map<
    string,
    IRStmt & { type: "Function" }
  >();

  // Tracks which specializations are currently being lowered (for cycle detection)
  private specializationInProgress = new Set<string>();

  // Track whether we're at the top level (script scope)
  isTopLevel = false;

  // ── Shared workspace/class registry ─────────────────────────────
  // Shared by reference between this context and all child contexts
  // (workspace file contexts, class file contexts). See WorkspaceRegistry.
  registry = createWorkspaceRegistry();

  // If this context is a class file context, the class name it belongs to
  ownerClassName: string | null = null;

  // Per-external-method-file local functions (methodName → local helper ASTs)
  // These are scoped to individual method files and temporarily installed during lowering
  externalMethodLocalFunctions = new Map<
    string,
    (Stmt & { type: "Function" })[]
  >();

  constructor(
    public readonly fileSource: string,
    public readonly mainFileName: string
  ) {}

  /**
   * Compute the relative path of a file within its search path.
   * When no search paths are configured, returns the path as-is (backward compat).
   * Uses the most specific (longest) matching search path to handle
   * overlapping paths (e.g. /a/b and /a/b/tools both on the search path).
   */
  getRelativePath(absolutePath: string): string {
    if (this.registry.searchPaths.length === 0) return absolutePath;
    let bestPrefix = "";
    for (const sp of this.registry.searchPaths) {
      const prefix = sp.endsWith("/") ? sp : sp + "/";
      if (
        absolutePath.startsWith(prefix) &&
        prefix.length > bestPrefix.length
      ) {
        bestPrefix = prefix;
      }
    }
    if (bestPrefix) return absolutePath.slice(bestPrefix.length);
    return absolutePath; // fallback: no matching search path
  }

  /** Look up a pre-parsed AST by fileName. Throws if not cached. */
  getCachedAST(fileName: string): AbstractSyntaxTree {
    const cached = this.registry.fileASTCache.get(fileName);
    if (!cached) {
      throw new Error(
        `FileASTCache miss: no cached AST for "${fileName}". All files must be pre-parsed.`
      );
    }
    return cached;
  }

  /** Get the shared file AST cache (for pre-populating from the outside). */
  get fileASTCache(): FileASTCache {
    return this.registry.fileASTCache;
  }

  /**
   * Create a fresh context that shares only the workspace registry.
   * Used for JIT compilation, which compiles workspace/class functions
   * in other files — not the main script's local functions.
   */
  createJitContext(): LoweringContext {
    const ctx = new LoweringContext("", "<jit>");
    ctx.registry = this.registry;
    return ctx;
  }

  // ── Scope management ──────────────────────────────────────────────

  pushScope(): void {
    const parent = this.scopes.length - 1;
    this.scopes.push({ parent, bindings: new Map() });
  }

  /** Push a scope isolated from the current scope chain (for subfunctions). */
  pushIsolatedScope(): void {
    this.scopes.push({ parent: null, bindings: new Map() });
  }

  popScope(): void {
    this.scopes.pop();
  }

  // ── Variable management ───────────────────────────────────────────

  defineVariable(name: string, ty: ItemType | undefined): IRVariable {
    if (!(name in this.nextVarCounters)) {
      this.nextVarCounters[name] = 0;
    }
    const idStr = `${name}_${this.nextVarCounters[name]}`;
    this.nextVarCounters[name]++;
    const id = new VarId(idStr);
    const isTopLevel = this.isTopLevel;
    const newVar: IRVariable = { id, name, ty, isTopLevel };
    const current = this.scopes.length - 1;
    this.scopes[current].bindings.set(name, newVar);
    this.allVariables.push(newVar);
    return newVar;
  }

  lookup(name: string): IRVariable | null {
    let scopeIdx: number | null = this.scopes.length - 1;
    while (scopeIdx !== null && scopeIdx >= 0) {
      const vv = this.scopes[scopeIdx].bindings.get(name);
      if (vv !== undefined) {
        return vv;
      }
      scopeIdx = this.scopes[scopeIdx].parent;
    }
    return null;
  }

  // ── Symbol resolution ─────────────────────────────────────────────

  isConstant(name: string): boolean {
    return BUILTIN_CONSTANTS.has(name);
  }

  private isLocalFunction(name: string): boolean {
    return this.localFunctionASTs.has(name);
  }

  // ── Local function management ─────────────────────────────────────

  /** Register a local function's AST (called before lowering begins). */
  registerLocalFunctionAST(stmt: Stmt & { type: "Function" }): void {
    this.localFunctionASTs.set(stmt.name, stmt);
  }

  /** Unregister a local function's AST and clear its cached specializations. */
  unregisterLocalFunctionAST(name: string): void {
    this.localFunctionASTs.delete(name);
    this.loweredFunctions.delete(name);
    // Clear any specializations for this function (specKey is JSON with "name" field)
    for (const key of this.specializedFunctions.keys()) {
      const parsed = JSON.parse(key);
      if (parsed.name === name) {
        this.specializedFunctions.delete(key);
      }
    }
  }

  /**
   * Get a lowered function by name, lowering it on demand if needed.
   * Returns null if the function is not a local function.
   */
  getOrLowerFunction(name: string): (IRStmt & { type: "Function" }) | null {
    // Already lowered?
    const existing = this.loweredFunctions.get(name);
    if (existing) return existing;

    // Not a local function?
    const ast = this.localFunctionASTs.get(name);
    if (!ast) return null;

    // Currently being lowered? (recursive call) — return a stub
    if (this.loweringInProgress.has(name)) {
      return this.createFunctionStub(ast);
    }

    // Lower it
    this.loweringInProgress.add(name);
    const lowered = lowerFunction(this, ast);
    this.loweringInProgress.delete(name);
    this.loweredFunctions.set(name, lowered);
    return lowered;
  }

  /**
   * Get or lower a specialized version of a function.
   * Returns null if the function is not a local function.
   * For recursive calls detected via specializationInProgress, returns a stub.
   */
  getOrLowerFunctionSpecialized(
    name: string,
    argTypes: ItemType[]
  ): (IRStmt & { type: "Function" }) | null {
    const specKey = computeSpecKey(name, argTypes);

    // Already specialized?
    const existing = this.specializedFunctions.get(specKey);
    if (existing) return existing;

    // Not a local function?
    const ast = this.localFunctionASTs.get(name);
    if (!ast) return null;

    // Currently being lowered? (recursive call) — return a stub
    if (this.specializationInProgress.has(specKey)) {
      return this.createFunctionStub(ast);
    }

    // Lower it with the specialized param types
    this.specializationInProgress.add(specKey);
    const lowered = lowerFunction(this, ast, argTypes);
    this.specializationInProgress.delete(specKey);
    this.specializedFunctions.set(specKey, lowered);
    return lowered;
  }

  /** Get a pre-registered stub for a local function (for candidates). */
  getLocalFunctionStub(name: string): (IRStmt & { type: "Function" }) | null {
    const lowered = this.loweredFunctions.get(name);
    if (lowered) return lowered;
    const ast = this.localFunctionASTs.get(name);
    if (!ast) return null;
    return this.createFunctionStub(ast);
  }

  /** Create a minimal stub for a function (before it's fully lowered). */
  private createFunctionStub(
    ast: Stmt & { type: "Function" }
  ): IRStmt & { type: "Function" } {
    return {
      type: "Function",
      originalName: ast.name,
      functionId: ast.functionId,
      params: [],
      outputs: [],
      body: [],
      hasVarargin: false,
      hasVarargout: false,
      argumentsBlocks: [],
      span: ast.span,
    };
  }

  /** Get all lowered functions (for codegen). */
  getLoweredFunctions(): Map<string, IRStmt & { type: "Function" }> {
    return this.loweredFunctions;
  }

  /** Pre-register a lowered function (used when lowering function statements). */
  registerLoweredFunction(stmt: IRStmt & { type: "Function" }): void {
    this.loweredFunctions.set(stmt.originalName, stmt);
  }

  // ── Workspace function management ───────────────────────────────

  /**
   * Register workspace files for on-demand resolution.
   * Extracts function names from filenames (top-level only, plus +pkg namespaces).
   * Also handles @ClassName class folders.
   */
  registerWorkspaceFiles(files: WorkspaceFile[]): void {
    // First pass: collect @folder groups
    const classFolderGroups = new Map<
      string,
      { classDefFile?: WorkspaceFile; methodFiles: WorkspaceFile[] }
    >();

    for (const file of files) {
      // Compute relative path from search paths for name resolution
      const relativePath = this.getRelativePath(file.name);
      const parts = relativePath.split("/");

      // Check for @ClassName folder
      const atIdx = parts.findIndex(p => p.startsWith("@"));
      if (atIdx >= 0) {
        // Files in @ClassName/private/ are private helper functions,
        // not methods. Register them as private functions scoped to the @ClassName/ dir.
        if (parts.includes("private")) {
          const privateIdx = parts.indexOf("private");
          const parentDir =
            parts.slice(0, privateIdx).join("/") + (privateIdx > 0 ? "/" : "");
          const baseName = parts[parts.length - 1].replace(/\.m$/, "");
          if (!this.registry.privateFilesByDir.has(parentDir)) {
            this.registry.privateFilesByDir.set(parentDir, new Map());
          }
          this.registry.privateFilesByDir.get(parentDir)!.set(baseName, {
            fileName: file.name,
            source: file.source,
          });
          continue;
        }

        // Build qualified class name including namespace prefix
        // e.g. "+geometry/@Point/Point.m" → "geometry.Point"
        const nsDirs = parts.slice(0, atIdx).filter(d => d.startsWith("+"));
        const baseClassName = parts[atIdx].slice(1); // remove "@"
        const className =
          nsDirs.length > 0
            ? [...nsDirs.map(d => d.slice(1)), baseClassName].join(".")
            : baseClassName;
        const fileName = parts[parts.length - 1];
        const baseName = fileName.replace(/\.m$/, "");

        if (!classFolderGroups.has(className)) {
          classFolderGroups.set(className, { methodFiles: [] });
        }
        const group = classFolderGroups.get(className)!;

        if (baseName === baseClassName) {
          group.classDefFile = file;
        } else {
          group.methodFiles.push(file);
        }
        continue;
      }

      // Check for private/ directory
      const privateIdx = parts.indexOf("private");
      if (privateIdx >= 0) {
        // e.g., "private/helper.m" or "+pkg/private/helper.m"
        const parentDir =
          parts.slice(0, privateIdx).join("/") + (privateIdx > 0 ? "/" : "");
        const baseName = parts[parts.length - 1].replace(/\.m$/, "");
        if (!this.registry.privateFilesByDir.has(parentDir)) {
          this.registry.privateFilesByDir.set(parentDir, new Map());
        }
        this.registry.privateFilesByDir.get(parentDir)!.set(baseName, {
          fileName: file.name,
          source: file.source,
        });
        continue;
      }

      // Check namespace parts: all directories must be +prefixed or it must be top-level
      const dirs = parts.slice(0, -1);
      const isNamespace = dirs.length > 0 && dirs.every(d => d.startsWith("+"));
      if (dirs.length > 0 && !isNamespace) continue; // skip non-namespace subdirs

      const baseName = parts[parts.length - 1].replace(/\.m$/, "");
      let funcName: string;
      if (isNamespace) {
        // +pkg/+sub/func.m → "pkg.sub.func"
        const nsParts = dirs.map(d => d.slice(1));
        funcName = [...nsParts, baseName].join(".");
      } else {
        funcName = baseName;
      }

      // Track file → funcName mapping
      this.registry.fileToFuncName.set(file.name, funcName);

      // Semi-eager class detection: if the file starts with 'classdef' (after
      // stripping whitespace and leading comments), register as class
      let trimmed = file.source.trimStart();
      while (trimmed.startsWith("%")) {
        const nl = trimmed.indexOf("\n");
        if (nl < 0) break;
        trimmed = trimmed.slice(nl + 1).trimStart();
      }
      if (trimmed.startsWith("classdef")) {
        this.registerWorkspaceClass(funcName, file);
      } else {
        // First-wins: earlier search paths take priority
        if (!this.registry.filesByFuncName.has(funcName)) {
          this.registry.filesByFuncName.set(funcName, {
            fileName: file.name,
            source: file.source,
          });
        }
      }
    }

    // Second pass: register @folder class groups
    for (const [className, group] of classFolderGroups) {
      if (!group.classDefFile) {
        // No classdef file — treat method files as regular workspace files
        for (const mf of group.methodFiles) {
          const baseName = mf.name.split("/").pop()!.replace(/\.m$/, "");
          this.registry.filesByFuncName.set(baseName, {
            fileName: mf.name,
            source: mf.source,
          });
        }
        continue;
      }

      // Register the class from the classdef file
      this.registerWorkspaceClass(className, group.classDefFile);

      // Attach external method files to the class info
      const info = this.registry.classesByName.get(className);
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

  isWorkspaceFunction(name: string): boolean {
    return this.registry.filesByFuncName.has(name);
  }

  // ── Private function management ──────────────────────────────────

  /**
   * Get the effective directory for this context's file, used for
   * private function resolution. If the file is inside a private/ folder,
   * the effective dir is the parent of private/ (so private functions
   * can call each other).
   */
  private getEffectiveDir(): string {
    const relativePath = this.getRelativePath(this.mainFileName);
    const parts = relativePath.split("/");
    parts.pop(); // remove filename
    if (parts.length > 0 && parts[parts.length - 1] === "private") {
      parts.pop(); // strip private/
    }
    return parts.length > 0 ? parts.join("/") + "/" : "";
  }

  getOrCreatePrivateFileContext(funcName: string): LoweringContext | null {
    const dir = this.getEffectiveDir();
    const dirMap = this.registry.privateFilesByDir.get(dir);
    if (!dirMap) return null;
    const entry = dirMap.get(funcName);
    if (!entry) return null;

    // Use a cache key that includes the directory to avoid collisions
    const cacheKey = `private:${dir}${funcName}`;
    const cached = this.registry.fileContexts.get(cacheKey);
    if (cached) return cached;

    const ast = this.getCachedAST(entry.fileName);
    const ctx = new LoweringContext(entry.source, entry.fileName);
    for (const stmt of ast.body) {
      if (stmt.type === "Function") {
        ctx.registerLocalFunctionAST(stmt);
      }
    }
    ctx.registry = this.registry;
    this.registry.fileContexts.set(cacheKey, ctx);
    return ctx;
  }

  getOrLowerPrivateFunctionSpecialized(
    funcName: string,
    argTypes: ItemType[]
  ): (IRStmt & { type: "Function" }) | null {
    const ctx = this.getOrCreatePrivateFileContext(funcName);
    if (!ctx) return null;
    return ctx.getOrLowerFunctionSpecialized(funcName, argTypes);
  }

  /**
   * Look up a private function entry by effective directory and name.
   * Used by JIT to resolve private functions from a specific calling file.
   */
  getPrivateFileEntry(
    callerFileName: string,
    funcName: string
  ): { fileName: string; source: string } | null {
    const relativePath = this.getRelativePath(callerFileName);
    const parts = relativePath.split("/");
    parts.pop();
    if (parts.length > 0 && parts[parts.length - 1] === "private") {
      parts.pop();
    }
    const dir = parts.length > 0 ? parts.join("/") + "/" : "";
    return this.registry.privateFilesByDir.get(dir)?.get(funcName) ?? null;
  }

  /**
   * Get or create an LoweringContext for a workspace file.
   * Parses the file on demand and registers its local functions.
   */
  getOrCreateWorkspaceFileContext(funcName: string): LoweringContext | null {
    const cached = this.registry.fileContexts.get(funcName);
    if (cached) return cached;

    const entry = this.registry.filesByFuncName.get(funcName);
    if (!entry) return null;

    // Look up the pre-parsed AST
    const ast = this.getCachedAST(entry.fileName);

    // Create a new context for this file
    const ctx = new LoweringContext(entry.source, entry.fileName);

    // Register local functions from the parsed AST
    for (const stmt of ast.body) {
      if (stmt.type === "Function") {
        ctx.registerLocalFunctionAST(stmt);
      }
    }

    // Share the workspace registry
    ctx.registry = this.registry;

    this.registry.fileContexts.set(funcName, ctx);
    return ctx;
  }

  /**
   * Get or lower a specialized workspace function.
   * The function name is the primary function of the workspace file (matches filename).
   */
  getOrLowerWorkspaceFunctionSpecialized(
    funcName: string,
    argTypes: ItemType[]
  ): (IRStmt & { type: "Function" }) | null {
    const ctx = this.getOrCreateWorkspaceFileContext(funcName);
    if (!ctx) return null;

    // The primary function name is the last segment (e.g., "pkg.func" → "func")
    const dotIdx = funcName.lastIndexOf(".");
    const primaryName = dotIdx >= 0 ? funcName.slice(dotIdx + 1) : funcName;

    return ctx.getOrLowerFunctionSpecialized(primaryName, argTypes);
  }

  /** Get the workspace file context (for codegen to temporarily swap contexts). */
  getWorkspaceFileContext(funcName: string): LoweringContext | null {
    return this.registry.fileContexts.get(funcName) ?? null;
  }

  // ── Class management ────────────────────────────────────────────────

  /**
   * Register a workspace file as a class definition.
   * Parses the file to extract class structure (properties, methods).
   */
  private registerWorkspaceClass(
    qualifiedName: string,
    file: WorkspaceFile
  ): void {
    const ast = this.getCachedAST(file.name);
    const classDef = ast.body.find(
      (s): s is Stmt & { type: "ClassDef" } => s.type === "ClassDef"
    );
    if (!classDef) {
      // Not actually a class — register as a regular workspace function
      this.registry.filesByFuncName.set(qualifiedName, {
        fileName: file.name,
        source: file.source,
      });
      return;
    }

    const info = extractClassInfo(
      classDef,
      qualifiedName,
      file.name,
      file.source
    );
    this.registry.classesByName.set(qualifiedName, info);
  }

  /**
   * Register a classdef statement found in the local (main) file.
   */
  registerLocalClass(stmt: Stmt & { type: "ClassDef" }): void {
    const info = extractClassInfo(
      stmt,
      stmt.name,
      this.mainFileName,
      this.fileSource
    );
    this.registry.localClassesByName.set(stmt.name, info);
  }

  isWorkspaceClass(name: string): boolean {
    return this.registry.classesByName.has(name);
  }

  isLocalClass(name: string): boolean {
    return this.registry.localClassesByName.has(name);
  }

  isClass(name: string): boolean {
    return this.isWorkspaceClass(name) || this.isLocalClass(name);
  }

  getClassInfo(name: string): ClassInfo | null {
    return (
      this.registry.classesByName.get(name) ??
      this.registry.localClassesByName.get(name) ??
      null
    );
  }

  /** Check if a class (or any ancestor) has a given method. */
  classHasMethod(className: string, methodName: string): boolean {
    let current: string | null = className;
    while (current) {
      const info = this.getClassInfo(current);
      if (!info) return false;
      if (info.methodNames.has(methodName)) return true;
      current = info.superClass;
    }
    return false;
  }

  /** Check if a class (or any ancestor) has a given static method. */
  classHasStaticMethod(className: string, methodName: string): boolean {
    let current: string | null = className;
    while (current) {
      const info = this.getClassInfo(current);
      if (!info) return false;
      if (info.staticMethodNames.has(methodName)) return true;
      current = info.superClass;
    }
    return false;
  }

  /**
   * Walk the inheritance chain to find the class that defines a given method.
   * Checks instance methods, external method files, static methods, and constructors.
   */
  findDefiningClass(className: string, methodName: string): string {
    let current: string | null = className;
    while (current) {
      const info = this.getClassInfo(current);
      if (!info) break;
      if (
        info.methodNames.has(methodName) ||
        info.externalMethodFiles.has(methodName) ||
        info.staticMethodNames.has(methodName) ||
        methodName === info.constructorName
      ) {
        return current;
      }
      current = info.superClass;
    }
    return className;
  }

  /**
   * Get or create an LoweringContext for a class file.
   * Registers all methods as local functions so they can be specialized on demand.
   * For constructors, the AST is modified to prepend the output variable as a parameter
   * (Instance is passed as first arg).
   */
  getOrCreateClassFileContext(className: string): LoweringContext | null {
    const info = this.getClassInfo(className);
    if (!info) return null;
    if (info.ctx) return info.ctx;

    // Create a new context for this class file
    const ctx = new LoweringContext(info.source, info.fileName);

    // Register all methods as local functions
    for (const member of info.ast.members) {
      if (member.type !== "Methods") continue;
      for (const methodStmt of member.body) {
        if (methodStmt.type !== "Function") continue;

        if (methodStmt.name === info.constructorName) {
          // Constructor: prepend the output variable as a hidden first parameter.
          // Constructors have `function obj = ClassName(w, h)` where
          // obj is the output. We transform this to accept obj as first param,
          // matching the old system's calling convention.
          const outputName =
            methodStmt.outputs.length > 0 ? methodStmt.outputs[0] : "obj";
          const transformedStmt: Stmt & { type: "Function" } = {
            ...methodStmt,
            params: [outputName, ...methodStmt.params],
          };
          ctx.registerLocalFunctionAST(transformedStmt);
        } else {
          ctx.registerLocalFunctionAST(methodStmt);
        }
      }
    }

    // Register file-local functions defined after the classdef block
    // (e.g., helper subfunctions at the end of the .m file)
    const fullAst = this.getCachedAST(info.fileName);
    for (const stmt of fullAst.body) {
      if (stmt.type === "Function") {
        // Skip if already registered as a class method
        if (!ctx.isLocalFunction(stmt.name)) {
          ctx.registerLocalFunctionAST(stmt);
        }
      }
    }

    // Register external method files (from @ClassName/ folders)
    for (const [_methodName, mf] of info.externalMethodFiles) {
      const methodAst = this.getCachedAST(mf.fileName);
      const localHelpers: (Stmt & { type: "Function" })[] = [];
      for (const s of methodAst.body) {
        if (s.type === "Function") {
          if (s.name === _methodName) {
            // Register the primary method in the class context
            ctx.registerLocalFunctionAST(s);
          } else {
            // Collect local helpers (per-file scope)
            localHelpers.push(s);
          }
        }
      }
      if (localHelpers.length > 0) {
        ctx.externalMethodLocalFunctions.set(_methodName, localHelpers);
      }
    }

    // Share the workspace registry
    ctx.registry = this.registry;

    ctx.ownerClassName = className;
    info.ctx = ctx;
    return ctx;
  }

  /**
   * Get or lower a specialized class method.
   */
  getOrLowerClassMethodSpecialized(
    className: string,
    methodName: string,
    argTypes: ItemType[]
  ): (IRStmt & { type: "Function" }) | null {
    const ctx = this.getOrCreateClassFileContext(className);
    if (!ctx) return null;

    return ctx.withMethodScope(methodName, () =>
      ctx.getOrLowerFunctionSpecialized(methodName, argTypes)
    );
  }

  /**
   * Run a callback with per-file local helper functions temporarily installed
   * for the given method name. Ensures cleanup even on exception.
   */
  withMethodScope<T>(methodName: string, fn: () => T): T {
    const helpers = this.externalMethodLocalFunctions.get(methodName);
    // Save any existing local functions that will be shadowed by per-method helpers,
    // so we can restore them after the method scope exits. Without this, a local
    // helper in an external method file (e.g., inv.m's parseInputs) would permanently
    // delete the classdef file's local function with the same name.
    const savedLocals = new Map<string, Stmt & { type: "Function" }>();
    if (helpers) {
      for (const h of helpers) {
        const existing = this.localFunctionASTs.get(h.name);
        if (existing) {
          savedLocals.set(h.name, existing);
        }
        this.registerLocalFunctionAST(h);
      }
    }
    try {
      return fn();
    } finally {
      if (helpers) {
        for (const h of helpers) {
          this.unregisterLocalFunctionAST(h.name);
        }
        // Restore any local functions that were shadowed
        for (const [, ast] of savedLocals) {
          this.registerLocalFunctionAST(ast);
        }
      }
    }
  }

  /**
   * Build a FunctionIndex from the current state of the registry and local functions.
   * Should be called once after registerWorkspaceFiles() and registerLocalFunctionAST().
   * Parses all workspace files eagerly to discover subfunctions.
   */
  buildFunctionIndex(jsUserFunctionNames?: string[]): FunctionIndex {
    // 1. Builtins
    const builtins = new Set(getAllBuiltinNames());

    // Separate JS user functions from builtins — they resolve at workspace priority
    const jsUserFunctions = new Set<string>();
    if (jsUserFunctionNames) {
      for (const name of jsUserFunctionNames) {
        if (builtins.has(name)) {
          builtins.delete(name);
          jsUserFunctions.add(name);
        }
      }
    }

    // 2. Main script local functions
    const mainLocalFunctions = new Set(this.localFunctionASTs.keys());

    // 3. Primary workspace functions
    const workspaceFunctions = new Set(this.registry.filesByFuncName.keys());

    // 4. All classes (workspace + local)
    const workspaceClasses = new Set([
      ...this.registry.classesByName.keys(),
      ...this.registry.localClassesByName.keys(),
    ]);

    // 5. Workspace file subfunctions — parse each file to discover function names
    const workspaceFileSubfunctions = new Map<string, Set<string>>();
    for (const [funcName, entry] of this.registry.filesByFuncName) {
      const ast = this.getCachedAST(entry.fileName);
      const subs = new Set<string>();
      for (const stmt of ast.body) {
        if (stmt.type === "Function" && stmt.name !== funcName) {
          subs.add(stmt.name);
        }
      }
      if (subs.size > 0) {
        workspaceFileSubfunctions.set(funcName, subs);
      }
    }

    // 6. Class file subfunctions — functions in a class file that aren't methods/constructor
    const classFileSubfunctions = new Map<string, Set<string>>();
    for (const [className, info] of this.registry.classesByName) {
      const subs = new Set<string>();

      // Look up the class file to find top-level functions (non-method helpers)
      const classAst = this.getCachedAST(info.fileName);
      for (const stmt of classAst.body) {
        if (
          stmt.type === "Function" &&
          !info.methodNames.has(stmt.name) &&
          !info.staticMethodNames.has(stmt.name) &&
          stmt.name !== info.constructorName
        ) {
          subs.add(stmt.name);
        }
      }

      // Also collect local helpers from external method files
      for (const [methodName, mf] of info.externalMethodFiles) {
        const methodAst = this.getCachedAST(mf.fileName);
        for (const stmt of methodAst.body) {
          if (stmt.type === "Function" && stmt.name !== methodName) {
            subs.add(stmt.name);
          }
        }
      }

      if (subs.size > 0) {
        classFileSubfunctions.set(className, subs);
      }
    }
    // Also handle local classes
    for (const [className, info] of this.registry.localClassesByName) {
      const subs = new Set<string>();
      const classAst = this.getCachedAST(info.fileName);
      for (const stmt of classAst.body) {
        if (
          stmt.type === "Function" &&
          !info.methodNames.has(stmt.name) &&
          !info.staticMethodNames.has(stmt.name) &&
          stmt.name !== info.constructorName
        ) {
          subs.add(stmt.name);
        }
      }
      if (subs.size > 0) {
        classFileSubfunctions.set(className, subs);
      }
    }

    // 7. Class methods — flatten through inheritance chain
    const classInstanceMethods = new Map<string, Set<string>>();
    const classStaticMethods = new Map<string, Set<string>>();
    const allClasses = [
      ...this.registry.classesByName.entries(),
      ...this.registry.localClassesByName.entries(),
    ];
    for (const [className] of allClasses) {
      const instanceMethods = new Set<string>();
      const staticMethods = new Set<string>();
      let current: string | null = className;
      while (current) {
        const info = this.getClassInfo(current);
        if (!info) break;
        for (const m of info.methodNames) instanceMethods.add(m);
        for (const m of info.staticMethodNames) staticMethods.add(m);
        // Include external method files
        for (const m of info.externalMethodFiles.keys()) {
          if (!info.staticMethodNames.has(m)) {
            instanceMethods.add(m);
          }
        }
        current = info.superClass;
      }
      if (instanceMethods.size > 0)
        classInstanceMethods.set(className, instanceMethods);
      if (staticMethods.size > 0)
        classStaticMethods.set(className, staticMethods);
    }

    // 8. Class constructors
    const classConstructors = new Map<string, string>();
    for (const [className, info] of allClasses) {
      if (info.constructorName) {
        classConstructors.set(className, info.constructorName);
      }
    }

    // 9. Private functions
    const privateFunctions = new Map<string, Set<string>>();
    for (const [dir, entries] of this.registry.privateFilesByDir) {
      privateFunctions.set(dir, new Set(entries.keys()));
    }

    // 10. Private file subfunctions — parse each private file to discover local helpers
    const privateFileSubfunctions = new Map<string, Set<string>>();
    for (const [dir, entries] of this.registry.privateFilesByDir) {
      for (const [funcName, entry] of entries) {
        const ast = this.getCachedAST(entry.fileName);
        const subs = new Set<string>();
        for (const stmt of ast.body) {
          if (stmt.type === "Function" && stmt.name !== funcName) {
            subs.add(stmt.name);
          }
        }
        if (subs.size > 0) {
          privateFileSubfunctions.set(`${dir}${funcName}`, subs);
        }
      }
    }

    // 11. InferiorClasses — build map from superior class → set of inferior classes
    const classInferiorClasses = new Map<string, Set<string>>();
    for (const [className, info] of allClasses) {
      if (info.inferiorClasses.length > 0) {
        classInferiorClasses.set(className, new Set(info.inferiorClasses));
      }
    }

    const index: FunctionIndex = {
      builtins,
      mainFileName: this.mainFileName,
      mainLocalFunctions,
      workspaceFunctions,
      jsUserFunctions,
      workspaceClasses,
      workspaceFileSubfunctions,
      classFileSubfunctions,
      classInstanceMethods,
      classStaticMethods,
      classConstructors,
      privateFunctions,
      privateFileSubfunctions,
      classInferiorClasses,
      searchPaths: this.registry.searchPaths,
      fileToFuncName: this.registry.fileToFuncName,
    };
    this.registry.functionIndex = index;
    return index;
  }

  /** Get the function index (must call buildFunctionIndex() first on any context sharing the same registry). */
  get functionIndex(): FunctionIndex {
    if (!this.registry.functionIndex) {
      throw new Error(
        "FunctionIndex not built yet — call buildFunctionIndex() first"
      );
    }
    return this.registry.functionIndex;
  }
}

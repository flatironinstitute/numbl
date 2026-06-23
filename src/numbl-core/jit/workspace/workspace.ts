/**
 * Workspace + cross-file function dispatch.
 *
 * Thin adapter over numbl's vendored `LoweringContext` (see
 * [../parser/index.ts](../parser/index.ts) for how mtoc2 imports
 * numbl directly via sibling-relative paths). Numbl's
 * `resolveFunction` is the source of truth for "which function does
 * `foo(...)` refer to from this call site"; it implements MATLAB's
 * precedence rules (local > workspace > builtin, plus class-method
 * dispatch on `obj.method(args)` and `ClassName.method(args)`).
 * mtoc2 translates the result back into its own narrow
 * `ResolvedTarget` shape and fences off everything outside v1
 * (private/, +pkg/ namespaces, .numbl.js, imports) with a clean
 * `UnsupportedConstruct`.
 *
 * mtoc2's `MType → ItemType` adapter is intentionally lossy: the
 * resolver inspects `kind === "ClassInstance"` (and only the
 * `className` there); every other shape collapses to `Unknown`.
 * That's all the resolver needs to apply class-method precedence.
 */

import type { AbstractSyntaxTree, Stmt, Span } from "../parser/index.js";
import {
  parseMFile,
  SyntaxError as ParseSyntaxError,
} from "../parser/index.js";
import {
  LoweringContext,
  resolveFunction,
  type CallSite,
  type ItemType,
  type ClassInfo,
} from "../numbl/index.js";

import { UnsupportedConstruct } from "../lowering/errors.js";
import type { Type } from "../lowering/types.js";
import {
  registerClassDef,
  type ClassRegistration,
} from "../lowering/classDefs.js";
import type { Builtin } from "../builtins/registry.js";
import { hashType } from "../lowering/types.js";
import { loadMtoc2UserFunction } from "./mtoc2UserFunctionLoader.js";

/** Slash-only path helpers — workspace file names use forward slashes
 *  everywhere (CLI absolutifies before storing, web IDE is flat). */
function dirnameOf(p: string): string {
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(0, i + 1) : "";
}
function joinPath(dir: string, rel: string): string {
  if (dir === "") return rel;
  return dir.endsWith("/") ? dir + rel : dir + "/" + rel;
}

type FuncStmt = Extract<Stmt, { type: "Function" }>;
type ClassDefStmt = Extract<Stmt, { type: "ClassDef" }>;

export interface WorkspaceFile {
  /** Absolute (or web-IDE-flat) file name used in error attribution
   *  and in numbl's resolver. Numbl uses `fileToFuncName` to derive
   *  bare workspace-function names from this. */
  name: string;
  source: string;
  /** Pre-parsed AST. Set by `Workspace.addFile`; callers populate
   *  it via `parseMFile` before registration. */
  ast?: AbstractSyntaxTree;
}

/** Narrow shape mtoc2 cares about. The resolver may return additional
 *  kinds (private, JS user functions, ...) — those are fenced off
 *  here with `UnsupportedConstruct` so the diagnostic gets a span at
 *  the call site. */
export type ResolvedTarget =
  | {
      kind: "userFunction";
      /** Source-level name (as written at the call site). */
      name: string;
      /** AST of the function definition. */
      ast: FuncStmt;
      /** Source file the function lives in. Used to salt
       *  specialization mangling so two files defining a subfunction
       *  with the same name get distinct C names. */
      file: string;
    }
  | {
      /** Numbl says this name resolves to a builtin. mtoc2 still
       *  validates it against its own builtin registry. */
      kind: "builtin";
      name: string;
    }
  | {
      /** A `.mtoc2.js` user function discovered in the workspace.
       *  The evaluated `Builtin` lives in `Workspace.userBuiltins` —
       *  fetch via `getUserBuiltin(name)` for both lowering and
       *  codegen. The resolved-target carries the source-level name
       *  (which the lowerer also uses as the emitted C call name). */
      kind: "mtoc2UserFunction";
      name: string;
      file: string;
    }
  | {
      /** `Foo(args)` — class constructor call. The class is looked
       *  up in `Workspace.classes`. */
      kind: "classConstructor";
      className: string;
    }
  | {
      /** Class method dispatch — covers `obj.method(args)`,
       *  `method(obj, args)`, and `ClassName.staticMethod(args)`. */
      kind: "classMethod";
      className: string;
      methodName: string;
      /** When true the receiver is NOT passed as a C arg (a static
       *  method called via `ClassName.method(args)` or via instance-
       *  style `obj.staticMethod(args)`). When false the receiver is
       *  the implicit first C arg. */
      stripInstance: boolean;
    };

export class Workspace {
  /** File source + AST, keyed by file name. The AST cache is also
   *  mirrored into the vendored LoweringContext, but this side map
   *  keeps `source` retrievable for diagnostics
   *  (`offsetToLineCol`). */
  readonly files: Map<string, WorkspaceFile> = new Map();
  /** The entry file (active file) — bare-name calls from sibling
   *  files don't see its local functions. */
  readonly mainFile: string;
  /** Search paths used by the vendored resolver to compute relative
   *  paths (and hence workspace-function names). For the CLI,
   *  `[dirname(absoluteEntry)]`. For the web IDE (flat file names),
   *  empty — the resolver treats every name as already-relative. */
  readonly searchPaths: ReadonlyArray<string>;

  /** Numbl resolution context. Holds the workspace registry
   *  (`filesByFuncName`, `classesByName`, `localClassesByName`) and
   *  the `FunctionIndex` used by `resolveFunction`. Settable so the
   *  `fromExistingContext` JIT factory can replace the constructor-
   *  built default with a caller-owned context. */
  ctx: LoweringContext;

  /** Resolved class registry — populated by `finalize()` by walking
   *  every classdef numbl knows about (workspace + local) and
   *  applying mtoc2's validation/property-type inference. */
  classes: Map<string, ClassRegistration> = new Map();

  /** Classes that numbl knows about but mtoc2's `registerClassDef`
   *  rejected (e.g. inheritance, unsupported attributes). Only
   *  populated in `fromExistingContext` (JIT bridge) mode — the
   *  standalone path still fails finalize() on the first bad class.
   *
   *  `isClass(name)` returns `true` for these so call-site dispatch
   *  doesn't get steered into a wrong code path (treating a class
   *  reference as a function call). `requireClass(name)` re-throws
   *  the saved `UnsupportedConstruct` at the actual use site, which
   *  the JIT bridge catches as a decline and falls back to the
   *  interpreter for that call. */
  failedClassValidations: Map<string, UnsupportedConstruct> = new Map();

  /** Workspace-scoped `Builtin` objects loaded from `.mtoc2.js` files.
   *  Keyed by source-level function name (matches numbl's
   *  `mtoc2UserFunctionsByName` keys). Populated lazily by `finalize`. */
  private userBuiltins: Map<string, Builtin> = new Map();

  /** Sibling C/H files pulled in by `.mtoc2.js` user functions via
   *  `exports.cSources`. Each entry carries the originating
   *  `.mtoc2.js` function name (for namespacing in the build dir),
   *  the file's relative path (as the user wrote it), and the file
   *  text. Populated lazily by `finalize`; exposed via
   *  `getUserCSources()` for the build pipeline. */
  private userCSources: Array<{
    ownerFunc: string;
    ownerHash: string;
    relPath: string;
    source: string;
  }> = [];

  private finalized = false;

  /** Set by `fromExistingContext` to indicate that `this.ctx` was
   *  populated by the caller (numbl's interpreter, in the JIT case)
   *  and re-registering its local functions / classes / workspace
   *  files in `finalize()` would duplicate work and possibly error.
   *  The mtoc2-side phases (class registry, `.mtoc2.js` loader) still
   *  run regardless — those are mtoc2-only bookkeeping. */
  private skipCtxRegistration = false;

  constructor(mainFile: string, searchPaths: ReadonlyArray<string> = []) {
    this.mainFile = mainFile;
    this.searchPaths = searchPaths;
    this.ctx = new LoweringContext("", mainFile);
    this.ctx.registry.searchPaths = [...searchPaths];
  }

  /** Build a Workspace that wraps an existing `LoweringContext`. The
   *  caller's `ctx` must already have file ASTs cached, local
   *  functions and classes for the main file registered, workspace
   *  files registered, and the function index built — typically
   *  because a host runtime (e.g. numbl's interpreter) has been
   *  driving the same context. Mtoc2 still runs its own class-registry
   *  + `.mtoc2.js` loader bookkeeping on top.
   *
   *  Used by the numbl JIT bridge so a single `LoweringContext` is
   *  shared between numbl's interpreter and mtoc2's JIT compile path
   *  per session. */
  static fromExistingContext(
    ctx: LoweringContext,
    mainFile: string,
    files: Iterable<WorkspaceFile>
  ): Workspace {
    const ws = new Workspace(mainFile, []);
    ws.ctx = ctx;
    ws.skipCtxRegistration = true;
    for (const f of files) {
      ws.files.set(f.name, f);
    }
    return ws;
  }

  /** Register a file by name.
   *
   *  - `.m` files require a pre-parsed `ast` (from `parseMFile`) so
   *    numbl's resolver and mtoc2's lowerer share one cached AST per
   *    file.
   *  - `.mtoc2.js` files carry source text only; the workspace passes
   *    the source to numbl's registry under the function's basename
   *    and (later) evaluates the JS via `loadMtoc2UserFunctions`. No
   *    AST is required. */
  addFile(file: WorkspaceFile): void {
    if (
      file.name.endsWith(".mtoc2.js") ||
      file.name.endsWith(".c") ||
      file.name.endsWith(".h")
    ) {
      // `.mtoc2.js` source text (read by the user-function loader).
      // `.c` / `.h` are sibling files referenced via `cSources`;
      // they live in `this.files` but never get an AST and never
      // reach numbl's workspace registration.
      this.files.set(file.name, file);
      return;
    }
    if (!file.ast) {
      throw new Error(
        `Workspace.addFile: '${file.name}' must be pre-parsed (ast missing)`
      );
    }
    this.files.set(file.name, file);
    this.ctx.fileASTCache.set(file.name, file.ast);
  }

  /** Build the function index + class registry. Call once after
   *  every file has been added. Subsequent calls are no-ops. */
  finalize(): void {
    if (this.finalized) return;

    if (!this.skipCtxRegistration) {
      // Register top-level functions and classdefs from the MAIN file.
      // These have a different visibility rule (local-to-main, not
      // callable from siblings) than workspace files. Workspace files
      // are registered en masse below — `registerWorkspaceFiles`
      // detects classdef-headed files via a source-text sniff.
      const mainEntry = this.files.get(this.mainFile);
      if (mainEntry?.ast) {
        for (const s of mainEntry.ast.body) {
          if (s.type === "Function") {
            this.ctx.registerLocalFunctionAST(s);
          } else if (s.type === "ClassDef") {
            this.ctx.registerLocalClass(s);
          }
        }
      }

      // Workspace files = everything except the main file. `.c`/`.h`
      // sibling files are kept on `this.files` for the user-function
      // loader to read, but they're not workspace functions and never
      // reach numbl's resolver.
      const wsFiles = [...this.files.values()]
        .filter(f => f.name !== this.mainFile)
        .filter(f => !f.name.endsWith(".c") && !f.name.endsWith(".h"))
        .map(f => ({ name: f.name, source: f.source }));
      this.ctx.registerWorkspaceFiles(wsFiles);
      this.ctx.buildFunctionIndex();
    }

    // Build the mtoc2-shaped class registry from every classdef
    // numbl knows about. We re-walk the parsed AST (numbl's ClassInfo
    // has the AST attached) to apply mtoc2's stricter validation —
    // class attributes / inheritance / events / etc. all reject at
    // this point, before any constructor specialization runs.
    //
    // `@ClassName/<methodName>.m` external method files are already
    // discovered and parsed by numbl during `registerWorkspaceFiles`
    // (they live in `info.externalMethodFiles`, with their ASTs in
    // `ctx.fileASTCache`). We pluck out each file's primary Function
    // statement and feed them into `registerClassDef` so they join
    // the same validation pipeline as in-body methods.
    //
    // When `skipCtxRegistration` is set (JIT bridge mode), a class
    // mtoc2 can't validate is recorded in `failedClassValidations`
    // instead of `classes`. The saved error re-throws at use time
    // via `requireClass(name)` — `isClass(name)` still returns true
    // for these names so name-resolution sites don't take a wrong
    // branch (e.g. treating a class constructor call as a function
    // lookup). The standalone path still aborts here on the first
    // bad class; that's the explicit "fail fast on unsupported
    // construct" contract for a program mtoc2 is being asked to
    // translate end-to-end.
    const registerOrDefer = (
      name: string,
      register: () => ClassRegistration
    ): void => {
      if (this.skipCtxRegistration) {
        try {
          this.classes.set(name, register());
        } catch (e) {
          if (!(e instanceof UnsupportedConstruct)) throw e;
          this.failedClassValidations.set(name, e);
        }
      } else {
        this.classes.set(name, register());
      }
    };

    for (const [name, info] of this.ctx.registry.classesByName) {
      // Old-style (pre-classdef) @folder classes have no classdef AST and run
      // exclusively on the interpreter — never register them with the JIT.
      if (info.isOldStyle) continue;
      registerOrDefer(name, () =>
        registerClassDef(
          info.ast!,
          info.fileName,
          this.collectExternalMethods(info)
        )
      );
    }
    for (const [name, info] of this.ctx.registry.localClassesByName) {
      if (this.classes.has(name) || this.failedClassValidations.has(name)) {
        // A workspace file already registered this name as a class;
        // a local class with the same name is a conflict.
        throw new UnsupportedConstruct(
          `class '${name}' is defined both locally and as a workspace class`,
          info.ast?.span
        );
      }
      registerOrDefer(name, () => registerClassDef(info.ast!, info.fileName));
    }

    // Reject classes that shadow a registered workspace function or
    // a builtin: call-site dispatch routes by name, so disambiguation
    // would be ambiguous.
    const fi = this.ctx.functionIndex;
    for (const [cName, reg] of this.classes) {
      if (fi.builtins.has(cName)) {
        throw new UnsupportedConstruct(
          `class '${cName}' shadows a builtin with the same name`,
          reg.constructor?.span ?? this.spanFromClassFile(reg)
        );
      }
    }

    // Evaluate every `.mtoc2.js` user function numbl discovered. Each
    // file's source runs through `new Function` once here; errors
    // (parse, throw at top-level, missing fields, cBody-eval failure)
    // surface as UnsupportedConstruct at workspace-init time, NOT
    // lazily at first call site — so a broken user file fails fast
    // with a clear file attribution rather than confusingly later.
    //
    // The workspace-relative path is used as the prefix-hash input so
    // two `.mtoc2.js` files with the same function name in different
    // directories (or different packages) get distinct C-namespace
    // prefixes for their private helpers.
    for (const [funcName, entry] of this.ctx.registry
      .mtoc2UserFunctionsByName) {
      const relPath = this.workspaceRelativePath(entry.fileName);
      const fileDir = dirnameOf(entry.fileName);
      const loaded = loadMtoc2UserFunction(
        entry.fileName,
        entry.source,
        funcName,
        relPath,
        siblingRelPath => {
          const f = this.files.get(joinPath(fileDir, siblingRelPath));
          return f?.source;
        }
      );
      this.userBuiltins.set(funcName, loaded.builtin);
      const ownerHash = hashType(relPath);
      for (const cs of loaded.cSources) {
        this.userCSources.push({
          ownerFunc: funcName,
          ownerHash,
          relPath: cs.relPath,
          source: cs.source,
        });
      }
    }

    this.finalized = true;
  }

  /** Compute the workspace-relative path for `absPath`. Used to hash
   *  per-file C-namespace prefixes for `.mtoc2.js` user functions so
   *  the same hash falls out regardless of where the project lives
   *  on disk. Falls back to the bare basename when no search path
   *  contains the file (web IDE, ad-hoc absolute path outside the
   *  entry's directory). */
  private workspaceRelativePath(absPath: string): string {
    let best = "";
    for (const sp of this.searchPaths) {
      const prefix = sp.endsWith("/") ? sp : sp + "/";
      if (absPath.startsWith(prefix) && prefix.length > best.length) {
        best = prefix;
      }
    }
    if (best) return absPath.slice(best.length);
    // No search path matched (e.g. web IDE flat layout). Fall back to
    // the basename — stable enough since names within a workspace
    // are unique anyway.
    const i = absPath.lastIndexOf("/");
    return i >= 0 ? absPath.slice(i + 1) : absPath;
  }

  /** Look up an evaluated `.mtoc2.js` user function by source-level
   *  name. Returns `undefined` if no such workspace user function
   *  exists. Both the lowerer and codegen consult this — the former
   *  to call `transfer`, the latter to call `emit`. */
  getUserBuiltin(name: string): Builtin | undefined {
    this.finalize();
    return this.userBuiltins.get(name);
  }

  /** Sibling C/H files supplied by `.mtoc2.js` user functions via
   *  `exports.cSources`. The build pipeline writes each file to a
   *  per-owner subdirectory under the build root and adds `-I` for
   *  that subdir; `.c` entries are compiled along with the main C,
   *  `.h` entries just sit in the include path. Each entry's
   *  `ownerHash` is the same FNV-1a hash that names the owner's
   *  C-symbol prefix, so per-user-function subdirs stay collision-
   *  free even if two files share a basename. */
  getUserCSources(): ReadonlyArray<{
    ownerFunc: string;
    ownerHash: string;
    relPath: string;
    source: string;
  }> {
    this.finalize();
    return this.userCSources;
  }

  /** Pull the primary Function AST from each `@ClassName/<methodName>.m`
   *  external method file that numbl registered for `info`. Any other
   *  top-level functions in the file are per-method-file local helpers
   *  (numbl scopes them via `withMethodScope`); mtoc2 surfaces them
   *  through the resolve path via `target.source.from === "classFile"`,
   *  so we don't need to collect them here — only the primary needs to
   *  join the class's method registry. Returns `undefined` when the
   *  class has no external methods. */
  private collectExternalMethods(
    info: ClassInfo
  ): Map<string, FuncStmt> | undefined {
    if (info.externalMethodFiles.size === 0) return undefined;
    const out = new Map<string, FuncStmt>();
    for (const [methodName, mf] of info.externalMethodFiles) {
      const ast = this.ctx.fileASTCache.get(mf.fileName);
      if (!ast) {
        throw new UnsupportedConstruct(
          `internal: external method file '${mf.fileName}' for ` +
            `'${info.qualifiedName}.${methodName}' was not parsed`,
          info.ast?.span
        );
      }
      // Dispatch is by file name: the primary method is the file's first
      // top-level function regardless of its internal name. Prefer an exact
      // name match, but fall back to the first function so files whose
      // declared name differs from the file name (a MATLAB-tolerated
      // mismatch) still resolve.
      let primary: FuncStmt | null = null;
      let firstFn: FuncStmt | null = null;
      for (const stmt of ast.body) {
        if (stmt.type !== "Function") continue;
        if (!firstFn) firstFn = stmt;
        if (stmt.name === methodName) {
          primary = stmt;
        }
      }
      primary ??= firstFn;
      if (!primary) {
        throw new UnsupportedConstruct(
          `external method file '${mf.fileName}' has no function`,
          info.ast?.span
        );
      }
      out.set(methodName, primary);
    }
    return out;
  }

  private spanFromClassFile(reg: ClassRegistration): Span {
    // Fallback span for classes without a constructor (no FuncStmt to
    // borrow a span from). Find the ClassDef stmt in the file's AST.
    const fileEntry = this.files.get(reg.file);
    if (fileEntry?.ast) {
      for (const s of fileEntry.ast.body) {
        if (s.type === "ClassDef" && s.name === reg.className) {
          return s.span;
        }
      }
    }
    // Last-ditch: a zero-length span at the start of the file.
    return { file: reg.file, start: 0, end: 0 };
  }

  /** Is `name` a registered class (workspace or local)? Used by the
   *  lowerer to route `Foo(args)` to the constructor path and to
   *  detect `ClassName.staticMethod(args)` against an Ident base. */
  isClass(name: string): boolean {
    return this.classes.has(name) || this.failedClassValidations.has(name);
  }

  /** Like `classes.get(name)`, but re-throws the saved
   *  `UnsupportedConstruct` if the class was registered in
   *  `failedClassValidations` (JIT bridge mode). Use this at sites
   *  that would otherwise do `classes.get(name)!` and dereference
   *  `undefined`. */
  requireClass(name: string): ClassRegistration | undefined {
    const failed = this.failedClassValidations.get(name);
    if (failed) throw failed;
    return this.classes.get(name);
  }

  /** Resolve a call site to a single target. Wraps numbl's
   *  `resolveFunction`, applies mtoc2-narrow validation, and routes
   *  classMethod verdicts (instance + static) through one shape. */
  resolve(
    name: string,
    argTypes: ReadonlyArray<Type>,
    callSite: CallSite,
    span: Span
  ): ResolvedTarget | null {
    this.finalize();
    const itemTypes = argTypes.map(mtypeToItemType);
    // Numbl's resolver only routes a call to a class-file subfunction
    // when callSite.{className,methodName} are set. mtoc2 hasn't
    // threaded those through every call site explicitly, but for an
    // `@ClassName/<methodName>.m` file the values are recoverable
    // from the path. Augment the callSite so external-method-file
    // helpers resolve correctly without touching every caller.
    const augmented = augmentCallSiteFromFile(callSite);
    const target = resolveFunction(
      name,
      itemTypes,
      augmented,
      this.ctx.functionIndex
    );
    if (!target) return null;
    switch (target.kind) {
      case "builtin":
        return { kind: "builtin", name: target.name };
      case "workspaceFunction": {
        const entry = this.ctx.registry.filesByFuncName.get(target.name);
        if (!entry) {
          throw new UnsupportedConstruct(
            `internal: resolver claimed '${target.name}' is a workspace ` +
              `function but no file is registered`,
            span
          );
        }
        const ast = firstFunctionInFile(
          this.ctx.fileASTCache.get(entry.fileName)
        );
        if (!ast) {
          throw new UnsupportedConstruct(
            `'${entry.fileName}' has no function definitions; mtoc2 cannot ` +
              `use it as a workspace function`,
            span
          );
        }
        return { kind: "userFunction", name, ast, file: entry.fileName };
      }
      case "localFunction": {
        if (target.source.from === "main") {
          const ast = findFunctionInBody(
            this.files.get(this.mainFile)?.ast?.body,
            name
          );
          if (!ast) {
            throw new UnsupportedConstruct(
              `internal: resolver claimed '${name}' is a main-file local ` +
                `function but no AST is registered`,
              span
            );
          }
          return { kind: "userFunction", name, ast, file: this.mainFile };
        }
        if (target.source.from === "workspaceFile") {
          const wsName = target.source.wsName;
          const entry = this.ctx.registry.filesByFuncName.get(wsName);
          if (!entry) {
            throw new UnsupportedConstruct(
              `internal: resolver claimed '${name}' is a subfunction of ` +
                `workspace file '${wsName}' but no file is registered`,
              span
            );
          }
          const ast = findFunctionInBody(
            this.ctx.fileASTCache.get(entry.fileName)?.body,
            name
          );
          if (!ast) {
            throw new UnsupportedConstruct(
              `internal: resolver claimed '${name}' is a subfunction of ` +
                `'${entry.fileName}' but no matching Function stmt was found`,
              span
            );
          }
          return { kind: "userFunction", name, ast, file: entry.fileName };
        }
        if (target.source.from === "classFile") {
          const { className, methodScope } = target.source;
          if (methodScope === undefined) {
            // No method-scope context = a call to a class-file
            // subfunction from outside any method. Numbl wouldn't
            // route the call here in that case (its resolver gates
            // on `callSite.methodName`), so reaching here means the
            // call came from inside a class method but mtoc2 didn't
            // recover the method scope. Surface the gap directly.
            throw new UnsupportedConstruct(
              `function '${name}' resolves to a subfunction of class ` +
                `'${className}', but the call site has no method scope ` +
                `(mtoc2 only resolves class-file subfunctions from inside ` +
                `that class's methods)`,
              span
            );
          }
          const classInfo = this.ctx.registry.classesByName.get(className);
          const methodFile = classInfo?.externalMethodFiles.get(methodScope);
          if (methodFile === undefined) {
            // The methodScope must be an external method file: the
            // helper was discovered in that file's body. classdef-body
            // subfunctions (helpers at the top level of `<Class>.m`
            // itself, visible to every method) are a related feature
            // class mtoc2 hasn't wired yet — keep the explicit reject.
            throw new UnsupportedConstruct(
              `function '${name}' resolves to a subfunction of class ` +
                `'${className}', but the owning method scope ` +
                `'${methodScope}' is not an external method file ` +
                `(classdef-body subfunctions are not yet supported by mtoc2)`,
              span
            );
          }
          const ast = findFunctionInBody(
            this.ctx.fileASTCache.get(methodFile.fileName)?.body,
            name
          );
          if (!ast) {
            throw new UnsupportedConstruct(
              `internal: resolver claimed '${name}' is a local helper of ` +
                `'${methodFile.fileName}' but no matching Function stmt ` +
                `was found`,
              span
            );
          }
          return {
            kind: "userFunction",
            name,
            ast,
            file: methodFile.fileName,
          };
        }
        throw new UnsupportedConstruct(
          `function '${name}' resolves to a subfunction of a ` +
            `private file (private/ directories are not yet supported by mtoc2)`,
          span
        );
      }
      case "classMethod":
        return {
          kind: "classMethod",
          className: target.className,
          methodName: target.methodName,
          stripInstance: target.stripInstance,
        };
      case "workspaceClassConstructor":
        return { kind: "classConstructor", className: target.className };
      case "privateFunction":
        throw new UnsupportedConstruct(
          `private functions (under a 'private/' directory) are not yet ` +
            `supported by mtoc2`,
          span
        );
      case "jsUserFunction":
        throw new UnsupportedConstruct(
          `JS user functions (.numbl.js) are not yet supported by mtoc2`,
          span
        );
      case "mtoc2UserFunction": {
        const entry = this.ctx.registry.mtoc2UserFunctionsByName.get(
          target.name
        );
        if (!entry) {
          throw new UnsupportedConstruct(
            `internal: resolver claimed '${target.name}' is a .mtoc2.js ` +
              `user function but no entry is registered`,
            span
          );
        }
        // The evaluated `Builtin` is already in `userBuiltins` (loaded
        // during `finalize`). The lowerer / codegen pull it via
        // `getUserBuiltin`; the resolved-target carries enough for
        // diagnostics.
        return {
          kind: "mtoc2UserFunction",
          name: target.name,
          file: entry.fileName,
        };
      }
      default: {
        const _exhaustive: never = target;
        void _exhaustive;
        throw new UnsupportedConstruct(
          `internal: unhandled resolved-target kind`,
          span
        );
      }
    }
  }

  /** Look up the source text of a file by its name. Used by the
   *  lowerer's `printtype` directive to map a span offset to a
   *  line/column in the right file. */
  sourceOf(file: string): string | undefined {
    return this.files.get(file)?.source;
  }
}

/** Pre-parse a list of source files into `WorkspaceFile`s.
 *
 *  The `mainName` file's parse errors are propagated; for every other
 *  file, parse errors are reported via `onWarn` and the file is
 *  dropped from the result. This matches numbl, which warns and
 *  skips workspace siblings that fail to parse so that a single bad
 *  file in an `addpath`'d tree doesn't poison the whole run.
 *
 *  Files ending in `.mtoc2.js` skip MATLAB parsing — they're plain
 *  JavaScript text that the workspace will hand to the mtoc2 user-
 *  function loader later. Their `ast` field stays undefined. */
export function parseFiles(
  files: ReadonlyArray<{ name: string; source: string }>,
  mainName?: string,
  onWarn: (msg: string) => void = m => console.warn(m)
): WorkspaceFile[] {
  const out: WorkspaceFile[] = [];
  for (const f of files) {
    // `.mtoc2.js` files are JavaScript text — no MATLAB parse.
    // `.c` / `.h` files are sibling files referenced via
    // `exports.cSources`; they pass straight into the workspace
    // map so the user-function loader can read them by relative
    // path, but no parser ever touches them.
    if (
      f.name.endsWith(".mtoc2.js") ||
      f.name.endsWith(".c") ||
      f.name.endsWith(".h")
    ) {
      out.push({ name: f.name, source: f.source });
      continue;
    }
    try {
      out.push({
        name: f.name,
        source: f.source,
        ast: parseMFile(f.source, f.name),
      });
    } catch (e) {
      if (mainName !== undefined && f.name === mainName) throw e;
      if (e instanceof ParseSyntaxError) {
        onWarn(
          `Warning: skipping ${f.name} (syntax error at line ${e.line ?? "?"})`
        );
      } else {
        onWarn(`Warning: skipping ${f.name} (parse error)`);
      }
    }
  }
  return out;
}

/** Adapter from mtoc2's `Type` to numbl's `ItemType`. The resolver
 *  only reads `kind === "ClassInstance"` and `className` from there;
 *  every other shape collapses to `Unknown`. */
export function mtypeToItemType(t: Type): ItemType {
  if (t.kind === "Class")
    return { kind: "ClassInstance", className: t.className };
  return { kind: "Unknown" };
}

function firstFunctionInFile(
  ast: AbstractSyntaxTree | undefined
): FuncStmt | null {
  if (!ast) return null;
  for (const s of ast.body) {
    if (s.type === "Function") return s;
  }
  return null;
}

function findFunctionInBody(
  body: Stmt[] | undefined,
  name: string
): FuncStmt | null {
  if (!body) return null;
  for (const s of body) {
    if (s.type === "Function" && s.name === name) return s;
  }
  return null;
}

/** Recover `{className, methodName}` from a call-site file path of the
 *  form `.../@<ClassName>/<methodName>.m` so numbl's resolver can
 *  route class-file subfunction calls correctly. Returns the
 *  callSite unchanged when:
 *
 *   - the path doesn't match the `@`-folder shape (caller isn't
 *     inside a class method); or
 *   - one of the two fields is already set (caller was explicit and
 *     knows better than the path heuristic).
 */
function augmentCallSiteFromFile(cs: CallSite): CallSite {
  if (cs.className !== undefined && cs.methodName !== undefined) return cs;
  const segs = cs.file.split("/");
  const fileSeg = segs[segs.length - 1];
  const parentSeg = segs[segs.length - 2];
  if (!fileSeg || !parentSeg) return cs;
  if (!fileSeg.endsWith(".m")) return cs;
  if (!parentSeg.startsWith("@")) return cs;
  const derivedClass = parentSeg.slice(1);
  const derivedMethod = fileSeg.slice(0, -2);
  if (!derivedClass || !derivedMethod) return cs;
  return {
    ...cs,
    className: cs.className ?? derivedClass,
    methodName: cs.methodName ?? derivedMethod,
  };
}

// Re-export for callers that need the ClassDef AST type without
// importing it from elsewhere.
export type { ClassDefStmt };

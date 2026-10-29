/**
 * Workspace + cross-file function dispatch.
 *
 * Thin adapter over numbl's `CodeIndex`, shared with the
 * interpreter. Numbl's `resolveFunction` is the source of truth for
 * "which function does `foo(...)` refer to from this call site"; it
 * implements MATLAB's precedence rules (local > workspace > builtin,
 * plus class-method dispatch on `obj.method(args)` and
 * `ClassName.method(args)`). The JIT translates the result back into
 * its own narrow `ResolvedTarget` shape and fences off everything it
 * doesn't support (private/, .numbl.js, ...) with a clean
 * `UnsupportedConstruct`.
 *
 * The `Type → ItemType` adapter is intentionally lossy: the resolver
 * inspects `kind === "ClassInstance"` (and only the `className`
 * there); every other shape collapses to `Unknown`. That's all the
 * resolver needs to apply class-method precedence.
 */

import type { AbstractSyntaxTree, Stmt, Span } from "../numbl/index.js";
import {
  resolveFunction,
  type CodeIndex,
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

type FuncStmt = Extract<Stmt, { type: "Function" }>;

export interface WorkspaceFile {
  /** Absolute (or web-IDE-flat) file name used in error attribution
   *  and in numbl's resolver. Numbl uses `fileToFuncName` to derive
   *  bare workspace-function names from this. */
  name: string;
  source: string;
  /** Pre-parsed AST (shared with numbl's `fileASTCache`). */
  ast?: AbstractSyntaxTree;
}

/** Narrow shape the JIT cares about. The resolver may return additional
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
      /** Numbl says this name resolves to a builtin. The JIT still
       *  validates it against its own builtin registry. */
      kind: "builtin";
      name: string;
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
  /** File source + AST, keyed by file name. This side map keeps
   *  `source` retrievable for diagnostics (`offsetToLineCol`) and the
   *  main file's AST for local-function resolution. */
  readonly files: Map<string, WorkspaceFile> = new Map();
  /** The entry file (active file) — bare-name calls from sibling
   *  files don't see its local functions. */
  readonly mainFile: string;

  /** Numbl's code index, shared with the interpreter. Holds the
   *  workspace files (`filesByFuncName`), the class table and the
   *  `FunctionIndex` used by `resolveFunction`. */
  readonly code: CodeIndex;

  /** Resolved class registry — populated by `finalize()` by walking
   *  every classdef numbl knows about (workspace + local) and
   *  applying the JIT's validation/property-type inference. */
  classes: Map<string, ClassRegistration> = new Map();

  /** Classes that numbl knows about but `registerClassDef` rejected
   *  (e.g. inheritance, unsupported attributes).
   *
   *  `isClass(name)` returns `true` for these so call-site dispatch
   *  doesn't get steered into a wrong code path (treating a class
   *  reference as a function call). `requireClass(name)` re-throws
   *  the saved `UnsupportedConstruct` at the actual use site, which
   *  the executor catches as a decline and falls back to the
   *  interpreter for that call. */
  failedClassValidations: Map<string, UnsupportedConstruct> = new Map();

  private finalized = false;

  private constructor(code: CodeIndex, mainFile: string) {
    this.code = code;
    this.mainFile = mainFile;
  }

  /** Build a Workspace that wraps an existing `CodeIndex`. The
   *  index must already have file ASTs cached, the main file set,
   *  workspace files registered, and the function index built —
   *  because numbl's interpreter has been driving the same index. The
   *  JIT runs its own class-registry bookkeeping on top. */
  static fromCodeIndex(
    code: CodeIndex,
    mainFile: string,
    files: Iterable<WorkspaceFile>
  ): Workspace {
    const ws = new Workspace(code, mainFile);
    for (const f of files) {
      ws.files.set(f.name, f);
    }
    return ws;
  }

  /** Build the class registry. Subsequent calls are no-ops. */
  finalize(): void {
    if (this.finalized) return;

    // Build the JIT class registry from every classdef numbl knows
    // about. We re-walk the parsed AST (numbl's ClassInfo has the AST
    // attached) to apply the JIT's stricter validation — class
    // attributes / inheritance / events / etc. all reject at this
    // point, before any constructor specialization runs.
    //
    // `@ClassName/<methodName>.m` external method files are already
    // discovered and parsed by numbl during `registerWorkspaceFiles`
    // (they live in `info.externalMethodFiles`, with their ASTs in
    // `code.fileASTCache`). We pluck out each file's primary Function
    // statement and feed them into `registerClassDef` so they join
    // the same validation pipeline as in-body methods.
    //
    // A class the JIT can't validate is recorded in
    // `failedClassValidations` instead of `classes`. The saved error
    // re-throws at use time via `requireClass(name)` — `isClass(name)`
    // still returns true for these names so name-resolution sites
    // don't take a wrong branch (e.g. treating a class constructor
    // call as a function lookup).
    const registerOrDefer = (
      name: string,
      register: () => ClassRegistration
    ): void => {
      try {
        this.classes.set(name, register());
      } catch (e) {
        if (!(e instanceof UnsupportedConstruct)) throw e;
        this.failedClassValidations.set(name, e);
      }
    };

    for (const [name, info] of this.code.classes.workspaceClasses) {
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
    for (const [name, info] of this.code.classes.localClasses) {
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
    const fi = this.code.functionIndex;
    for (const [cName, reg] of this.classes) {
      if (fi.builtins.has(cName)) {
        throw new UnsupportedConstruct(
          `class '${cName}' shadows a builtin with the same name`,
          reg.constructor?.span ?? this.spanFromClassFile(reg)
        );
      }
    }

    this.finalized = true;
  }

  /** Pull the primary Function AST from each `@ClassName/<methodName>.m`
   *  external method file that numbl registered for `info`. Any other
   *  top-level functions in the file are per-method-file local helpers;
   *  the JIT surfaces them
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
      const ast = this.code.fileASTCache.get(mf.fileName);
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
   *  `resolveFunction`, applies JIT-narrow validation, and routes
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
    // when callSite.{className,methodName} are set. The JIT hasn't
    // threaded those through every call site explicitly, but for an
    // `@ClassName/<methodName>.m` file the values are recoverable
    // from the path. Augment the callSite so external-method-file
    // helpers resolve correctly without touching every caller.
    const augmented = augmentCallSiteFromFile(callSite);
    const target = resolveFunction(
      name,
      itemTypes,
      augmented,
      this.code.functionIndex
    );
    if (!target) return null;
    switch (target.kind) {
      case "builtin":
        return { kind: "builtin", name: target.name };
      case "workspaceFunction": {
        const entry = this.code.filesByFuncName.get(target.name);
        if (!entry) {
          throw new UnsupportedConstruct(
            `internal: resolver claimed '${target.name}' is a workspace ` +
              `function but no file is registered`,
            span
          );
        }
        const ast = firstFunctionInFile(
          this.code.fileASTCache.get(entry.fileName)
        );
        if (!ast) {
          throw new UnsupportedConstruct(
            `'${entry.fileName}' has no function definitions; the JIT cannot ` +
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
          const entry = this.code.filesByFuncName.get(wsName);
          if (!entry) {
            throw new UnsupportedConstruct(
              `internal: resolver claimed '${name}' is a subfunction of ` +
                `workspace file '${wsName}' but no file is registered`,
              span
            );
          }
          const ast = findFunctionInBody(
            this.code.fileASTCache.get(entry.fileName)?.body,
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
            // call came from inside a class method but the JIT didn't
            // recover the method scope. Surface the gap directly.
            throw new UnsupportedConstruct(
              `function '${name}' resolves to a subfunction of class ` +
                `'${className}', but the call site has no method scope ` +
                `(the JIT only resolves class-file subfunctions from inside ` +
                `that class's methods)`,
              span
            );
          }
          const classInfo = this.code.classes.workspaceClasses.get(className);
          const methodFile = classInfo?.externalMethodFiles.get(methodScope);
          if (methodFile === undefined) {
            // The methodScope must be an external method file: the
            // helper was discovered in that file's body. classdef-body
            // subfunctions (helpers at the top level of `<Class>.m`
            // itself, visible to every method) are a related feature
            // class the JIT hasn't wired yet — keep the explicit reject.
            throw new UnsupportedConstruct(
              `function '${name}' resolves to a subfunction of class ` +
                `'${className}', but the owning method scope ` +
                `'${methodScope}' is not an external method file ` +
                `(classdef-body subfunctions are not yet supported by the JIT)`,
              span
            );
          }
          const ast = findFunctionInBody(
            this.code.fileASTCache.get(methodFile.fileName)?.body,
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
            `private file (private/ directories are not yet supported by the JIT)`,
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
            `supported by the JIT`,
          span
        );
      case "jsUserFunction":
        throw new UnsupportedConstruct(
          `JS user functions (.numbl.js) are not yet supported by the JIT`,
          span
        );
      case "builtinMethod":
        throw new UnsupportedConstruct(
          `methods of TypeScript-implemented classes are not supported by the JIT`,
          span
        );
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

/** Adapter from the JIT's `Type` to numbl's `ItemType`. The resolver
 *  only reads `kind === "ClassInstance"` and `className` from there;
 *  every other shape collapses to `Unknown`. */
function mtypeToItemType(t: Type): ItemType {
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

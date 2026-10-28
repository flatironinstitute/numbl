/**
 * The function index: a complete, upfront map from function names to where
 * they are defined. Built from a `CodeIndex` once its files are registered
 * (and rebuilt after addpath/rmpath/cd); `resolveFunction` uses it for
 * definitive, single-target resolution.
 */

import type { Stmt } from "../parser/index.js";
import { getAllBuiltinNames } from "../helpers/registry.js";
import { getAllIBuiltinNames } from "../interpreter/builtins/index.js";
import type { CodeIndex } from "./codeIndex.js";

export type ImportEntry =
  | {
      wildcard: false;
      qualifiedName: string;
      shortName: string;
      staticMethod?: { className: string; methodName: string };
    }
  | { wildcard: true; namespace: string };

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

  /** Per-file import entries: fileName → ImportEntry[] */
  fileImports: Map<string, ImportEntry[]>;
}

/** Build the function index from the current state of `code`. Every
 *  registered file's AST must already be in `code.fileASTCache`. */
export function buildFunctionIndex(code: CodeIndex): FunctionIndex {
  const classes = code.classes;

  // 1. Builtins
  const builtins = new Set([...getAllBuiltinNames(), ...getAllIBuiltinNames()]);

  // JS user functions resolve at workspace priority (after .m, before
  // builtins).
  const jsUserFunctions = new Set<string>(code.jsUserFunctionsByName.keys());

  // 2. Main script local functions
  const mainLocalFunctions = new Set(code.mainLocalFunctions.keys());

  // 3. Primary workspace functions
  const workspaceFunctions = new Set(code.filesByFuncName.keys());

  // 4. All classes (workspace + local)
  const workspaceClasses = new Set([
    ...classes.workspaceClasses.keys(),
    ...classes.localClasses.keys(),
  ]);

  // 5. Workspace file subfunctions — parse each file to discover function names
  const workspaceFileSubfunctions = new Map<string, Set<string>>();
  for (const [funcName, entry] of code.filesByFuncName) {
    const ast = code.getCachedAST(entry.fileName);
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
  for (const [className, info] of classes.workspaceClasses) {
    const subs = new Set<string>();

    // Look up the class file to find top-level functions (non-method helpers)
    const classAst = code.getCachedAST(info.fileName);
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
      const methodAst = code.getCachedAST(mf.fileName);
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
  for (const [className, info] of classes.localClasses) {
    const subs = new Set<string>();
    const classAst = code.getCachedAST(info.fileName);
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
  const allClasses = [...classes.entries()];
  for (const [className] of allClasses) {
    const instanceMethods = new Set<string>();
    const staticMethods = new Set<string>();
    const self = classes.get(className);
    for (const info of self ? classes.lineage(self) : []) {
      for (const m of info.methodNames) instanceMethods.add(m);
      for (const m of info.staticMethodNames) staticMethods.add(m);
      // Include external method files. For old-style @folder classes the
      // constructor is also an external file (@Name/Name.m); exclude it so
      // `Name(x)` resolves as construction, not a method call.
      for (const m of info.externalMethodFiles.keys()) {
        if (info.isOldStyle && m === info.constructorName) continue;
        if (!info.staticMethodNames.has(m)) {
          instanceMethods.add(m);
        }
      }
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
  for (const [dir, entries] of code.privateFilesByDir) {
    privateFunctions.set(dir, new Set(entries.keys()));
  }

  // 10. Private file subfunctions — parse each private file to discover local helpers
  const privateFileSubfunctions = new Map<string, Set<string>>();
  for (const [dir, entries] of code.privateFilesByDir) {
    for (const [funcName, entry] of entries) {
      const ast = code.getCachedAST(entry.fileName);
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

  // 12. File imports — collect import statements from all file ASTs.
  // `import` is function-scoped in MATLAB, so a function file's imports live
  // inside the function body (and may appear inside nested blocks). numbl
  // tracks imports at file granularity, so recurse through nested statement
  // bodies and merge every import into the file's entry list.
  const fileImports = new Map<string, ImportEntry[]>();
  const gatherImports = (
    body: Stmt[],
    out: (Stmt & { type: "Import" })[]
  ): void => {
    for (const stmt of body) {
      switch (stmt.type) {
        case "Import":
          out.push(stmt);
          break;
        case "Function":
        case "While":
        case "For":
          gatherImports(stmt.body, out);
          break;
        case "If":
          gatherImports(stmt.thenBody, out);
          for (const b of stmt.elseifBlocks) gatherImports(b.body, out);
          if (stmt.elseBody) gatherImports(stmt.elseBody, out);
          break;
        case "Switch":
          for (const c of stmt.cases) gatherImports(c.body, out);
          if (stmt.otherwise) gatherImports(stmt.otherwise, out);
          break;
        case "TryCatch":
          gatherImports(stmt.tryBody, out);
          gatherImports(stmt.catchBody, out);
          break;
        case "ClassDef":
          // Methods (incl. the constructor) carry function-scoped imports.
          for (const member of stmt.members) {
            if (member.type === "Methods") gatherImports(member.body, out);
          }
          break;
      }
    }
  };
  const collectImportsFromBody = (body: Stmt[], fileName: string): void => {
    const entries: ImportEntry[] = [];
    const importStmts: (Stmt & { type: "Import" })[] = [];
    gatherImports(body, importStmts);
    for (const stmt of importStmts) {
      if (stmt.wildcard) {
        entries.push({ wildcard: true, namespace: stmt.path.join(".") });
      } else {
        const qualifiedName = stmt.path.join(".");
        const shortName = stmt.path[stmt.path.length - 1];
        // Check if this is a static method import (e.g. import pkg.MyClass.method)
        // by seeing if the prefix is a known class with a matching static method.
        const prefix = stmt.path.slice(0, -1).join(".");
        const isStaticMethod =
          prefix.length > 0 && classStaticMethods.get(prefix)?.has(shortName);
        entries.push({
          wildcard: false,
          qualifiedName,
          shortName,
          ...(isStaticMethod
            ? { staticMethod: { className: prefix, methodName: shortName } }
            : {}),
        });
      }
    }
    if (entries.length > 0) {
      fileImports.set(fileName, entries);
    }
  };

  // Main script
  const mainAst = code.fileASTCache.get(code.mainFileName);
  if (mainAst) collectImportsFromBody(mainAst.body, code.mainFileName);
  // Workspace files
  for (const [, entry] of code.filesByFuncName) {
    const ast = code.getCachedAST(entry.fileName);
    collectImportsFromBody(ast.body, entry.fileName);
  }
  // Class files
  for (const [, info] of classes.workspaceClasses) {
    const ast = code.getCachedAST(info.fileName);
    collectImportsFromBody(ast.body, info.fileName);
  }
  // Private files
  for (const [, entries] of code.privateFilesByDir) {
    for (const [, entry] of entries) {
      const ast = code.getCachedAST(entry.fileName);
      collectImportsFromBody(ast.body, entry.fileName);
    }
  }

  return {
    builtins,
    mainFileName: code.mainFileName,
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
    searchPaths: code.searchPaths,
    fileToFuncName: code.fileToFuncName,
    fileImports,
  };
}

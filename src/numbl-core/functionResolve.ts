/**
 * Unified function resolution.
 *
 * Single source of truth for resolving a function name + arg types + call site
 * to a definitive target. Used by both codegen (compile-time) and JIT (runtime).
 */

import type { FunctionIndex } from "./lowering/loweringContext.js";
import type { CallSite } from "./runtime/runtimeHelpers.js";
import type { ItemType } from "./lowering/itemTypes.js";

// ── Resolution target types ──────────────────────────────────────────
// Describes WHAT function to compile, determined by the call site context.

export type ResolvedTarget =
  | {
      kind: "classMethod";
      className: string;
      methodName: string;
      compileArgTypes: ItemType[];
      stripInstance: boolean;
    }
  | {
      kind: "localFunction";
      name: string;
      argTypes: ItemType[];
      source:
        | { from: "main" }
        | { from: "classFile"; className: string; methodScope?: string }
        | { from: "workspaceFile"; wsName: string }
        | { from: "privateFile"; callerFile: string };
    }
  | {
      kind: "privateFunction";
      name: string;
      argTypes: ItemType[];
      callerFile: string; // needed to look up the private file from the registry
    }
  | {
      kind: "workspaceFunction";
      name: string;
      argTypes: ItemType[];
    }
  | {
      kind: "workspaceClassConstructor";
      className: string;
      argTypes: ItemType[];
    }
  | {
      kind: "builtin";
      name: string;
    };

/** Compute the relative path from search paths (longest match wins). */
function getRelativePathFromSearchPaths(
  absolutePath: string,
  searchPaths: string[]
): string {
  if (searchPaths.length === 0) return absolutePath;
  let bestPrefix = "";
  for (const sp of searchPaths) {
    const prefix = sp.endsWith("/") ? sp : sp + "/";
    if (absolutePath.startsWith(prefix) && prefix.length > bestPrefix.length) {
      bestPrefix = prefix;
    }
  }
  if (bestPrefix) return absolutePath.slice(bestPrefix.length);
  return absolutePath;
}

/** Compute the effective parent directory for private function scoping. */
function getEffectiveDir(fileName: string, searchPaths: string[]): string {
  const relativePath = getRelativePathFromSearchPaths(fileName, searchPaths);
  const parts = relativePath.split("/");
  parts.pop(); // remove filename
  if (parts.length > 0 && parts[parts.length - 1] === "private") {
    parts.pop();
  }
  return parts.length > 0 ? parts.join("/") + "/" : "";
}

/**
 * Try to resolve a name via import entries for the calling file.
 * @param wildcardOnly - if true, only check wildcard imports; if false, only explicit.
 */
function resolveViaImports(
  name: string,
  argTypes: ItemType[],
  callSite: CallSite,
  index: FunctionIndex,
  wildcardOnly: boolean
): ResolvedTarget | null {
  const imports = index.fileImports.get(callSite.file);
  if (!imports) return null;

  for (const imp of imports) {
    if (imp.wildcard !== wildcardOnly) continue;

    if (imp.wildcard) {
      // import pkg.* → try "pkg.<name>"
      const candidateName = `${imp.namespace}.${name}`;
      if (index.workspaceFunctions.has(candidateName)) {
        return { kind: "workspaceFunction", name: candidateName, argTypes };
      }
      if (index.workspaceClasses.has(candidateName)) {
        return {
          kind: "workspaceClassConstructor",
          className: candidateName,
          argTypes,
        };
      }
      // Check if namespace is a class and name is a static method
      if (index.classStaticMethods.get(imp.namespace)?.has(name)) {
        return {
          kind: "classMethod",
          className: imp.namespace,
          methodName: name,
          compileArgTypes: argTypes,
          stripInstance: false,
        };
      }
    } else {
      // Explicit: import pkg.foo → only matches if name === shortName
      if (name !== imp.shortName) continue;

      if (imp.staticMethod) {
        return {
          kind: "classMethod",
          className: imp.staticMethod.className,
          methodName: imp.staticMethod.methodName,
          compileArgTypes: argTypes,
          stripInstance: false,
        };
      }
      if (index.workspaceFunctions.has(imp.qualifiedName)) {
        return {
          kind: "workspaceFunction",
          name: imp.qualifiedName,
          argTypes,
        };
      }
      if (index.workspaceClasses.has(imp.qualifiedName)) {
        return {
          kind: "workspaceClassConstructor",
          className: imp.qualifiedName,
          argTypes,
        };
      }
    }
  }
  return null;
}

/**
 * Definitively resolve a function name to a single target using the FunctionIndex.
 * Returns null only if the function is truly not found.
 *
 * @param name - Function name to resolve
 * @param argTypes - Argument types (may include undefined for unknown)
 * @param callSite - Where the call originates
 * @param index - The upfront function index
 */
export function resolveFunction(
  name: string,
  argTypes: ItemType[],
  callSite: CallSite,
  index: FunctionIndex
): ResolvedTarget | null {
  // 0. If targetClassName is set, resolve the class method directly
  if (callSite.targetClassName) {
    const className = callSite.targetClassName;
    // Check if the method is static — if so, strip the instance from argTypes
    // because methodDispatch passes [instance, ...actualArgs] but static
    // methods don't take the instance as a parameter.
    const isStatic =
      index.classStaticMethods.get(className)?.has(name) ?? false;
    const stripInstance =
      isStatic && argTypes.length > 0 && argTypes[0]?.kind === "ClassInstance";
    const compileArgTypes = stripInstance ? argTypes.slice(1) : argTypes;
    return {
      kind: "classMethod",
      className,
      methodName: name,
      compileArgTypes,
      stripInstance,
    };
  }

  // 0.5. Explicit (non-wildcard) imports — higher priority than local functions
  {
    const imported = resolveViaImports(name, argTypes, callSite, index, false);
    if (imported) return imported;
  }

  // 1. Local functions in the calling file
  if (callSite.className) {
    // Inside a class method — try local helpers in the class file,
    // but skip if name is a known class method (prevents e.g.
    // subsref(obj.data, S) inside MyClass.subsref from re-compiling
    // the class method as a local function with wrong arg types).
    const className = callSite.className;
    const isClassMethod =
      index.classInstanceMethods.get(className)?.has(name) ||
      index.classConstructors.get(className) === name ||
      index.classStaticMethods.get(className)?.has(name);
    if (
      !isClassMethod &&
      index.classFileSubfunctions.get(className)?.has(name)
    ) {
      return {
        kind: "localFunction",
        name,
        argTypes,
        source: {
          from: "classFile",
          className,
          methodScope: callSite.methodName,
        },
      };
    }
  } else {
    // Main script local functions — only visible from within the main script itself.
    // Local functions are file-scoped, so workspace functions must not see them.
    if (
      callSite.file === index.mainFileName &&
      index.mainLocalFunctions.has(name)
    ) {
      return {
        kind: "localFunction",
        name,
        argTypes,
        source: { from: "main" },
      };
    }

    // If the call originates from a workspace function file, check its subfunctions
    if (
      callSite.file &&
      callSite.file.endsWith(".m") &&
      !callSite.file.startsWith("@")
    ) {
      const wsName =
        index.fileToFuncName.get(callSite.file) ??
        callSite.file.replace(/\.m$/, "");
      if (index.workspaceFileSubfunctions.get(wsName)?.has(name)) {
        return {
          kind: "localFunction",
          name,
          argTypes,
          source: { from: "workspaceFile", wsName },
        };
      }
    }

    // If the call originates from a private function file, check its subfunctions
    if (callSite.file && callSite.file.includes("/private/")) {
      const dir = getEffectiveDir(callSite.file, index.searchPaths);
      const callerFunc = callSite.file.replace(/\.m$/, "").split("/").pop()!;
      if (
        index.privateFileSubfunctions?.get(`${dir}${callerFunc}`)?.has(name)
      ) {
        return {
          kind: "localFunction",
          name,
          argTypes,
          source: { from: "privateFile", callerFile: callSite.file },
        };
      }
    }
  }

  // 1.5. Wildcard imports — after local functions, before private functions
  {
    const imported = resolveViaImports(name, argTypes, callSite, index, true);
    if (imported) return imported;
  }

  // 2. Private functions (scoped to caller's directory)
  //    Precedence: private functions before object functions.
  if (callSite.file) {
    const dir = getEffectiveDir(callSite.file, index.searchPaths);
    if (index.privateFunctions.get(dir)?.has(name)) {
      return {
        kind: "privateFunction",
        name,
        argTypes,
        callerFile: callSite.file,
      };
    }
  }

  // 3. Class method (if any arg is a class instance)
  //    Collect all candidate classes that have the method, then pick the
  //    most dominant one using InferiorClasses relationships.
  {
    const candidates: string[] = [];
    for (const argType of argTypes) {
      if (argType?.kind === "ClassInstance") {
        const className = argType.className;
        if (
          !candidates.includes(className) &&
          (index.classInstanceMethods.get(className)?.has(name) ||
            index.classStaticMethods.get(className)?.has(name))
        ) {
          candidates.push(className);
        }
      }
    }
    if (candidates.length === 1) {
      return {
        kind: "classMethod",
        className: candidates[0],
        methodName: name,
        compileArgTypes: argTypes,
        stripInstance: false,
      };
    }
    if (candidates.length > 1) {
      // Pick the superior class: a class that declares others as inferior wins.
      // If class A lists class B in its InferiorClasses, A is superior to B.
      let winner = candidates[0];
      for (let i = 1; i < candidates.length; i++) {
        const c = candidates[i];
        // If c declares winner as inferior, c wins
        if (index.classInferiorClasses.get(c)?.has(winner)) {
          winner = c;
        }
        // If winner declares c as inferior, winner stays (no change needed)
      }
      return {
        kind: "classMethod",
        className: winner,
        methodName: name,
        compileArgTypes: argTypes,
        stripInstance: false,
      };
    }
  }

  // 4. Workspace function
  if (index.workspaceFunctions.has(name)) {
    return {
      kind: "workspaceFunction",
      name,
      argTypes,
    };
  }

  // 4b. JS user function (same priority tier as workspace, but .m wins)
  if (index.jsUserFunctions.has(name)) {
    return { kind: "builtin", name };
  }

  // 5. Workspace class constructor
  if (index.workspaceClasses.has(name)) {
    return {
      kind: "workspaceClassConstructor",
      className: name,
      argTypes,
    };
  }

  // 6. Builtin
  if (index.builtins.has(name)) {
    return { kind: "builtin", name };
  }

  return null;
}

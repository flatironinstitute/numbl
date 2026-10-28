/**
 * Function-index tier of name resolution.
 *
 * Resolves a function name + argument types + call site to a single target
 * using the `FunctionIndex`: imports, local functions of the calling file,
 * private functions, class methods (by object argument), workspace and
 * `.numbl.js` functions, class constructors and builtins. `resolveName`
 * (resolveName.ts) layers variables, nested functions and constants on
 * top. The JIT calls this directly.
 */

import type { FunctionIndex } from "../workspace/functionIndex.js";
import { callerPrivateScope } from "../workspace/searchPaths.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import type { ItemType } from "../itemTypes.js";
import {
  type RuntimeValue,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
} from "../runtime/types.js";

// ── Resolution target types ──────────────────────────────────────────
// Describes WHAT function to compile, determined by the call site context.

export type ResolvedTarget =
  | {
      kind: "classMethod";
      className: string;
      methodName: string;
      stripInstance: boolean;
    }
  | {
      kind: "localFunction";
      name: string;
      source:
        | { from: "main" }
        | { from: "classFile"; className: string; methodScope?: string }
        | { from: "workspaceFile"; wsName: string }
        | { from: "privateFile"; callerFile: string };
    }
  | {
      kind: "privateFunction";
      name: string;
      callerFile: string; // needed to look up the private file from the registry
    }
  | {
      kind: "workspaceFunction";
      name: string;
    }
  | {
      kind: "jsUserFunction";
      name: string;
    }
  | {
      kind: "workspaceClassConstructor";
      className: string;
    }
  | {
      kind: "builtin";
      name: string;
    };

// ── Resolution memoization ───────────────────────────────────────────
// `resolveFunction` is hot: it runs on every interpreter call, redoing
// the same Map lookups + path-splitting (`callerPrivateScope`) for the same
// call site every time. The decision depends only on `(name, callSite,
// and the class-names of any ClassInstance args)` — for the common
// numeric call it depends only on `(name, callSite)`. We cache the
// resolved target per FunctionIndex (a WeakMap, so a rebuilt index gets
// a fresh cache and there's no manual invalidation: the index's maps are
// never mutated in place). On a hit we re-inject the *live* argTypes,
// since downstream specialization keys on them.
const resolveCaches = new WeakMap<
  FunctionIndex,
  Map<string, ResolvedTarget | null>
>();

function getResolveCache(
  index: FunctionIndex
): Map<string, ResolvedTarget | null> {
  let c = resolveCaches.get(index);
  if (c === undefined) {
    c = new Map();
    resolveCaches.set(index, c);
  }
  return c;
}

/** Build a cache key. Only ClassInstance args (by position + class name)
 *  affect resolution; everything else is interchangeable, so non-class
 *  calls collapse to a single signature (`"-"`) and skip string-building
 *  on the hot path. */
function makeResolveKey(
  name: string,
  callSite: CallSite,
  argTypes: ItemType[]
): string {
  let sig = "-";
  for (let i = 0; i < argTypes.length; i++) {
    const t = argTypes[i];
    if (t !== undefined && t.kind === "ClassInstance") {
      if (sig === "-") sig = "";
      sig += i + ":" + t.className + ";";
    }
  }
  return (
    callSite.file +
    "\u0000" +
    (callSite.className ?? "") +
    "\u0000" +
    (callSite.methodName ?? "") +
    "\u0000" +
    (callSite.targetClassName ?? "") +
    "\u0000" +
    name +
    "\u0000" +
    sig
  );
}

const UNKNOWN_ARG: ItemType = { kind: "Unknown" };

/** The argument type that resolution needs: only an object's class affects
 *  which function a name resolves to. */
export function resolutionArgType(v: unknown): ItemType {
  if (
    typeof v === "object" &&
    v !== null &&
    (isRuntimeClassInstance(v as RuntimeValue) ||
      isRuntimeClassInstanceArray(v as RuntimeValue))
  ) {
    return {
      kind: "ClassInstance",
      className: (v as { className: string }).className,
    };
  }
  return UNKNOWN_ARG;
}

/**
 * Try to resolve a name via import entries for the calling file.
 * @param wildcardOnly - if true, only check wildcard imports; if false, only explicit.
 */
function resolveViaImports(
  name: string,
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
        return { kind: "workspaceFunction", name: candidateName };
      }
      if (index.workspaceClasses.has(candidateName)) {
        return {
          kind: "workspaceClassConstructor",
          className: candidateName,
        };
      }
      // Check if namespace is a class and name is a static method
      if (index.classStaticMethods.get(imp.namespace)?.has(name)) {
        return {
          kind: "classMethod",
          className: imp.namespace,
          methodName: name,
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
          stripInstance: false,
        };
      }
      if (index.workspaceFunctions.has(imp.qualifiedName)) {
        return {
          kind: "workspaceFunction",
          name: imp.qualifiedName,
        };
      }
      if (index.workspaceClasses.has(imp.qualifiedName)) {
        return {
          kind: "workspaceClassConstructor",
          className: imp.qualifiedName,
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
  const cache = getResolveCache(index);
  const key = makeResolveKey(name, callSite, argTypes);
  // Stored values are `ResolvedTarget | null` (a cached miss), never
  // `undefined`, so `undefined` unambiguously means "not yet cached".
  const cached = cache.get(key);
  if (cached !== undefined) return cached;
  const result = resolveFunctionImpl(name, argTypes, callSite, index);
  cache.set(key, result);
  return result;
}

function resolveFunctionImpl(
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
    return {
      kind: "classMethod",
      className,
      methodName: name,
      stripInstance,
    };
  }

  // 0.5. Explicit (non-wildcard) imports — higher priority than local functions
  {
    const imported = resolveViaImports(name, callSite, index, false);
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
          source: { from: "workspaceFile", wsName },
        };
      }
    }

    // If the call originates from a private function file, check its subfunctions
    if (callSite.file && callSite.file.includes("/private/")) {
      const dir = callerPrivateScope(callSite.file, index.searchPaths);
      const callerFunc = callSite.file.replace(/\.m$/, "").split("/").pop()!;
      if (
        index.privateFileSubfunctions?.get(`${dir}${callerFunc}`)?.has(name)
      ) {
        return {
          kind: "localFunction",
          name,
          source: { from: "privateFile", callerFile: callSite.file },
        };
      }
    }
  }

  // 1.5. Wildcard imports — after local functions, before private functions
  {
    const imported = resolveViaImports(name, callSite, index, true);
    if (imported) return imported;
  }

  // 2. Private functions (scoped to caller's directory)
  //    Precedence: private functions before object functions.
  if (callSite.file) {
    const dir = callerPrivateScope(callSite.file, index.searchPaths);
    if (index.privateFunctions.get(dir)?.has(name)) {
      return {
        kind: "privateFunction",
        name,
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
        // `Name(...)` where Name is the class's own constructor is always
        // construction, never method dispatch — even when an argument is an
        // instance of that class (e.g. SE3(anotherSE3) copies via the
        // constructor). Let it fall through to the constructor resolution below.
        if (index.classConstructors.get(className) === name) continue;
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
        stripInstance: false,
      };
    }
  }

  // 4. Workspace function
  if (index.workspaceFunctions.has(name)) {
    return {
      kind: "workspaceFunction",
      name,
    };
  }

  // 4b. JS user function (.numbl.js) — same priority tier as workspace,
  // but .m wins because it's checked first.
  if (index.jsUserFunctions.has(name)) {
    return { kind: "jsUserFunction", name };
  }

  // 5. Workspace class constructor
  if (index.workspaceClasses.has(name)) {
    return {
      kind: "workspaceClassConstructor",
      className: name,
    };
  }

  // 6. Builtin
  if (index.builtins.has(name)) {
    return { kind: "builtin", name };
  }

  return null;
}

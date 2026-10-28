/**
 * Name resolution: what a name means in a scope.
 *
 * `resolveName` answers for a bare name, in MATLAB's precedence order:
 *
 *   variable > nested function > explicit import > local function (same
 *   file) > wildcard import > private function > class method (object
 *   argument) > workspace / `.numbl.js` function > class constructor >
 *   constant / interpreter special / builtin
 *
 * The function tiers are `resolveFunction`'s. A workspace `.m` (including
 * the bundled stdlib) shadows a builtin, and `.m` wins over `.numbl.js`.
 * `resolveQualified` answers for a dotted name (`pkg.fn`, `pkg.Class`,
 * `Class.staticMethod`, `Class.EnumMember`, `Class.empty`, nested packages),
 * honoring the scope's imports.
 */

import { getConstant } from "../helpers/constants.js";
import { getInterpreterSpecialBuiltin } from "../interpreter/interpreterSpecialBuiltins.js";
import type { Environment, FunctionDef } from "../interpreter/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import type { ImportEntry } from "../workspace/functionIndex.js";
import {
  resolveFunction,
  resolutionArgType,
  type ResolvedTarget,
} from "./resolveFunction.js";
import type { Scope } from "./scope.js";
import { functionBodyFor } from "./functionBody.js";

type Target<K extends ResolvedTarget["kind"]> = Extract<
  ResolvedTarget,
  { kind: K }
>;

/** What a name resolves to. Kinds carrying a `target` are callable through
 *  the interpreter's `interpretTarget`. */
export type NameResolution =
  | { kind: "variable"; name: string }
  /** A builtin constant such as `pi` or `true` (no arguments). */
  | { kind: "constant"; name: string; value: RuntimeValue }
  | { kind: "nested"; name: string; fn: FunctionDef; env: Environment }
  /** An interpreter special (eval, exist, ...): a builtin that needs the
   *  caller's scope. `target` is the registered builtin it falls back to. */
  | { kind: "special"; name: string; target: Target<"builtin"> | null }
  | { kind: "builtin"; name: string; target: Target<"builtin"> }
  /** A local function of the main file. */
  | { kind: "mainLocal"; name: string; target: Target<"localFunction"> }
  /** A local function of a workspace, class or private file. */
  | { kind: "fileLocal"; name: string; target: Target<"localFunction"> }
  | { kind: "private"; name: string; target: Target<"privateFunction"> }
  /** A method selected by an object argument. */
  | { kind: "classMethod"; name: string; target: Target<"classMethod"> }
  | { kind: "staticMethod"; name: string; target: Target<"classMethod"> }
  | {
      kind: "classConstructor";
      name: string;
      target: Target<"workspaceClassConstructor">;
    }
  | {
      kind: "packageFunction";
      name: string;
      target: Target<"workspaceFunction">;
    }
  | { kind: "jsUserFunction"; name: string; target: Target<"jsUserFunction"> }
  | {
      kind: "workspaceFunction";
      name: string;
      target: Target<"workspaceFunction">;
    }
  /** `Class.Member` for an enumeration member. */
  | { kind: "enumMember"; name: string; className: string; memberName: string }
  /** `Class.name` where `name` is not a static method or enumeration member
   *  (an instance method, a constant property, ...). */
  | { kind: "classMember"; name: string; target: Target<"classMethod"> }
  | { kind: "notFound"; name: string };

export interface ResolveOptions {
  /** The call's arguments. Object arguments select class methods, and a
   *  call without arguments may resolve to a constant. Default: none. */
  args?: readonly unknown[];
  /** Whether variables in scope count (default true). */
  variables?: boolean;
  /** Whether nested functions count (default true). MATLAB's string-named
   *  calls (`feval('f', ...)`) do not see them. */
  nested?: boolean;
}

const NO_ARGS: readonly unknown[] = [];

/** Resolve `name` in `scope`. A dotted name goes to `resolveQualified`. */
export function resolveName(
  name: string,
  scope: Scope,
  opts: ResolveOptions = {}
): NameResolution {
  if (name.includes(".")) return resolveQualified(name.split("."), scope, opts);
  const env = scope.env;
  if (env) {
    if (opts.variables !== false && env.has(name))
      return { kind: "variable", name };
    if (opts.nested !== false) {
      const nested = env.getNestedFunction(name);
      if (nested)
        return { kind: "nested", name, fn: nested.fn, env: nested.env };
    }
  }
  const args = opts.args ?? NO_ARGS;
  const target = resolveFunction(
    name,
    args.map(resolutionArgType),
    scope,
    scope.index
  );
  return classifyTarget(name, target, scope, args.length === 0);
}

function classifyTarget(
  name: string,
  target: ResolvedTarget | null,
  scope: Scope,
  noArgs: boolean
): NameResolution {
  if (target === null || target.kind === "builtin") {
    if (noArgs) {
      const value = getConstant(name);
      if (value !== undefined) return { kind: "constant", name, value };
    }
    if (getInterpreterSpecialBuiltin(name))
      return { kind: "special", name, target };
    return target
      ? { kind: "builtin", name, target }
      : { kind: "notFound", name };
  }
  switch (target.kind) {
    case "localFunction":
      return target.source.from === "main"
        ? { kind: "mainLocal", name, target }
        : { kind: "fileLocal", name, target };
    case "privateFunction":
      return { kind: "private", name, target };
    case "classMethod":
      return scope.index.classStaticMethods
        .get(target.className)
        ?.has(target.methodName)
        ? { kind: "staticMethod", name, target }
        : { kind: "classMethod", name, target };
    case "workspaceClassConstructor":
      return { kind: "classConstructor", name, target };
    case "workspaceFunction":
      return target.name.includes(".")
        ? { kind: "packageFunction", name, target }
        : { kind: "workspaceFunction", name, target };
    case "jsUserFunction":
      return { kind: "jsUserFunction", name, target };
  }
}

/** Resolve a dotted name given as its parts. The name is tried as imported
 *  by an explicit import of its first part, then under each wildcard-import
 *  namespace, then as written. `opts.variables` (default true): a variable
 *  named like the first part makes the whole name a variable access. */
export function resolveQualified(
  parts: readonly string[],
  scope: Scope,
  opts: Pick<ResolveOptions, "variables"> = {}
): NameResolution {
  const name = parts.join(".");
  if (opts.variables !== false && scope.env?.has(parts[0]))
    return { kind: "variable", name };
  for (const candidate of qualifiedCandidates(parts, scope.imports)) {
    const r = resolveQualifiedAs(candidate, name, scope);
    if (r) return r;
  }
  return { kind: "notFound", name };
}

function* qualifiedCandidates(
  parts: readonly string[],
  imports: readonly ImportEntry[]
): Generator<readonly string[]> {
  for (const imp of imports) {
    if (!imp.wildcard && !imp.staticMethod && imp.shortName === parts[0])
      yield [...imp.qualifiedName.split("."), ...parts.slice(1)];
  }
  for (const imp of imports) {
    if (imp.wildcard) yield [...imp.namespace.split("."), ...parts];
  }
  yield parts;
}

function resolveQualifiedAs(
  parts: readonly string[],
  name: string,
  scope: Scope
): NameResolution | null {
  const index = scope.index;
  const full = parts.join(".");
  if (index.workspaceFunctions.has(full))
    return {
      kind: "packageFunction",
      name,
      target: { kind: "workspaceFunction", name: full },
    };
  if (index.workspaceClasses.has(full))
    return {
      kind: "classConstructor",
      name,
      target: { kind: "workspaceClassConstructor", className: full },
    };
  if (parts.length < 2) return null;
  const className = parts.slice(0, -1).join(".");
  if (!index.workspaceClasses.has(className)) return null;
  const memberName = parts[parts.length - 1];
  const info = scope.code.classes.get(className);
  if (info?.enumMembers?.some(m => m.name === memberName))
    return { kind: "enumMember", name, className, memberName };
  const target: Target<"classMethod"> = {
    kind: "classMethod",
    className,
    methodName: memberName,
    stripInstance: false,
  };
  // Every class has an implicit static `empty`.
  return index.classStaticMethods.get(className)?.has(memberName) ||
    memberName === "empty"
    ? { kind: "staticMethod", name, target }
    : { kind: "classMember", name, target };
}

/** The call target of a callable resolution, else null. */
export function targetOf(r: NameResolution): ResolvedTarget | null {
  return "target" in r ? r.target : null;
}

/** The file defining what `r` names, or undefined for variables, builtins,
 *  enumeration members and names not found. Backs `which`. */
export function definingFile(
  r: NameResolution,
  scope: Scope
): string | undefined {
  const code = scope.code;
  switch (r.kind) {
    case "variable":
    case "constant":
    case "special":
    case "builtin":
    case "enumMember":
    case "notFound":
      return undefined;
    case "nested":
      return scope.file;
    case "classConstructor":
      return code.classes.get(r.target.className)?.fileName;
    case "jsUserFunction":
      return code.jsUserFunctionsByName.get(r.target.name)?.fileName;
    case "workspaceFunction":
    case "packageFunction":
      return code.filesByFuncName.get(r.target.name)?.fileName;
    case "mainLocal":
    case "fileLocal":
    case "private":
    case "classMethod":
    case "classMember":
    case "staticMethod": {
      const body = functionBodyFor(code, r.target);
      if (body) return body.file;
      // The implicit `Class.empty`.
      return r.kind === "staticMethod"
        ? code.classes.get(r.target.className)?.fileName
        : undefined;
    }
  }
}

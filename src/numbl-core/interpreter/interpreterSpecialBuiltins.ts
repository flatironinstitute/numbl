/**
 * Interpreter special builtins: functions that need access to interpreter
 * context (env, callerEnv, etc.) and run before normal function resolution.
 */

import { DEFAULT_TEXT_ENCODING } from "../helpers/textCodec.js";
import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeFunction,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import {
  getIBuiltin,
  getIBuiltinHelp,
  getIBuiltinNargin,
} from "./builtins/index.js";
import {
  getAllBuiltinNames,
  registerExtraBuiltinNames,
} from "../helpers/registry.js";
import { toNumber, toString } from "../runtime/convert.js";
import { getConstant } from "../helpers/constants.js";
import type { Environment } from "./types.js";
import type { Runtime } from "../runtime/runtime.js";
import type { NameResolution } from "../resolution/resolveName.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface InterpreterContext {
  env: Environment;
  setEnv: (env: Environment) => void;
  callerEnv: Environment | undefined;
  workspaceEnv: Environment | undefined;
  evalInLocalScope: (
    codeArg: unknown,
    fileName?: string,
    nargout?: number
  ) => unknown;
  /** Call a function named by a string, from the caller's scope (no nested
   *  functions, as in MATLAB). */
  callByName: (name: string, args: unknown[], nargout: number) => unknown;
  /** What `name` means in the caller's scope, and the file defining it. */
  lookupName: (name: string) => {
    resolution: NameResolution;
    file: string | undefined;
  };
  /** Declared input count of the function `name` resolves to (negative
   *  with varargin), or undefined. */
  declaredNargin: (name: string) => number | undefined;
  rt: Runtime;
  /** Optimization mode: "0" = no JIT, "1" = JS-JIT, "2" = C-JIT. */
  optimization: import("../executors/plugins.js").OptLevel;
  /** Resolve a workspace function or class name to its source file,
   *  or undefined if no workspace file provides that name.
   *  `kind` distinguishes a regular .m function from a .numbl.js
   *  user function (treated as a MEX-equivalent) or a class file.
   *  `source` is the raw .m source for "function"/"class" kinds, or the
   *  help text registered by a "jsfunction" (empty if none). */
  lookupWorkspaceFile: (name: string) =>
    | {
        path: string;
        kind: "function" | "jsfunction" | "class";
        source: string;
      }
    | undefined;
  /** Public (GetAccess public) property names of a class, including inherited
   *  ones, in declaration order. Returns undefined if `className` is not a
   *  known class. Backs `properties(obj)`. */
  classPublicProperties: (className: string) => string[] | undefined;
  /** Public method names of a class (including inherited, static and the
   *  constructor), or undefined if not a known class. Backs `ismethod`. */
  classPublicMethods: (className: string) => Set<string> | undefined;
}

export const FALL_THROUGH: unique symbol = Symbol("FALL_THROUGH");

export type InterpreterSpecialBuiltinHandler = (
  ctx: InterpreterContext,
  args: unknown[],
  nargout: number
) => unknown | typeof FALL_THROUGH;

// ── Registry ─────────────────────────────────────────────────────────────

const registry = new Map<string, InterpreterSpecialBuiltinHandler>();

function register(
  name: string,
  handler: InterpreterSpecialBuiltinHandler
): void {
  registry.set(name, handler);
  registerExtraBuiltinNames([name]);
}

export function getInterpreterSpecialBuiltin(
  name: string
): InterpreterSpecialBuiltinHandler | undefined {
  return registry.get(name);
}

// ── Handlers ─────────────────────────────────────────────────────────────

register("eval", (ctx, args, nargout) => {
  if (args.length === 1)
    return ctx.evalInLocalScope(args[0], undefined, nargout);
  if (args.length === 2) {
    // eval(try, catch): run the catch text if the try text errors.
    try {
      return ctx.evalInLocalScope(args[0], undefined, nargout);
    } catch {
      return ctx.evalInLocalScope(args[1], undefined, nargout);
    }
  }
  throw new RuntimeError("eval requires 1 or 2 arguments");
});

/** The environment named by an evalin/assignin scope argument. `'workspace'`
 *  is accepted as a synonym for `'base'`. At the top level the caller is the
 *  base workspace. */
function scopeEnv(
  ctx: InterpreterContext,
  fn: string,
  arg: unknown
): Environment {
  const scope = toString(ensureRuntimeValue(arg));
  if (scope === "caller") return ctx.callerEnv ?? ctx.workspaceEnv ?? ctx.env;
  if (scope === "base" || scope === "workspace")
    return ctx.workspaceEnv ?? ctx.env;
  throw new RuntimeError(`${fn}: first argument must be 'base' or 'caller'`);
}

register("evalin", (ctx, args, nargout) => {
  if (args.length < 2)
    throw new RuntimeError("evalin requires at least 2 arguments");
  const scope = toString(ensureRuntimeValue(args[0]));
  const targetEnv = scopeEnv(ctx, "evalin", args[0]);
  const code = toString(ensureRuntimeValue(args[1]));
  // Simple variable name — just look it up
  if (/^[a-zA-Z_]\w*$/.test(code)) {
    const val = targetEnv.get(code);
    if (val !== undefined) return val;
    if (args.length >= 3) return ensureRuntimeValue(args[2]);
    throw new RuntimeError(
      `Variable '${code}' does not exist in ${scope} scope`
    );
  }
  // Otherwise evaluate as code in target scope
  const savedEnv = ctx.env;
  ctx.setEnv(targetEnv);
  try {
    return ctx.evalInLocalScope(args[1], undefined, nargout);
  } finally {
    ctx.setEnv(savedEnv);
  }
});

register("str2num", (ctx, args) => {
  // MATLAB's str2num evaluates the text as the expression `[<text>]`, so it
  // handles vectors, matrices and arithmetic -- not just a single scalar.
  // (The strings.ts IBuiltin only covers the degenerate non-text fallback.)
  // Returns [] if the text can't be evaluated.
  if (args.length !== 1) return FALL_THROUGH;
  const v = ensureRuntimeValue(args[0]);
  if (!isRuntimeChar(v) && !isRuntimeString(v)) return FALL_THROUGH;
  const empty = () => RTV.tensor(new Float64Array(0), [0, 0]);
  const s = toString(v);
  // Capture the result through a temp variable rather than eval's return
  // value: eval's returnValue is unreliable for a bare-literal result (e.g.
  // `[42]` yields 0), whereas variable propagation is exact.
  const tmp = "__numbl_str2num_result__";
  const saved = ctx.env.get(tmp);
  try {
    ctx.evalInLocalScope(`${tmp} = [${s}];`);
    let result = ctx.env.get(tmp);
    if (result === undefined || result === null) return empty();
    // The `[...]` wrapper yields a tensor even for a single value; demote a
    // 1x1 result to a scalar so str2num('42') matches MATLAB's scalar double
    // (numbl distinguishes a scalar number from a 1x1 tensor).
    if (isRuntimeTensor(result) && result.data.length === 1) {
      result =
        result.imag && result.imag[0] !== 0
          ? RTV.complex(result.data[0], result.imag[0])
          : RTV.num(result.data[0]);
    }
    return result;
  } catch {
    return empty();
  } finally {
    // Restore any pre-existing binding; otherwise remove the temp.
    if (saved !== undefined) ctx.env.set(tmp, saved);
    else ctx.env.delete(tmp);
  }
});

register("assignin", (ctx, args) => {
  if (args.length < 3) throw new RuntimeError("assignin requires 3 arguments");
  const varName = toString(ensureRuntimeValue(args[1]));
  const val = ensureRuntimeValue(args[2]);
  scopeEnv(ctx, "assignin", args[0]).set(varName, val);
  return undefined;
});

register("input", (ctx, args) => {
  const prompt = args.length >= 1 ? toString(ensureRuntimeValue(args[0])) : "";
  const stringMode =
    args.length >= 2 && toString(ensureRuntimeValue(args[1])) === "s";
  const line = ctx.rt.readInput(prompt);
  if (stringMode) return RTV.char(line);
  if (line.trim() === "") return RTV.tensor(new Float64Array(0), [0, 0]);
  // Evaluate the response as an expression in the calling scope, through a
  // temp variable (as str2num does).
  const tmp = "__numbl_input_result__";
  try {
    ctx.evalInLocalScope(`${tmp} = ${line};`);
    const result = ctx.env.get(tmp);
    if (result === undefined)
      throw new RuntimeError("input: the response did not produce a value");
    return result;
  } finally {
    ctx.env.delete(tmp);
  }
});

register("clear", (ctx, args) => {
  // `clear` (no args): remove all locals in the current scope.
  // `clear name1 name2 ...`: remove the named locals.
  // Reserved-word args like 'all', 'global', 'functions', 'classes',
  // '-regexp', etc. are not (yet) supported — fall through to the
  // IBuiltin no-op stub so existing scripts that pass them keep
  // running without error.
  if (args.length === 0) {
    ctx.env.clearLocals();
    return undefined;
  }
  const RESERVED = new Set([
    "all",
    "global",
    "functions",
    "classes",
    "import",
    "java",
    "mex",
    "variables",
    "-regexp",
    "-except",
  ]);
  for (const arg of args) {
    const name = toString(ensureRuntimeValue(arg));
    if (RESERVED.has(name)) return FALL_THROUGH;
    ctx.env.delete(name);
  }
  return undefined;
});

/** Extract the leading help comment block from a .m source file.
 *  MATLAB convention: comment lines immediately following the
 *  function/classdef declaration (or the top of a script), terminated
 *  by the first non-comment line. The leading `%` and one optional
 *  space are stripped from each line. */
function extractMHelp(source: string): string {
  const lines = source.split(/\r?\n/);
  let i = 0;
  // Skip an optional leading function/classdef declaration, including
  // line continuations (lines ending in `...`).
  if (i < lines.length && /^\s*(function|classdef)\b/.test(lines[i])) {
    while (i < lines.length && /\.\.\.\s*(%[^\n]*)?$/.test(lines[i])) i++;
    i++;
  }
  // Collect consecutive comment lines (allow blank lines only inside the block).
  const helpLines: string[] = [];
  let sawComment = false;
  while (i < lines.length) {
    const trimmed = lines[i].trimStart();
    if (trimmed.startsWith("%")) {
      let line = trimmed.slice(1);
      if (line.startsWith(" ")) line = line.slice(1);
      helpLines.push(line);
      sawComment = true;
      i++;
    } else if (!sawComment && trimmed === "") {
      // Allow blank lines before the first comment (after the decl line).
      i++;
    } else {
      break;
    }
  }
  return helpLines.join("\n");
}

/** Convert a path-style help target (e.g. `+mip/install.m`) into the
 *  workspace function name that numbl uses internally (`mip.install`).
 *  Returns the input unchanged if it doesn't look like a path. */
function helpPathToFuncName(name: string): string {
  if (!/[/\\]/.test(name) && !name.endsWith(".m")) return name;
  const stripped = name.replace(/\.m$/, "");
  const parts = stripped.split(/[/\\]/).filter(p => p.length > 0);
  return parts.map(p => (p.startsWith("+") ? p.slice(1) : p)).join(".");
}

register("help", (ctx, args, nargout) => {
  // 0-arg form is handled by the IBuiltin (lists builtins).
  if (args.length === 0) return FALL_THROUGH;
  const rawName = toString(ensureRuntimeValue(args[0]));
  // Builtins (with or without registered help) take precedence — let the
  // existing IBuiltin handle them so messages stay consistent.
  if (getIBuiltinHelp(rawName) || getAllBuiltinNames().includes(rawName)) {
    return FALL_THROUGH;
  }
  // Workspace .m function or class — extract leading comment block.
  // Accept both bare names ("mip", "mip.install") and path-style targets
  // ("+mip/install.m") as MATLAB's help() does.
  const lookupName = helpPathToFuncName(rawName);
  const ws =
    ctx.lookupWorkspaceFile(lookupName) ??
    (lookupName !== rawName ? ctx.lookupWorkspaceFile(rawName) : undefined);
  if (ws) {
    const text =
      ws.kind === "jsfunction" ? ws.source.trimEnd() : extractMHelp(ws.source);
    if (text) {
      const out = text + "\n";
      if (nargout === 0) ctx.rt.output(out);
      return nargout >= 1 ? RTV.char(out) : undefined;
    }
  }
  // Fall back to the IBuiltin's "Unknown function" / "No help available" path.
  return FALL_THROUGH;
});

register("feval", (ctx, args, nargout) => {
  if (args.length < 1) return FALL_THROUGH;
  const first = ensureRuntimeValue(args[0]);
  if (isRuntimeFunction(first)) {
    return ctx.rt.index(first, args.slice(1), nargout);
  }
  // Class instance: fall through to normal resolution
  if (isRuntimeClassInstance(first)) {
    return FALL_THROUGH;
  }
  const funcName = toString(first);
  return ctx.callByName(funcName, args.slice(1), nargout);
});

register("exist", (ctx, args) => {
  if (args.length < 1) return FALL_THROUGH;
  const nameArg = toString(ensureRuntimeValue(args[0]));
  const typeArg = args.length >= 2 ? toString(ensureRuntimeValue(args[1])) : "";

  const fio = ctx.rt.fileIO;
  const isBuiltin = (): boolean =>
    !!getIBuiltin(nameArg) ||
    registry.has(nameArg) ||
    getConstant(nameArg) !== undefined;
  const resolution = ctx.lookupName(nameArg).resolution;
  // Nested, local and private functions visible from the caller are files
  // (2), checked before the search path.
  const scopedFunction =
    resolution.kind === "nested" ||
    resolution.kind === "mainLocal" ||
    resolution.kind === "fileLocal" ||
    resolution.kind === "private";

  // Map a file path to a MATLAB type ID:
  // .numbl.js → 3 (MEX), everything else → 2.
  const fileTypeFromExt = (path: string): number =>
    /\.numbl\.js$/i.test(path) ? 3 : 2;

  // Whether `nameArg` already has a recognized MATLAB code-file extension.
  const nameHasKnownExt = /\.(numbl\.js|m|mlx|mlapp)$/i.test(nameArg);

  const isAbsolutePath = (p: string): boolean =>
    p.startsWith("/") || p.startsWith("\\") || /^[a-zA-Z]:[/\\]/.test(p);
  const joinPath = (dir: string, name: string): string => {
    if (!dir) return name;
    if (dir.endsWith("/") || dir.endsWith("\\")) return dir + name;
    return dir + "/" + name;
  };

  // Walk the entire search path looking for `nameArg` as a file or folder.
  // Per-directory precedence:
  //   folder > .m / .mlx / .mlapp > .numbl.js (MEX) > literal name.
  // Note: .m beats .numbl.js to mirror numbl's function-call resolution
  // (`resolution/resolveFunction.ts`), which intentionally diverges from MATLAB's
  // MEX-wins precedence so that scripted code shadows the JS user function.
  // Across directories the first hit wins.  An absolute or rooted path
  // bypasses the search path and is checked directly.
  const walkSearchPath = (acceptDir: boolean): number => {
    if (!fio?.existsPath) return 0;
    const dirs = isAbsolutePath(nameArg)
      ? [""]
      : ctx.rt.searchPaths.length > 0
        ? ctx.rt.searchPaths
        : [""];
    for (const dir of dirs) {
      // 1. Folder match
      if (acceptDir) {
        const t = fio.existsPath(joinPath(dir, nameArg));
        if (t === "dir") return 7;
      }
      // 2. Registered code-file extensions (only if name has no known ext)
      if (!nameHasKnownExt) {
        for (const ext of [".m", ".mlx", ".mlapp"]) {
          const t = fio.existsPath(joinPath(dir, nameArg + ext));
          // A classdef file is a file too (MATLAB: exist('MyClass') is 2).
          if (t === "file") return 2;
        }
        const numblJs = fio.existsPath(joinPath(dir, nameArg + ".numbl.js"));
        if (numblJs === "file") return 3;
      }
      // 3. Literal name (with whatever extension was provided, or none)
      const lit = fio.existsPath(joinPath(dir, nameArg));
      if (lit === "file") return fileTypeFromExt(nameArg);
    }
    return 0;
  };

  // Name-resolution fallback: covers @-folder classes, package classes and
  // any host environment that doesn't expose existsPath (e.g. browser VFS).
  // MATLAB reports a class by its file (2) unless it lives in a package (8),
  // and a package function or a class method not at all (0).
  const resolvedTypeId = (): number => {
    switch (resolution.kind) {
      case "workspaceFunction":
        return 2;
      case "jsUserFunction":
        return 3;
      case "classConstructor":
        return resolution.target.className.includes(".") ? 8 : 2;
      default:
        return 0;
    }
  };

  if (typeArg === "var") {
    return ctx.env.has(nameArg) ? 1 : 0;
  }
  if (typeArg === "builtin") {
    return isBuiltin() ? 5 : 0;
  }
  if (typeArg === "class") {
    const ws = ctx.lookupWorkspaceFile(nameArg);
    return ws?.kind === "class" || resolution.kind === "classConstructor"
      ? 8
      : 0;
  }
  if (typeArg === "dir") {
    return walkSearchPath(true) === 7 ? 7 : 0;
  }
  if (typeArg === "file") {
    // 'file' searchType returns 2, 3, 4, 6, 7, 0 — folders are valid here.
    if (scopedFunction) return 2;
    const t = walkSearchPath(true);
    if (t) return t;
    const r = resolvedTypeId();
    return r === 8 ? 0 : r;
  }
  if (typeArg === "") {
    // No type specified — full MATLAB precedence:
    //   variable > local/private function > built-in > folder > files
    //   (per-dir order, first hit wins)
    if (ctx.env.has(nameArg)) return 1;
    if (scopedFunction) return 2;
    if (isBuiltin()) return 5;
    const t = walkSearchPath(true);
    if (t) return t;
    return resolvedTypeId();
  }
  return FALL_THROUGH;
});

register("which", (ctx, args) => {
  if (args.length < 1) return FALL_THROUGH;
  const nameArg = toString(ensureRuntimeValue(args[0]));
  const all = args
    .slice(1)
    .some(a => toString(ensureRuntimeValue(a)) === "-all");

  // What the name means in the caller's scope: a variable (MATLAB returns
  // the literal "variable"), a function or class file (its path), or a
  // builtin ("built-in (<path>)" in MATLAB; numbl has no source path).
  const { resolution: r, file } = ctx.lookupName(nameArg);
  let found = "";
  if (r.kind === "variable") found = "variable";
  else if (file !== undefined) found = file;
  else if (
    r.kind === "builtin" ||
    r.kind === "special" ||
    r.kind === "constant"
  )
    found = "built-in";
  else found = ctx.lookupWorkspaceFile(nameArg)?.path ?? "";

  // `which(name, '-all')` returns a cell array of the matches; numbl lists
  // only the one that takes precedence.
  if (all)
    return found ? RTV.cell([RTV.char(found)], [1, 1]) : RTV.cell([], [0, 0]);
  // Not found — MATLAB returns '' (empty char array).
  return RTV.char(found);
});

register("isfolder", (ctx, args) => {
  if (args.length < 1) return FALL_THROUGH;
  const fio = ctx.rt.fileIO;
  if (!fio?.existsPath) return FALL_THROUGH;
  const name = toString(ensureRuntimeValue(args[0]));
  return fio.existsPath(name) === "dir" ? 1 : 0;
});

register("isfile", (ctx, args) => {
  if (args.length < 1) return FALL_THROUGH;
  const fio = ctx.rt.fileIO;
  if (!fio?.existsPath) return FALL_THROUGH;
  const name = toString(ensureRuntimeValue(args[0]));
  return fio.existsPath(name) === "file" ? 1 : 0;
});

register("who", (ctx, args, nargout) => {
  const getters: Record<string, () => unknown> = {};
  for (const varName of ctx.env.localNames()) {
    if (varName.startsWith("$")) continue;
    const n = varName;
    getters[n] = () => ctx.env.get(n);
  }
  return ctx.rt.who(nargout, getters, args);
});

register("whos", (ctx, args, nargout) => {
  const getters: Record<string, () => unknown> = {};
  for (const varName of ctx.env.localNames()) {
    if (varName.startsWith("$")) continue;
    const n = varName;
    getters[n] = () => ctx.env.get(n);
  }
  return ctx.rt.whos(nargout, getters, args);
});

register("isa", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  return ctx.rt.isa(args[0], args[1]);
});

register("superclasses", (ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  const v = ensureRuntimeValue(args[0]);
  // MATLAB: superclasses(obj) or superclasses('ClassName') returns a column
  // cell array of the names of the class's superclasses (its ancestors, not
  // including the class itself), or a 0x1 empty cell when the class has no
  // (visible) superclasses or the input is a built-in type.
  let className: string | undefined;
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)) {
    className = v.className;
  } else if (isRuntimeChar(v) || isRuntimeString(v)) {
    className = toString(v);
  }
  const names = className ? ctx.rt.classAncestors(className) : [];
  return RTV.cell(
    names.map(n => RTV.string(n)),
    [names.length, 1]
  );
});

register("__inferred_type_str", (_ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  const rv = ensureRuntimeValue(args[0]);
  if (isRuntimeNumber(rv)) return RTV.string("Number");
  if (isRuntimeTensor(rv)) return RTV.string("Tensor");
  if (isRuntimeCell(rv)) return RTV.string("Cell");
  if (isRuntimeClassInstance(rv))
    return RTV.string(`ClassInstance(${rv.className})`);
  if (isRuntimeChar(rv)) return RTV.string("Char");
  if (isRuntimeString(rv)) return RTV.string("String");
  if (isRuntimeFunction(rv)) return RTV.string("Function");
  if (typeof rv === "boolean") return RTV.string("Boolean");
  return RTV.string("Unknown");
});

register("nargin", (ctx, args) => {
  if (args.length === 0) {
    const v = ctx.env.get("$nargin");
    return v !== undefined ? v : 0;
  }
  // nargin('name'): the declared input count of the function `name` names
  // in the caller's scope. (nargin(@f) is handled by the builtin.)
  const v = args.length === 1 ? ensureRuntimeValue(args[0]) : undefined;
  if (v !== undefined && (isRuntimeChar(v) || isRuntimeString(v))) {
    const name = toString(v);
    const n = ctx.declaredNargin(name) ?? getIBuiltinNargin(name);
    if (n !== undefined) return n;
  }
  return FALL_THROUGH;
});

register("nargout", (ctx, args) => {
  if (args.length !== 0) return FALL_THROUGH;
  const v = ctx.env.get("$nargout");
  return v !== undefined ? v : 0;
});

register("inputname", (ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  // inputname is only meaningful inside a function; $nargin marks a frame.
  const narginVal = ctx.env.get("$nargin");
  if (typeof narginVal !== "number") {
    throw new RuntimeError(
      "You can only call 'inputname' from within a MATLAB function."
    );
  }
  const k = toNumber(ensureRuntimeValue(args[0]));
  if (!Number.isInteger(k) || k < 1) {
    throw new RuntimeError(
      "Argument number must be a positive integer scalar."
    );
  }
  if (k > narginVal) {
    throw new RuntimeError(
      "Argument number exceeds number of function input arguments."
    );
  }
  const names = ctx.env.inputArgNames;
  const name = names && k <= names.length ? names[k - 1] : "";
  return RTV.char(name);
});

register("properties", (ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  const v = ensureRuntimeValue(args[0]);
  // MATLAB: a class object/array lists its public properties; a char/string
  // that names a class lists that class's properties; everything else
  // (numeric, char-not-a-class, struct, cell, logical, ...) yields a 0x1 cell.
  let names: string[] | undefined;
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)) {
    names = ctx.classPublicProperties(v.className);
  } else if (isRuntimeChar(v) || isRuntimeString(v)) {
    names = ctx.classPublicProperties(toString(v));
  }
  if (!names) return RTV.cell([], [0, 1]);
  return RTV.cell(
    names.map(n => RTV.string(n)),
    [names.length, 1]
  );
});

register("ismethod", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  const v = ensureRuntimeValue(args[0]);
  const nameArg = ensureRuntimeValue(args[1]);
  // MATLAB: true when the name is a public method of the object's class (a
  // char/string first argument names the class). Built-in types (numeric,
  // char, struct, cell, ...) are not modelled and return false.
  let className: string | undefined;
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)) {
    className = v.className;
  } else if (isRuntimeChar(v) || isRuntimeString(v)) {
    className = toString(v);
  }
  let methodName: string | undefined;
  if (isRuntimeChar(nameArg) || isRuntimeString(nameArg))
    methodName = toString(nameArg);
  else if (isRuntimeCell(nameArg) && nameArg.data.length === 1) {
    const el = nameArg.data[0];
    if (isRuntimeChar(el) || isRuntimeString(el)) methodName = toString(el);
  }
  if (!className || methodName === undefined) return RTV.logical(false);
  const names = ctx.classPublicMethods(className);
  return RTV.logical(names !== undefined && names.has(methodName));
});

register("narginchk", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  const narginVal = ctx.env.get("$nargin");
  const n = typeof narginVal === "number" ? narginVal : 0;
  const lo = toNumber(ensureRuntimeValue(args[0]));
  const hi = toNumber(ensureRuntimeValue(args[1]));
  if (n < lo) throw new RuntimeError("Not enough input arguments.");
  if (n > hi) throw new RuntimeError("Too many input arguments.");
  return undefined;
});

register("nargoutchk", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  const nargoutVal = ctx.env.get("$nargout");
  const n = typeof nargoutVal === "number" ? nargoutVal : 0;
  const lo = toNumber(ensureRuntimeValue(args[0]));
  const hi = toNumber(ensureRuntimeValue(args[1]));
  if (n < lo) throw new RuntimeError("Not enough output arguments.");
  if (n > hi) throw new RuntimeError("Too many output arguments.");
  return undefined;
});

register("run", (ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  const fio = ctx.rt.fileIO;
  if (!fio)
    throw new RuntimeError("File I/O is not available in this environment");
  const scriptname = toString(ensureRuntimeValue(args[0]));

  // Resolve the file path
  let filePath = scriptname;
  if (!filePath.endsWith(".m")) filePath += ".m";

  // If the path contains a directory separator, use it directly;
  // otherwise search the current directory (handled by fileread).
  // Resolve the absolute path *before* chdir so file reads work on both
  // Node (where chdir affects fs) and browser (where VFS ignores virtual cwd).
  const resolvedFile = fio.resolvePath ? fio.resolvePath(filePath) : filePath;

  const lastSep = Math.max(
    resolvedFile.lastIndexOf("/"),
    resolvedFile.lastIndexOf("\\")
  );
  const scriptDir = lastSep >= 0 ? resolvedFile.slice(0, lastSep) : null;

  // cd to script directory, execute, cd back (per MATLAB behavior).
  // If the script itself changes cwd, don't revert.
  // Both legs of the chdir invoke onCwdChange so the script directory becomes
  // the first-priority search path during execution and the original cwd is
  // restored as the implicit search path after.
  const sys = ctx.rt.system;
  const oldCwd = sys?.cwd() ?? "/";
  if (scriptDir && sys) {
    try {
      sys.chdir(scriptDir);
    } catch {
      throw new RuntimeError(`Cannot change directory to '${scriptDir}'`);
    }
    if (ctx.rt.onCwdChange) {
      ctx.rt.onCwdChange(sys.cwd());
    }
  }
  const cwdAfterCd = sys?.cwd() ?? "/";
  try {
    const bytes = ctx.rt.files!.readFileBytes(resolvedFile);
    const code = DEFAULT_TEXT_ENCODING.decode(bytes, 0, bytes.length).text;
    ctx.evalInLocalScope(code, resolvedFile);
  } finally {
    // Revert cwd only if the script didn't change it
    if (sys && sys.cwd() === cwdAfterCd) {
      try {
        sys.chdir(oldCwd);
      } catch {
        // ignore
      }
      if (ctx.rt.onCwdChange) {
        ctx.rt.onCwdChange(sys.cwd());
      }
    }
  }
  return undefined;
});

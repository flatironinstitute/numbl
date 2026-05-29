/**
 * Interpreter special builtins: functions that need access to interpreter
 * context (env, callerEnv, etc.) and run before normal function resolution.
 */

import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeFunction,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { getIBuiltin, getIBuiltinHelp } from "./builtins/index.js";
import { getAllBuiltinNames } from "../helpers/registry.js";
import { toNumber, toString } from "../runtime/convert.js";
import type { Environment } from "./types.js";
import type { Runtime } from "../runtime/runtime.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface InterpreterContext {
  env: Environment;
  setEnv: (env: Environment) => void;
  callerEnv: Environment | undefined;
  workspaceEnv: Environment | undefined;
  evalInLocalScope: (codeArg: unknown, fileName?: string) => unknown;
  callFunction: (name: string, args: unknown[], nargout: number) => unknown;
  rt: Runtime;
  /** Optimization mode: "0" = no JIT, "1" = JS-JIT, "2" = C-JIT. */
  optimization: import("../executors/plugins.js").OptLevel;
  /** Resolve a workspace function or class name to its source file,
   *  or undefined if no workspace file provides that name.
   *  `kind` distinguishes a regular .m function from a .numbl.js
   *  user function (treated as a MEX-equivalent) or a class file.
   *  `source` is the raw .m source for "function"/"class" kinds, or empty
   *  for "jsfunction" (where the implementation is JS, not .m text). */
  lookupWorkspaceFile: (name: string) =>
    | {
        path: string;
        kind: "function" | "jsfunction" | "class";
        source: string;
      }
    | undefined;
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
}

export function getInterpreterSpecialBuiltin(
  name: string
): InterpreterSpecialBuiltinHandler | undefined {
  return registry.get(name);
}

// ── Handlers ─────────────────────────────────────────────────────────────

register("eval", (ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  return ctx.evalInLocalScope(args[0]);
});

register("evalin", (ctx, args) => {
  if (args.length < 2) return FALL_THROUGH;
  const scope = toString(ensureRuntimeValue(args[0]));
  const targetEnv =
    scope === "caller"
      ? ctx.callerEnv
      : scope === "workspace"
        ? ctx.workspaceEnv
        : undefined;
  if (targetEnv) {
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
      return ctx.evalInLocalScope(args[1]);
    } finally {
      ctx.setEnv(savedEnv);
    }
  }
  return ctx.evalInLocalScope(args[1]);
});

register("assignin", (ctx, args) => {
  if (args.length < 3) return FALL_THROUGH;
  const scope = toString(ensureRuntimeValue(args[0]));
  const varName = toString(ensureRuntimeValue(args[1]));
  const val = ensureRuntimeValue(args[2]);
  const targetEnv =
    scope === "caller"
      ? ctx.callerEnv
      : scope === "workspace"
        ? ctx.workspaceEnv
        : undefined;
  if (targetEnv) {
    targetEnv.set(varName, val);
  } else {
    ctx.env.set(varName, val);
  }
  return undefined;
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
  if (ws && ws.kind !== "jsfunction") {
    const text = extractMHelp(ws.source);
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
  return ctx.callFunction(funcName, args.slice(1), nargout);
});

register("exist", (ctx, args) => {
  if (args.length < 1) return FALL_THROUGH;
  const nameArg = toString(ensureRuntimeValue(args[0]));
  const typeArg = args.length >= 2 ? toString(ensureRuntimeValue(args[1])) : "";

  const fio = ctx.rt.fileIO;
  const isBuiltin = (): boolean =>
    !!(ctx.rt.builtins[nameArg] || getIBuiltin(nameArg));

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
  // (`functionResolve.ts`), which intentionally diverges from MATLAB's
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
          if (t === "file") {
            // .m can be a class file — promote to 8 if so
            const ws = ctx.lookupWorkspaceFile(nameArg);
            return ws?.kind === "class" ? 8 : 2;
          }
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

  // Workspace registry fallback: covers @-folder class definitions and any
  // host environment that doesn't expose existsPath (e.g. browser VFS).
  const workspaceTypeId = (): number => {
    const ws = ctx.lookupWorkspaceFile(nameArg);
    if (!ws) return 0;
    switch (ws.kind) {
      case "function":
        return 2;
      case "jsfunction":
        return 3;
      case "class":
        return 8;
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
    return ws?.kind === "class" ? 8 : 0;
  }
  if (typeArg === "dir") {
    return walkSearchPath(true) === 7 ? 7 : 0;
  }
  if (typeArg === "file") {
    // 'file' searchType returns 2, 3, 4, 6, 7, 0 — folders are valid here.
    const t = walkSearchPath(true);
    if (t) return t;
    return workspaceTypeId();
  }
  if (typeArg === "") {
    // No type specified — full MATLAB precedence:
    //   variable > built-in > folder > files (per-dir order, first hit wins)
    if (ctx.env.has(nameArg)) return 1;
    if (isBuiltin()) return 5;
    const t = walkSearchPath(true);
    if (t) return t;
    return workspaceTypeId();
  }
  return FALL_THROUGH;
});

register("which", (ctx, args) => {
  if (args.length < 1) return FALL_THROUGH;
  const nameArg = toString(ensureRuntimeValue(args[0]));

  // Variable in current workspace — MATLAB returns the literal "variable".
  if (ctx.env.has(nameArg)) return RTV.char("variable");

  // Workspace function or class file — return the absolute file path.
  const ws = ctx.lookupWorkspaceFile(nameArg);
  if (ws) return RTV.char(ws.path);

  // Builtin — MATLAB returns "built-in (<path>)".  We don't track the
  // source path for numbl builtins, so return just "built-in".
  if (ctx.rt.builtins[nameArg] || getIBuiltin(nameArg)) {
    return RTV.char("built-in");
  }

  // Not found — MATLAB returns '' (empty char array).
  return RTV.char("");
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
  if (args.length !== 0) return FALL_THROUGH;
  const v = ctx.env.get("$nargin");
  return v !== undefined ? v : 0;
});

register("nargout", (ctx, args) => {
  if (args.length !== 0) return FALL_THROUGH;
  const v = ctx.env.get("$nargout");
  return v !== undefined ? v : 0;
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
    const code = fio.fileread(resolvedFile);
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

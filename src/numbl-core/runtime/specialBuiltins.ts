/**
 * Special builtin functions that require direct runtime access.
 *
 * These builtins are registered as IBuiltins (via registerDynamicIBuiltin)
 * that close over the Runtime instance, unifying them with the standard
 * IBuiltin dispatch path.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  toNumber,
  displayValue,
  RuntimeError,
} from "../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeFunction,
  isRuntimeNumber,
} from "../runtime/types.js";
import { sprintfFormat } from "../../numbl-core/helpers/string.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import {
  arrayfunImpl as _arrayfunImpl,
  cellfunImpl as _cellfunImpl,
  structfunImpl as _structfunImpl,
  bsxfunImpl as _bsxfunImpl,
  subsrefBuiltin as _subsrefBuiltin,
  subsasgnBuiltin as _subsasgnBuiltin,
} from "./runtimeDispatch.js";
import type { Runtime } from "./runtime.js";
import { registerDynamicIBuiltin } from "../interpreter/builtins/types.js";

/** Names of all special builtins (needed at lowering time before runtime exists). */
export const SPECIAL_BUILTIN_NAMES: readonly string[] = [
  "disp",
  "fprintf",
  "arrayfun",
  "cellfun",
  "structfun",
  "feval",
  "bsxfun",
  "subsref",
  "subsasgn",
  "builtin",
  "fopen",
  "fclose",
  "fgetl",
  "fgets",
  "fileread",
  "feof",
  "ferror",
  "fileparts",
  "fullfile",
  "assignin",
  "evalin",
  "drawnow",
  "pause",
  "mfilename",
  "addpath",
  "rmpath",
  "path",
  "mkdir",
  "websave",
  "delete",
  "rmdir",
  "unzip",
];

/** Helper: register a special builtin as an IBuiltin that accepts any args. */
function registerSpecial(
  name: string,
  fn: (nargout: number, args: RuntimeValue[]) => unknown
): void {
  registerDynamicIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "unknown" as const }],
      apply: (args: RuntimeValue[], nargout: number) =>
        fn(nargout, args) as RuntimeValue | RuntimeValue[],
    }),
  });
}

/**
 * Register all special builtins as IBuiltins closing over the runtime instance.
 */
export function registerSpecialBuiltins(rt: Runtime): void {
  registerSpecial("disp", (_nargout, args) => {
    if (args.length >= 1) {
      const mv = ensureRuntimeValue(args[0]);
      if (isRuntimeTensor(mv) && mv.data.length === 0) return 0;
      rt.output(displayValue(mv) + "\n");
    }
    return 0;
  });

  registerSpecial("fprintf", (_nargout, args) => {
    let output = "";
    if (args.length >= 1) {
      const margs = args.map(a => ensureRuntimeValue(a));

      // Detect optional leading fid argument (numeric first arg with >= 2 args)
      let fid = 1; // default stdout
      let fmtIdx = 0;
      if (margs.length >= 2 && isRuntimeNumber(margs[0])) {
        fid = toNumber(margs[0]);
        fmtIdx = 1;
      }

      const fmt = toString(margs[fmtIdx]);
      const scalarArgs: RuntimeValue[] = [];
      for (let i = fmtIdx + 1; i < margs.length; i++) {
        const a = margs[i];
        if (isRuntimeTensor(a)) {
          for (let j = 0; j < a.data.length; j++)
            scalarArgs.push(RTV.num(a.data[j]));
        } else {
          scalarArgs.push(a);
        }
      }
      const specCount = (fmt.match(/%[^%]/g) || []).length;
      if (specCount === 0 || scalarArgs.length === 0) {
        output = sprintfFormat(fmt, scalarArgs);
      } else {
        let idx = 0;
        while (idx < scalarArgs.length) {
          const batch = scalarArgs.slice(idx, idx + specCount);
          output += sprintfFormat(fmt, batch);
          idx += specCount;
        }
      }

      if (fid === 1 || fid === 2) {
        rt.output(output);
      } else {
        if (!rt.fileIO)
          throw new RuntimeError(
            "File I/O is not available in this environment"
          );
        rt.fileIO.fwrite(fid, output);
      }
    }
    return output.length;
  });

  registerSpecial("arrayfun", (nargout, args) => {
    return _arrayfunImpl(rt, nargout, args);
  });

  registerSpecial("cellfun", (nargout, args) => {
    return _cellfunImpl(rt, nargout, args);
  });

  registerSpecial("structfun", (nargout, args) => {
    return _structfunImpl(rt, nargout, args);
  });

  registerSpecial("feval", (nargout, args) => {
    if (args.length < 1)
      throw new RuntimeError("feval requires at least 1 argument");
    const fn = ensureRuntimeValue(args[0]);
    if (isRuntimeFunction(fn)) {
      if (fn.jsFn) {
        return fn.jsFnExpectsNargout
          ? fn.jsFn(nargout, ...args.slice(1))
          : fn.jsFn(...args.slice(1));
      }
      return rt.dispatch(fn.name, nargout, args.slice(1));
    }
    if (isRuntimeChar(fn) || isRuntimeString(fn)) {
      const name = typeof fn === "string" ? fn : fn.value;
      return rt.dispatch(name, nargout, args.slice(1));
    }
    throw new RuntimeError(
      "feval: first argument must be a function handle or name"
    );
  });

  registerSpecial("bsxfun", (nargout, args) => {
    return _bsxfunImpl(rt, nargout, args);
  });

  registerSpecial("subsref", (nargout, args) => {
    return _subsrefBuiltin(rt, nargout, args);
  });

  registerSpecial("subsasgn", (nargout, args) => {
    return _subsasgnBuiltin(rt, nargout, args);
  });

  registerSpecial("builtin", (nargout, args) => {
    if (args.length < 1)
      throw new RuntimeError("builtin requires at least 1 argument");
    const fnNameArg = ensureRuntimeValue(args[0]);
    let fnName: string;
    if (isRuntimeFunction(fnNameArg)) {
      fnName = fnNameArg.name;
    } else if (isRuntimeChar(fnNameArg) || isRuntimeString(fnNameArg)) {
      fnName = isRuntimeString(fnNameArg) ? fnNameArg : fnNameArg.value;
    } else {
      throw new RuntimeError(
        "builtin: first argument must be a function name or handle"
      );
    }
    return rt.callBuiltin(fnName, nargout, args.slice(1));
  });

  // ── File I/O builtins ──────────────────────────────────────────────

  const requireFileIO = () => {
    if (!rt.fileIO)
      throw new RuntimeError("File I/O is not available in this environment");
    return rt.fileIO;
  };

  registerSpecial("fopen", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fopen requires at least 1 argument");
    const filename = toString(margs[0]);
    const permission = margs.length >= 2 ? toString(margs[1]) : "r";
    return RTV.num(io.fopen(filename, permission));
  });

  registerSpecial("fclose", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fclose requires 1 argument");
    const arg = margs[0];
    if (isRuntimeChar(arg) || isRuntimeString(arg)) {
      const s = toString(arg);
      if (s === "all") return RTV.num(io.fclose("all"));
      throw new RuntimeError("fclose: invalid argument");
    }
    return RTV.num(io.fclose(toNumber(arg)));
  });

  registerSpecial("fgetl", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fgetl requires 1 argument");
    const result = io.fgetl(toNumber(margs[0]));
    return typeof result === "number" ? RTV.num(result) : RTV.char(result);
  });

  registerSpecial("fgets", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fgets requires 1 argument");
    const result = io.fgets(toNumber(margs[0]));
    return typeof result === "number" ? RTV.num(result) : RTV.char(result);
  });

  registerSpecial("fileread", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fileread requires 1 argument");
    return RTV.char(io.fileread(toString(margs[0])));
  });

  registerSpecial("feof", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("feof requires 1 argument");
    return RTV.num(io.feof(toNumber(margs[0])));
  });

  registerSpecial("ferror", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("ferror requires 1 argument");
    return RTV.char(io.ferror(toNumber(margs[0])));
  });

  registerSpecial("mkdir", (nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("mkdir requires at least 1 argument");
    if (!io.mkdir)
      throw new RuntimeError("mkdir is not available in this environment");
    let dirPath: string;
    if (margs.length === 1) {
      dirPath = toString(margs[0]);
    } else {
      // mkdir(parent, newDir) form
      dirPath = toString(margs[0]) + "/" + toString(margs[1]);
    }
    const ok = io.mkdir(dirPath);
    if (nargout === 0) {
      if (!ok)
        throw new RuntimeError(`mkdir: cannot create directory '${dirPath}'`);
      return undefined;
    }
    return nargout <= 1
      ? RTV.char(ok ? "true" : "false")
      : [
          RTV.char(ok ? "true" : "false"),
          RTV.char(ok ? "" : `Cannot create directory '${dirPath}'`),
          RTV.char(""),
        ];
  });

  // ── websave ─────────────────────────────────────────────────────────

  registerSpecial("websave", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 2)
      throw new RuntimeError("websave requires at least 2 arguments");
    if (!io.websave)
      throw new RuntimeError("websave is not available in this environment");
    const filename = toString(margs[0]);
    let url = toString(margs[1]);
    // Append query parameters (name-value pairs)
    const queryParts: string[] = [];
    for (let i = 2; i + 1 < margs.length; i += 2) {
      const name = encodeURIComponent(toString(margs[i]));
      const value = encodeURIComponent(toString(margs[i + 1]));
      queryParts.push(`${name}=${value}`);
    }
    if (queryParts.length > 0) {
      const sep = url.includes("?") ? "&" : "?";
      url += sep + queryParts.join("&");
    }
    io.websave(url, filename);
    return RTV.char(filename);
  });

  // ── delete (file deletion) ──────────────────────────────────────────

  registerSpecial("delete", (_nargout, args) => {
    const io = requireFileIO();
    if (!io.deleteFile)
      throw new RuntimeError("delete is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("delete requires at least 1 argument");
    for (const arg of margs) {
      io.deleteFile(toString(arg));
    }
    return 0;
  });

  // ── rmdir (directory removal) ────────────────────────────────────────

  registerSpecial("rmdir", (nargout, args) => {
    const io = requireFileIO();
    if (!io.rmdir)
      throw new RuntimeError("rmdir is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("rmdir requires at least 1 argument");
    const dirPath = toString(margs[0]);
    const recursive =
      margs.length >= 2 && toString(margs[1]).toLowerCase() === "s";
    const ok = io.rmdir(dirPath, recursive);
    if (nargout === 0) {
      if (!ok)
        throw new RuntimeError(`rmdir: cannot remove directory '${dirPath}'`);
      return undefined;
    }
    return nargout <= 1
      ? RTV.num(ok ? 1 : 0)
      : [
          RTV.num(ok ? 1 : 0),
          RTV.char(ok ? "" : `Cannot remove directory '${dirPath}'`),
          RTV.char(""),
        ];
  });

  // ── unzip (ZIP extraction) ──────────────────────────────────────────

  registerSpecial("unzip", (nargout, args) => {
    const io = requireFileIO();
    if (!io.unzip)
      throw new RuntimeError("unzip is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("unzip requires at least 1 argument");

    // Parse name-value pairs (Password=...)
    let zipfilename = toString(margs[0]);
    let outputfolder = ".";
    let nextArg = 1;

    // Second positional arg is outputfolder (if not a name-value pair)
    if (margs.length >= 2 && toString(margs[1]) !== "Password") {
      outputfolder = toString(margs[1]);
      nextArg = 2;
    }

    // Check for Password name-value pair (not supported, but give clear error)
    for (let i = nextArg; i < margs.length; i += 2) {
      const name = toString(margs[i]);
      if (name === "Password") {
        throw new RuntimeError(
          "unzip: Password-protected ZIP files are not supported"
        );
      }
    }

    // If no extension, try appending .zip
    if (!zipfilename.includes(".")) {
      zipfilename = zipfilename + ".zip";
    }

    const extracted = io.unzip(zipfilename, outputfolder);

    if (nargout >= 1) {
      // Return cell array of extracted file names
      const cellData: RuntimeValue[] = extracted.map(f => RTV.char(f));
      return RTV.cell(cellData, [1, cellData.length]);
    }
    return 0;
  });

  // ── Path utility builtins (pure string operations, no fs needed) ──

  registerSpecial("fileparts", (nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fileparts requires 1 argument");
    const p = toString(margs[0]);
    // Find the last path separator
    const lastSep = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    const dir = lastSep >= 0 ? p.slice(0, lastSep) : "";
    const rest = lastSep >= 0 ? p.slice(lastSep + 1) : p;
    // Find the extension
    const dotIdx = rest.lastIndexOf(".");
    const name = dotIdx >= 0 ? rest.slice(0, dotIdx) : rest;
    const ext = dotIdx >= 0 ? rest.slice(dotIdx) : "";
    if (nargout <= 1) return RTV.char(dir);
    if (nargout === 2) return [RTV.char(dir), RTV.char(name)];
    return [RTV.char(dir), RTV.char(name), RTV.char(ext)];
  });

  registerSpecial("fullfile", (_nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    const parts = margs.map(a => toString(a));
    return RTV.char(parts.join("/"));
  });

  // ── Workspace builtins ───────────────────────────────────────────

  registerSpecial("assignin", (_nargout, args) => {
    if (args.length < 3)
      throw new RuntimeError("assignin requires 3 arguments");
    const margs = args.map(a => ensureRuntimeValue(a));
    const ws = toString(margs[0]);
    if (ws !== "base" && ws !== "caller" && ws !== "workspace")
      throw new RuntimeError(
        "assignin: first argument must be 'base', 'caller', or 'workspace'"
      );
    const varName = toString(margs[1]);
    if (ws === "caller") {
      rt.setCallerVariable(varName, args[2]);
    } else {
      rt.setWorkspaceVariable(varName, args[2]);
    }
    return 0;
  });

  registerSpecial("evalin", (_nargout, args) => {
    if (args.length < 2)
      throw new RuntimeError("evalin requires at least 2 arguments");
    const margs = args.map(a => ensureRuntimeValue(a));
    const ws = toString(margs[0]);
    if (ws !== "base" && ws !== "caller" && ws !== "workspace")
      throw new RuntimeError(
        "evalin: first argument must be 'base', 'caller', or 'workspace'"
      );
    const varName = toString(margs[1]);
    const val =
      ws === "caller"
        ? rt.getCallerVariable(varName)
        : rt.getWorkspaceVariable(varName);
    if (val === undefined) {
      if (args.length >= 3) return args[2];
      throw new RuntimeError(
        `evalin: variable '${varName}' does not exist in ${ws}`
      );
    }
    return val;
  });

  // ── Plot builtins ────────────────────────────────────────────────

  registerSpecial("drawnow", () => {
    rt.drawnow();
    return 0;
  });

  registerSpecial("pause", (_nargout, args) => {
    rt.pause(args[0] ?? 0);
    return 0;
  });

  // ── mfilename builtin ────────────────────────────────────────────

  registerSpecial("mfilename", (_nargout, args) => {
    const file = rt.$file ?? "";
    if (args.length > 0) {
      const margs = args.map(a => ensureRuntimeValue(a));
      const opt = toString(margs[0]);
      if (opt === "fullpath") {
        // Return full path without .m extension
        return RTV.char(file.replace(/\.m$/, ""));
      }
    }
    // Return just the base name without path or extension
    const lastSep = Math.max(file.lastIndexOf("/"), file.lastIndexOf("\\"));
    const baseName = lastSep >= 0 ? file.slice(lastSep + 1) : file;
    return RTV.char(baseName.replace(/\.m$/, ""));
  });

  // ── Path management builtins ──────────────────────────────────────

  registerSpecial("addpath", (nargout, args) => {
    if (!rt.onPathChange) {
      throw new RuntimeError("addpath is not available in this environment");
    }
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("addpath requires at least 1 argument");

    // Detect '-end' or '-begin' flag as last argument
    let position: "begin" | "end" = "begin";
    const lastArg = margs[margs.length - 1];
    if (isRuntimeChar(lastArg) || isRuntimeString(lastArg)) {
      const flag = toString(lastArg);
      if (flag === "-end") {
        position = "end";
        margs.pop();
      } else if (flag === "-begin") {
        margs.pop();
      }
    }

    for (const arg of margs) {
      const dirStr = toString(arg);
      // MATLAB supports pathsep-separated dirs in a single string
      const dirs = dirStr.split(";");
      for (const d of dirs) {
        const trimmed = d.trim();
        if (trimmed) {
          rt.onPathChange("add", trimmed, position);
        }
      }
    }

    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return 0;
  });

  registerSpecial("rmpath", (nargout, args) => {
    if (!rt.onPathChange) {
      throw new RuntimeError("rmpath is not available in this environment");
    }
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("rmpath requires at least 1 argument");

    for (const arg of margs) {
      const dirStr = toString(arg);
      const dirs = dirStr.split(";");
      for (const d of dirs) {
        const trimmed = d.trim();
        if (trimmed) {
          rt.onPathChange("remove", trimmed, "begin");
        }
      }
    }

    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return 0;
  });

  registerSpecial("path", (nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length === 0) {
      // path() — return current path
      return RTV.char(rt.searchPaths.join(";"));
    }
    // path(newpath) — set the entire path (not commonly used, but supported)
    if (margs.length >= 1 && rt.onPathChange) {
      // Remove all current paths then add the new ones
      for (const sp of [...rt.searchPaths]) {
        rt.onPathChange("remove", sp, "begin");
      }
      const newPath = toString(margs[0]);
      const dirs = newPath.split(";");
      for (const d of dirs) {
        const trimmed = d.trim();
        if (trimmed) {
          rt.onPathChange("add", trimmed, "end");
        }
      }
    }
    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return 0;
  });
}

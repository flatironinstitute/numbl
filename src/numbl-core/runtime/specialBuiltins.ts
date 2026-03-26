/**
 * Special builtin functions that require direct runtime access.
 *
 * These builtins are registered separately from the standard builtin registry
 * because they need access to the Runtime instance methods.
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

/** Names of all special builtins (functions requiring runtime access). */
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
];

/**
 * Register all special builtins on the provided runtime instance.
 */
export function registerSpecialBuiltins(rt: Runtime): void {
  rt.builtins["disp"] = (_nargout: number, args: unknown[]) => {
    if (args.length >= 1) {
      const mv = ensureRuntimeValue(args[0]);
      if (isRuntimeTensor(mv) && mv.data.length === 0) return 0;
      rt.output(displayValue(mv) + "\n");
    }
    return 0;
  };

  rt.builtins["fprintf"] = (_nargout: number, args: unknown[]) => {
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
  };

  rt.builtins["arrayfun"] = (nargout: number, args: unknown[]) => {
    return _arrayfunImpl(rt, nargout, args);
  };

  rt.builtins["cellfun"] = (nargout: number, args: unknown[]) => {
    return _cellfunImpl(rt, nargout, args);
  };

  rt.builtins["structfun"] = (nargout: number, args: unknown[]) => {
    return _structfunImpl(rt, nargout, args);
  };

  rt.builtins["feval"] = (nargout: number, args: unknown[]) => {
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
  };

  rt.builtins["bsxfun"] = (nargout: number, args: unknown[]) => {
    return _bsxfunImpl(rt, nargout, args);
  };

  rt.builtins["subsref"] = (nargout: number, args: unknown[]) => {
    return _subsrefBuiltin(rt, nargout, args);
  };

  rt.builtins["subsasgn"] = (nargout: number, args: unknown[]) => {
    return _subsasgnBuiltin(rt, nargout, args);
  };

  rt.builtins["builtin"] = (nargout: number, args: unknown[]) => {
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
    const fn = rt.builtins[fnName];
    if (fn) {
      return fn(nargout, args.slice(1));
    }
    return rt.callBuiltin(fnName, nargout, args.slice(1));
  };

  // ── File I/O builtins ──────────────────────────────────────────────

  const requireFileIO = () => {
    if (!rt.fileIO)
      throw new RuntimeError("File I/O is not available in this environment");
    return rt.fileIO;
  };

  rt.builtins["fopen"] = (_nargout: number, args: unknown[]) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fopen requires at least 1 argument");
    const filename = toString(margs[0]);
    const permission = margs.length >= 2 ? toString(margs[1]) : "r";
    return RTV.num(io.fopen(filename, permission));
  };

  rt.builtins["fclose"] = (_nargout: number, args: unknown[]) => {
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
  };

  rt.builtins["fgetl"] = (_nargout: number, args: unknown[]) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fgetl requires 1 argument");
    const result = io.fgetl(toNumber(margs[0]));
    return typeof result === "number" ? RTV.num(result) : RTV.char(result);
  };

  rt.builtins["fgets"] = (_nargout: number, args: unknown[]) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fgets requires 1 argument");
    const result = io.fgets(toNumber(margs[0]));
    return typeof result === "number" ? RTV.num(result) : RTV.char(result);
  };

  rt.builtins["fileread"] = (_nargout: number, args: unknown[]) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fileread requires 1 argument");
    return RTV.char(io.fileread(toString(margs[0])));
  };

  rt.builtins["feof"] = (_nargout: number, args: unknown[]) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("feof requires 1 argument");
    return RTV.num(io.feof(toNumber(margs[0])));
  };

  rt.builtins["ferror"] = (_nargout: number, args: unknown[]) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("ferror requires 1 argument");
    return RTV.char(io.ferror(toNumber(margs[0])));
  };

  // ── Path utility builtins (pure string operations, no fs needed) ──

  rt.builtins["fileparts"] = (nargout: number, args: unknown[]) => {
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
  };

  rt.builtins["fullfile"] = (_nargout: number, args: unknown[]) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    const parts = margs.map(a => toString(a));
    return RTV.char(parts.join("/"));
  };

  // ── Workspace builtins ───────────────────────────────────────────

  rt.builtins["assignin"] = (_nargout: number, args: unknown[]) => {
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
  };

  rt.builtins["evalin"] = (_nargout: number, args: unknown[]) => {
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
  };

  // ── Plot builtins ────────────────────────────────────────────────

  rt.builtins["drawnow"] = () => {
    rt.drawnow();
    return 0;
  };

  rt.builtins["pause"] = (_nargout: number, args: unknown[]) => {
    rt.pause(args[0] ?? 0);
    return 0;
  };
}

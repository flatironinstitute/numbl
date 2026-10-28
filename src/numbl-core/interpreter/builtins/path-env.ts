/**
 * Search path, current folder, and environment builtins: mfilename,
 * addpath, rmpath, savepath, path, getenv, setenv, maxNumCompThreads, pwd,
 * cd.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  RuntimeError,
} from "../../runtime/index.js";
import {
  isRuntimeChar,
  isRuntimeString,
  isRuntimeDictionary,
} from "../../runtime/types.js";
import { hashKey } from "./dictionary.js";
import { builtin } from "./builtin.js";

builtin({
  name: "mfilename",
  nargin: [0, 1],
  apply: (args, _nargout, ctx) => {
    const file = ctx.rt.$file ?? "";
    if (args.length > 0 && toString(args[0]) === "fullpath") {
      // Return full path without .m extension
      return RTV.char(file.replace(/\.m$/, ""));
    }
    // Return just the base name without path or extension
    const lastSep = Math.max(file.lastIndexOf("/"), file.lastIndexOf("\\"));
    const baseName = lastSep >= 0 ? file.slice(lastSep + 1) : file;
    return RTV.char(baseName.replace(/\.m$/, ""));
  },
});

/** Each directory in a pathsep-separated (';') path string. */
function splitPathArg(v: RuntimeValue): string[] {
  return toString(v)
    .split(";")
    .map(d => d.trim())
    .filter(d => d);
}

builtin({
  name: "addpath",
  nargin: [1, Infinity],
  mfile: true,
  apply: (args, nargout, ctx) => {
    const rt = ctx.rt;
    if (!rt.onPathChange)
      throw new RuntimeError("addpath is not available in this environment");
    const dirs = [...args];

    // Detect '-end' or '-begin' flag as last argument
    let position: "begin" | "end" = "begin";
    const lastArg = dirs[dirs.length - 1];
    if (isRuntimeChar(lastArg) || isRuntimeString(lastArg)) {
      const flag = toString(lastArg);
      if (flag === "-end") {
        position = "end";
        dirs.pop();
      } else if (flag === "-begin") {
        dirs.pop();
      }
    }

    for (const arg of dirs)
      for (const d of splitPathArg(arg)) rt.onPathChange("add", d, position);

    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return undefined;
  },
});

builtin({
  name: "rmpath",
  nargin: [1, Infinity],
  mfile: true,
  apply: (args, nargout, ctx) => {
    const rt = ctx.rt;
    if (!rt.onPathChange)
      throw new RuntimeError("rmpath is not available in this environment");
    for (const arg of args)
      for (const d of splitPathArg(arg)) rt.onPathChange("remove", d, "begin");
    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return undefined;
  },
});

builtin({
  name: "savepath",
  nargin: [0, 1],
  mfile: true,
  apply: (_args, nargout, ctx) => {
    ctx.rt.output("Warning: savepath is a no-op in numbl\n");
    // MATLAB returns 0 on success when an output is requested.
    return nargout >= 1 ? RTV.num(0) : undefined;
  },
});

builtin({
  name: "path",
  nargin: [0, 2],
  mfile: true,
  apply: (args, nargout, ctx) => {
    const rt = ctx.rt;
    if (args.length === 0) {
      // path() — return current path
      return RTV.char(rt.searchPaths.join(";"));
    }
    // path(newpath) — set the entire path (not commonly used, but supported)
    if (rt.onPathChange) {
      // Remove all current paths then add the new ones
      for (const sp of [...rt.searchPaths]) {
        rt.onPathChange("remove", sp, "begin");
      }
      for (const d of splitPathArg(args[0])) rt.onPathChange("add", d, "end");
    }
    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return RTV.num(0);
  },
});

builtin({
  name: "getenv",
  nargin: [0, 1],
  apply: (args, _nargout, ctx) => {
    const sys = ctx.rt.system;
    if (args.length === 0) {
      // getenv() — return dictionary of all env vars
      const all = sys?.getAllEnv() ?? {};
      const entries = new Map<
        string,
        { key: RuntimeValue; value: RuntimeValue }
      >();
      for (const [k, v] of Object.entries(all)) {
        const key = RTV.string(k);
        entries.set(hashKey(key), { key, value: RTV.string(v) });
      }
      return RTV.dictionary(entries, "string", "string");
    }
    // getenv(varname) — return char value (empty if not set)
    return RTV.char(sys?.getEnv(toString(args[0])) ?? "");
  },
});

builtin({
  name: "setenv",
  nargin: [1, 2],
  nargout: 0,
  mfile: true,
  apply: (args, _nargout, ctx) => {
    const sys = ctx.rt.system;
    if (args.length === 2) {
      sys?.setEnv(toString(args[0]), toString(args[1]));
      return;
    }
    const d = args[0];
    if (isRuntimeDictionary(d)) {
      // setenv(d) — set all entries from dictionary
      for (const { key, value } of d.entries.values()) {
        sys?.setEnv(toString(key), toString(value));
      }
      return;
    }
    // setenv(varname) — set to empty string
    sys?.setEnv(toString(d), "");
  },
});

// numbl runs single-threaded, so there is always exactly one computational
// thread. maxNumCompThreads() reports it; maxNumCompThreads(N) / ('automatic')
// would set it in MATLAB and return the PREVIOUS value — here both forms just
// report 1 (the set is a no-op).
builtin({
  name: "maxNumCompThreads",
  nargin: [0, 1],
  apply: () => RTV.num(1),
});

builtin({
  name: "pwd",
  nargin: 0,
  mfile: true,
  apply: (_args, _nargout, ctx) => RTV.char(ctx.rt.system?.cwd() ?? "/"),
});

builtin({
  name: "cd",
  nargin: [0, 1],
  apply: (args, nargout, ctx) => {
    const rt = ctx.rt;
    const sys = rt.system;
    const curDir = sys?.cwd() ?? "/";
    if (args.length === 0) {
      // `cd` with no args: print current directory, return it only if
      // assigned (matches MATLAB, which does not set `ans`).
      if (nargout === 0) {
        rt.output(curDir + "\n");
        return undefined;
      }
      return RTV.char(curDir);
    }
    const target = toString(args[0]);
    if (sys) {
      try {
        sys.chdir(target);
      } catch {
        throw new RuntimeError(`Cannot change directory to '${target}'`);
      }
      // Trigger workspace rebuild for the new cwd (MATLAB semantics:
      // current directory is the first-priority search path).
      if (rt.onCwdChange) {
        rt.onCwdChange(sys.cwd());
      }
    }
    // With a destination argument, MATLAB returns the previous directory
    // only when an output is requested.
    return nargout >= 1 ? RTV.char(curDir) : undefined;
  },
});

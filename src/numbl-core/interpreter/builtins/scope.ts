/**
 * Builtins that work on the calling scope (declared with `scope: true`):
 * eval, evalin, assignin, input, run, clear, who, whos, exist, which,
 * nargin, nargout, narginchk, nargoutchk, inputname.
 */

import { DEFAULT_TEXT_ENCODING } from "../../helpers/textCodec.js";
import {
  isRuntimeChar,
  isRuntimeFunction,
  isRuntimeString,
  type RuntimeValue,
} from "../../runtime/types.js";
import { RTV } from "../../runtime/constructors.js";
import { RuntimeError } from "../../runtime/error.js";
import { toNumber, toString } from "../../runtime/convert.js";
import {
  getIBuiltinNargin,
  getIBuiltinNargout,
  isBuiltin,
} from "../../runtime/builtinRegistry.js";
import { getConstant } from "../../helpers/constants.js";
import type { CallFrame } from "../callFrame.js";
import type { CallerScope } from "../callerScope.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";

// ── eval, evalin, assignin ───────────────────────────────────────────────

builtin({
  name: "eval",
  nargin: [1, 2],
  nargout: Infinity,
  scope: true,
  apply: (args, nargout, { scope }) => {
    if (args.length === 1) return scope.evalText(args[0], nargout);
    // eval(try, catch): run the catch text if the try text errors.
    try {
      return scope.evalText(args[0], nargout);
    } catch {
      return scope.evalText(args[1], nargout);
    }
  },
});

/** The frame named by an evalin/assignin scope argument ('base' or
 *  'caller'). At the top level the caller is the base workspace. */
function scopeFrame(scope: CallerScope, arg: RuntimeValue): CallFrame {
  const name =
    isRuntimeChar(arg) || isRuntimeString(arg) ? toString(arg) : undefined;
  if (name === "caller") return scope.callerFrame ?? scope.baseFrame;
  if (name === "base") return scope.baseFrame;
  throw builtinError("MATLAB:badopt", "Unknown command option.");
}

const notTextScalar = () =>
  builtinError(
    "MATLAB:string:MustBeStringScalarOrCharacterVector",
    "Argument must be a text scalar."
  );

builtin({
  name: "evalin",
  nargin: [2, 3],
  nargout: Infinity,
  scope: true,
  apply: (args, nargout, { scope }) => {
    const target = scopeFrame(scope, args[0]);
    for (const a of args.slice(1))
      if (!isRuntimeChar(a) && !isRuntimeString(a)) throw notTextScalar();
    const run = (text: RuntimeValue) => {
      // A variable name is looked up directly.
      const code = toString(text);
      if (/^[a-zA-Z_]\w*$/.test(code)) {
        const val = target.env.get(code);
        if (val !== undefined) return val;
      }
      return scope.evalInFrame(target, text, nargout);
    };
    if (args.length === 2) return run(args[1]);
    // evalin(ws, try, catch): run the catch text if the try text errors.
    try {
      return run(args[1]);
    } catch {
      return run(args[2]);
    }
  },
});

builtin({
  name: "assignin",
  nargin: 3,
  nargout: 0,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    scopeFrame(scope, args[0]).env.set(toString(args[1]), args[2]);
  },
});

// ── input, run ───────────────────────────────────────────────────────────

builtin({
  name: "input",
  nargin: [1, 2],
  scope: true,
  apply: (args, _nargout, { rt, scope }) => {
    const prompt = toString(args[0]);
    const stringMode = args.length >= 2 && toString(args[1]) === "s";
    const line = rt.readInput(prompt);
    if (stringMode) return RTV.char(line);
    if (line.trim() === "") return RTV.tensor(new Float64Array(0), [0, 0]);
    // Evaluate the response as an expression in the calling scope, through
    // a temp variable (as str2num does).
    const tmp = "__numbl_input_result__";
    try {
      scope.evalText(`${tmp} = ${line};`);
      const result = scope.env.get(tmp);
      if (result === undefined)
        throw new RuntimeError("input: the response did not produce a value");
      return result;
    } finally {
      scope.env.delete(tmp);
    }
  },
});

builtin({
  name: "run",
  nargin: 1,
  nargout: 0,
  mfile: true,
  scope: true,
  apply: (args, _nargout, { rt, scope }) => {
    const fio = rt.fileIO;
    if (!fio)
      throw new RuntimeError("File I/O is not available in this environment");
    let filePath = toString(args[0]);
    if (!filePath.endsWith(".m")) filePath += ".m";

    // Resolve the absolute path *before* chdir so file reads work on both
    // Node (where chdir affects fs) and browser (where the VFS ignores the
    // virtual cwd).
    const resolvedFile = fio.resolvePath ? fio.resolvePath(filePath) : filePath;
    const lastSep = Math.max(
      resolvedFile.lastIndexOf("/"),
      resolvedFile.lastIndexOf("\\")
    );
    const scriptDir = lastSep >= 0 ? resolvedFile.slice(0, lastSep) : null;

    // cd to the script's directory, run it, and cd back (as MATLAB does),
    // unless the script itself changed the directory. Both legs notify
    // onCwdChange so the script's directory is searched first while it
    // runs.
    const sys = rt.system;
    const oldCwd = sys?.cwd() ?? "/";
    if (scriptDir && sys) {
      try {
        sys.chdir(scriptDir);
      } catch {
        throw new RuntimeError(`Cannot change directory to '${scriptDir}'`);
      }
      rt.onCwdChange?.(sys.cwd());
    }
    const cwdAfterCd = sys?.cwd() ?? "/";
    try {
      const bytes = rt.files!.readFileBytes(resolvedFile);
      const code = DEFAULT_TEXT_ENCODING.decode(bytes, 0, bytes.length).text;
      scope.evalText(code, 0, resolvedFile);
    } finally {
      if (sys && sys.cwd() === cwdAfterCd) {
        try {
          sys.chdir(oldCwd);
        } catch {
          // ignore
        }
        rt.onCwdChange?.(sys.cwd());
      }
    }
  },
});

// ── clear, who, whos ─────────────────────────────────────────────────────

/** `clear` options numbl does not implement; a call naming one is a no-op. */
const CLEAR_OPTIONS = new Set([
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

builtin({
  name: "clear",
  nargout: 0,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    // `clear`: remove all variables of the calling scope; `clear a b`: the
    // named ones.
    if (args.length === 0) {
      scope.env.clearLocals();
      return;
    }
    for (const arg of args) {
      const name = toString(arg);
      if (CLEAR_OPTIONS.has(name)) return;
      scope.env.delete(name);
    }
  },
});

/** Getters for the calling scope's variables, for who/whos. */
function variableGetters(scope: CallerScope): Record<string, () => unknown> {
  const getters: Record<string, () => unknown> = {};
  for (const name of scope.env.localNames()) {
    if (name.startsWith("$")) continue;
    getters[name] = () => scope.env.get(name);
  }
  return getters;
}

builtin({
  name: "who",
  scope: true,
  apply: (args, nargout, { rt, scope }) =>
    rt.who(nargout, variableGetters(scope), args) as RuntimeValue,
});

builtin({
  name: "whos",
  scope: true,
  apply: (args, nargout, { rt, scope }) =>
    rt.whos(nargout, variableGetters(scope), args) as RuntimeValue,
});

// ── exist, which ─────────────────────────────────────────────────────────

const isAbsolutePath = (p: string): boolean =>
  p.startsWith("/") || p.startsWith("\\") || /^[a-zA-Z]:[/\\]/.test(p);

const joinPath = (dir: string, name: string): string => {
  if (!dir) return name;
  if (dir.endsWith("/") || dir.endsWith("\\")) return dir + name;
  return dir + "/" + name;
};

builtin({
  name: "exist",
  nargin: [1, 2],
  scope: true,
  apply: (args, _nargout, { rt, scope }) => {
    const nameArg = toString(args[0]);
    const typeArg = args.length >= 2 ? toString(args[1]) : "";

    const fio = rt.fileIO;
    const builtinName = (): boolean =>
      isBuiltin(nameArg) || getConstant(nameArg) !== undefined;
    const resolution = scope.lookupName(nameArg).resolution;
    // Nested, local and private functions visible from the caller are
    // files (2), checked before the search path.
    const scopedFunction =
      resolution.kind === "nested" ||
      resolution.kind === "mainLocal" ||
      resolution.kind === "fileLocal" ||
      resolution.kind === "private";

    // Whether `nameArg` already has a recognized MATLAB code-file extension.
    const nameHasKnownExt = /\.(numbl\.js|m|mlx|mlapp)$/i.test(nameArg);

    // Walk the search path looking for `nameArg` as a file or folder.
    // Per directory: folder > .m / .mlx / .mlapp > .numbl.js (MEX, 3) >
    // literal name. (.m beats .numbl.js as in numbl's function resolution,
    // unlike MATLAB's MEX-wins precedence.) Across directories the first
    // hit wins. An absolute path bypasses the search path.
    const walkSearchPath = (acceptDir: boolean): number => {
      if (!fio?.existsPath) return 0;
      const dirs = isAbsolutePath(nameArg)
        ? [""]
        : rt.searchPaths.length > 0
          ? rt.searchPaths
          : [""];
      for (const dir of dirs) {
        if (acceptDir && fio.existsPath(joinPath(dir, nameArg)) === "dir")
          return 7;
        if (!nameHasKnownExt) {
          for (const ext of [".m", ".mlx", ".mlapp"]) {
            // A classdef file is a file too (MATLAB: exist('MyClass') is 2).
            if (fio.existsPath(joinPath(dir, nameArg + ext)) === "file")
              return 2;
          }
          if (fio.existsPath(joinPath(dir, nameArg + ".numbl.js")) === "file")
            return 3;
        }
        if (fio.existsPath(joinPath(dir, nameArg)) === "file")
          return /\.numbl\.js$/i.test(nameArg) ? 3 : 2;
      }
      return 0;
    };

    // Name-resolution fallback: covers @-folder classes, package classes
    // and hosts without existsPath (e.g. the browser VFS). MATLAB reports a
    // class by its file (2) unless it lives in a package (8), and a package
    // function or a class method not at all (0).
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

    switch (typeArg) {
      case "var":
        return scope.env.has(nameArg) ? 1 : 0;
      case "builtin":
        return builtinName() ? 5 : 0;
      case "class": {
        const ws = scope.lookupWorkspaceFile(nameArg);
        return ws?.kind === "class" || resolution.kind === "classConstructor"
          ? 8
          : 0;
      }
      case "dir":
        return walkSearchPath(true) === 7 ? 7 : 0;
      case "file": {
        // Folders count here too.
        if (scopedFunction) return 2;
        const t = walkSearchPath(true);
        if (t) return t;
        const r = resolvedTypeId();
        return r === 8 ? 0 : r;
      }
      case "": {
        // MATLAB's precedence: variable > local/private function >
        // built-in > folder > files (per directory, first hit wins).
        if (scope.env.has(nameArg)) return 1;
        if (scopedFunction) return 2;
        if (builtinName()) return 5;
        return walkSearchPath(true) || resolvedTypeId();
      }
    }
    throw builtinError(
      "MATLAB:exist:secondOptionalInputString",
      "The optional second input to exist must be 'var', 'builtin', 'class', 'dir' or 'file'."
    );
  },
});

builtin({
  name: "which",
  nargin: [1, Infinity],
  scope: true,
  apply: (args, _nargout, { scope }) => {
    const nameArg = toString(args[0]);
    const all = args.slice(1).some(a => toString(a) === "-all");

    // What the name means in the calling scope: a variable (MATLAB returns
    // the literal "variable"), a function or class file (its path), or a
    // builtin ("built-in (<path>)" in MATLAB; numbl has no source path).
    const { resolution: r, file } = scope.lookupName(nameArg);
    let found = "";
    if (r.kind === "variable") found = "variable";
    else if (file !== undefined) found = file;
    else if (r.kind === "builtin" || r.kind === "constant") found = "built-in";
    else found = scope.lookupWorkspaceFile(nameArg)?.path ?? "";

    // `which(name, '-all')` returns a cell array of the matches; numbl
    // lists only the one that takes precedence.
    if (all)
      return found ? RTV.cell([RTV.char(found)], [1, 1]) : RTV.cell([], [0, 0]);
    return RTV.char(found);
  },
});

// ── nargin, nargout, narginchk, nargoutchk, inputname ─────────────────────

const narginoutBadInput = () =>
  builtinError(
    "MATLAB:narginout:BadInput",
    "Argument must be either a character vector, string scalar, or a function handle."
  );
const notValidMfile = () =>
  builtinError("MATLAB:narginout:notValidMfile", "Not a valid MATLAB file.");

/** The running function's `$nargin` / `$nargout`. Outside a function (the
 *  base workspace or a script run from it) MATLAB raises an error. */
function callCount(scope: CallerScope, which: "$nargin" | "$nargout"): number {
  const v = scope.env.get(which);
  if (typeof v === "number") return v;
  throw builtinError(
    "MATLAB:narginout:NoStack",
    "You can only call nargin/nargout from within a MATLAB function."
  );
}

builtin({
  name: "nargin",
  help: {
    signatures: ["N = nargin", "N = nargin(FUN)"],
    description:
      "Number of input arguments of the current function, or the number declared by FUN (a function handle or name). A negative value -k means the last input is varargin and k-1 inputs precede it.",
  },
  nargin: [0, 1],
  scope: true,
  apply: (args, _nargout, { scope }) => {
    if (args.length === 0) return callCount(scope, "$nargin");
    const v = args[0];
    if (isRuntimeFunction(v)) {
      if (v.nargin !== undefined) return v.nargin;
      if (v.impl === "builtin" && v.name) return getIBuiltinNargin(v.name) ?? 0;
      return 0;
    }
    if (isRuntimeChar(v) || isRuntimeString(v)) {
      const name = toString(v);
      const n = scope.declaredNargin(name) ?? getIBuiltinNargin(name);
      if (n !== undefined) return n;
      throw notValidMfile();
    }
    throw narginoutBadInput();
  },
});

builtin({
  name: "nargout",
  help: {
    signatures: ["N = nargout", "N = nargout(FUN)"],
    description:
      "Number of output arguments of the current function, or the number declared by FUN (a function handle or name). A negative value -k means the last output is varargout and k-1 outputs precede it; anonymous functions report -1.",
  },
  nargin: [0, 1],
  scope: true,
  apply: (args, _nargout, { scope }) => {
    if (args.length === 0) return callCount(scope, "$nargout");
    const v = args[0];
    if (isRuntimeFunction(v)) {
      // Anonymous functions and handles to user functions carry their
      // count; other named handles are builtins.
      if (v.nargout !== undefined) return v.nargout;
      return (v.name ? getIBuiltinNargout(v.name) : undefined) ?? -1;
    }
    if (isRuntimeChar(v) || isRuntimeString(v)) {
      const name = toString(v);
      const n = scope.declaredNargout(name) ?? getIBuiltinNargout(name);
      if (n !== undefined) return n;
      throw notValidMfile();
    }
    throw narginoutBadInput();
  },
});

builtin({
  name: "narginchk",
  nargin: 2,
  nargout: 0,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    const n = callCount(scope, "$nargin");
    if (n < toNumber(args[0]))
      throw builtinError(
        "MATLAB:narginchk:notEnoughInputs",
        "Not enough input arguments."
      );
    if (n > toNumber(args[1]))
      throw builtinError(
        "MATLAB:narginchk:tooManyInputs",
        "Too many input arguments."
      );
  },
});

builtin({
  name: "nargoutchk",
  nargin: 2,
  nargout: 0,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    const n = callCount(scope, "$nargout");
    if (n < toNumber(args[0]))
      throw builtinError(
        "MATLAB:nargoutchk:notEnoughOutputs",
        "Not enough output arguments."
      );
    if (n > toNumber(args[1]))
      throw builtinError(
        "MATLAB:nargoutchk:tooManyOutputs",
        "Too many output arguments."
      );
  },
});

builtin({
  name: "inputname",
  nargin: 1,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    // inputname is only meaningful inside a function; $nargin marks one.
    const narginVal = scope.env.get("$nargin");
    if (typeof narginVal !== "number")
      throw new RuntimeError(
        "You can only call 'inputname' from within a MATLAB function."
      );
    const k = toNumber(args[0]);
    if (!Number.isInteger(k) || k < 1)
      throw new RuntimeError(
        "Argument number must be a positive integer scalar."
      );
    if (k > narginVal)
      throw new RuntimeError(
        "Argument number exceeds number of function input arguments."
      );
    const names = scope.inputNames;
    return RTV.char(names && k <= names.length ? names[k - 1] : "");
  },
});

/**
 * Time and system builtins: tic, toc, clock, etime, warning, computer, version,
 * ismac, ispc, isunix.
 */

import { isRuntimeChar, isRuntimeTensor } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toString } from "../../runtime/convert.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import { isText, unsupportedFormError } from "./text-args.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

// ── tic / toc ───────────────────────────────────────────────────────────

let ticTime = 0;

builtin({
  name: "tic",
  nargin: 0,
  apply: (_args, nargout) => {
    ticTime = performance.now();
    // Bare `tic;` starts the stopwatch and produces nothing: MATLAB does
    // not set `ans`.
    if (nargout === 0) return undefined;
    return RTV.num(ticTime / 1000);
  },
});

builtin({
  name: "toc",
  nargin: [0, 1],
  apply: (args, nargout, ctx) => {
    let startSeconds: number;
    if (args.length === 0) {
      startSeconds = ticTime / 1000;
    } else {
      const h = args[0];
      if (typeof h !== "number")
        throw new RuntimeError("toc: argument must be a tic handle");
      startSeconds = h;
    }
    const elapsed = performance.now() / 1000 - startSeconds;
    if (nargout === 0) {
      // Bare `toc;` prints and produces nothing: MATLAB does not set `ans`.
      ctx.rt.output(`Elapsed time is ${elapsed.toFixed(6)} seconds.\n`);
      return undefined;
    }
    return RTV.num(elapsed);
  },
});

// ── clock ───────────────────────────────────────────────────────────────

builtin({
  name: "clock",
  nargin: 0,
  apply: () => {
    const now = new Date();
    return RTV.tensor(
      allocFloat64Array([
        now.getFullYear(),
        now.getMonth() + 1,
        now.getDate(),
        now.getHours(),
        now.getMinutes(),
        now.getSeconds() + now.getMilliseconds() / 1000,
      ]),
      [1, 6]
    );
  },
});

// ── etime ───────────────────────────────────────────────────────────────

builtin({
  name: "etime",
  nargin: 2,
  mfile: true,
  apply: args => {
    const t1 = args[0];
    const t0 = args[1];
    if (!isRuntimeTensor(t1) || !isRuntimeTensor(t0))
      throw new RuntimeError("etime: arguments must be clock vectors");
    const toMs = (t: typeof t1) => {
      const d = new Date(
        t.data[0],
        t.data[1] - 1,
        t.data[2],
        t.data[3],
        t.data[4],
        Math.floor(t.data[5]),
        (t.data[5] % 1) * 1000
      );
      return d.getTime();
    };
    return RTV.num((toMs(t1) - toMs(t0)) / 1000);
  },
});

// ── version / computer ──────────────────────────────────────────────────

builtin({
  name: "version",
  nargin: [0, 1],
  apply: args => {
    if (args.length === 1) {
      const opt = isText(args[0]) ? toString(args[0]) : "";
      if (opt === "-release") return RTV.char("2023a");
      throw unsupportedFormError("version", `option '${opt}'`);
    }
    return RTV.char("9.14.0");
  },
});

// `ver` / `ver('product')` — version information. MATLAB returns a struct with
// fields Name, Version, Release, Date (a struct array for the no-argument
// form). numbl reports a single product struct matching `version` (9.14.0,
// R2023a); the named-product form just echoes the requested name.
builtin({
  name: "ver",
  nargin: [0, 1],
  mfile: true,
  apply: (args, nargout) => {
    const name =
      args.length >= 1 && isRuntimeChar(args[0]) ? toString(args[0]) : "MATLAB";
    if (nargout < 1) return undefined;
    return RTV.struct({
      Name: RTV.char(name),
      Version: RTV.char("9.14"),
      Release: RTV.char("(R2023a)"),
      Date: RTV.char("2023-02-09"),
    });
  },
});

function getComputerStrings(): { str: string; arch: string } {
  if (typeof process === "undefined") {
    return { str: "BROWSER", arch: "browser" };
  }
  const platform = process.platform;
  const cpuArch = process.arch;
  if (platform === "win32") return { str: "PCWIN64", arch: "win64" };
  if (platform === "darwin") {
    if (cpuArch === "arm64") return { str: "MACA64", arch: "maca64" };
    return { str: "MACI64", arch: "maci64" };
  }
  return { str: "GLNXA64", arch: "glnxa64" };
}

builtin({
  name: "computer",
  nargin: [0, 1],
  nargout: 3,
  apply: (args, nargout) => {
    const info = getComputerStrings();
    if (args.length === 1) {
      if (isText(args[0]) && toString(args[0]) === "arch")
        return RTV.char(info.arch);
      throw builtinError("MATLAB:badopt", "Unknown command option.");
    }
    const maxsize = 2 ** 48 - 1;
    const endian = RTV.char("L");
    if (nargout <= 1) return RTV.char(info.str);
    if (nargout === 2) return [RTV.char(info.str), RTV.num(maxsize)];
    return [RTV.char(info.str), RTV.num(maxsize), endian];
  },
});

function getMexExt(): string {
  return "numbl.js";
  // if (typeof process === "undefined") return "";
  // const platform = process.platform;
  // const cpuArch = process.arch;
  // if (platform === "win32") return "mexw64";
  // if (platform === "darwin")
  //   return cpuArch === "arm64" ? "mexmaca64" : "mexmaci64";
  // return "mexa64";
}

builtin({
  name: "mexext",
  nargin: [0, 1],
  apply: args => {
    if (args.length === 1)
      throw unsupportedFormError("mexext", "mexext('all')");
    return RTV.char(getMexExt());
  },
});

// ── Platform predicates ─────────────────────────────────────────────────

const _platform = typeof process !== "undefined" ? process.platform : "linux";
for (const [name, val] of [
  ["ismac", _platform === "darwin"],
  ["ispc", _platform === "win32"],
  ["isunix", _platform !== "win32"],
] as [string, boolean][]) {
  builtin({ name, nargin: 0, apply: () => RTV.logical(val) });
}

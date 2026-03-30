/**
 * Time and system builtins: tic, toc, clock, etime, warning, computer, version,
 * ismac, ispc, isunix.
 */

import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toString } from "../../runtime/convert.js";
import { defineBuiltin, registerIBuiltin } from "./types.js";

// ── tic / toc ───────────────────────────────────────────────────────────

let ticTime = 0;

defineBuiltin({
  name: "tic",
  cases: [
    {
      match: argTypes => (argTypes.length === 0 ? [{ kind: "number" }] : null),
      apply: () => {
        ticTime = performance.now();
        return RTV.num(ticTime / 1000);
      },
    },
  ],
});

defineBuiltin({
  name: "toc",
  cases: [
    {
      match: argTypes => (argTypes.length === 0 ? [{ kind: "number" }] : null),
      apply: () => RTV.num((performance.now() - ticTime) / 1000),
    },
  ],
});

// ── clock ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "clock",
  cases: [
    {
      match: argTypes =>
        argTypes.length === 0
          ? [{ kind: "tensor", isComplex: false, shape: [1, 6] }]
          : null,
      apply: () => {
        const now = new Date();
        return RTV.tensor(
          new FloatXArray([
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
    },
  ],
});

// ── etime ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "etime",
  cases: [
    {
      match: argTypes => (argTypes.length === 2 ? [{ kind: "number" }] : null),
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
    },
  ],
});

// ── version / computer ──────────────────────────────────────────────────

defineBuiltin({
  name: "version",
  cases: [
    {
      match: argTypes => (argTypes.length === 0 ? [{ kind: "char" }] : null),
      apply: () => RTV.char("9.14.0"),
    },
  ],
});

function getComputerStrings(): { str: string; arch: string } {
  const platform = typeof process !== "undefined" ? process.platform : "linux";
  const cpuArch = typeof process !== "undefined" ? process.arch : "x64";
  if (platform === "win32") return { str: "PCWIN64", arch: "win64" };
  if (platform === "darwin") {
    if (cpuArch === "arm64") return { str: "MACA64", arch: "maca64" };
    return { str: "MACI64", arch: "maci64" };
  }
  return { str: "GLNXA64", arch: "glnxa64" };
}

registerIBuiltin({
  name: "computer",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: (args, nargout) => {
      const info = getComputerStrings();
      if (args.length >= 1 && isRuntimeChar(args[0])) {
        const arg = toString(args[0]);
        if (arg === "arch") return RTV.char(info.arch);
        throw new RuntimeError(`computer: unknown argument '${arg}'`);
      }
      const maxsize = 2 ** 48 - 1;
      const endian = RTV.char("L");
      if (nargout <= 1) return RTV.char(info.str);
      if (nargout === 2) return [RTV.char(info.str), RTV.num(maxsize)];
      return [RTV.char(info.str), RTV.num(maxsize), endian];
    },
  }),
});

// ── Platform predicates ─────────────────────────────────────────────────

const _platform = typeof process !== "undefined" ? process.platform : "linux";
for (const [name, val] of [
  ["ismac", _platform === "darwin"],
  ["ispc", _platform === "win32"],
  ["isunix", _platform !== "win32"],
] as [string, boolean][]) {
  defineBuiltin({
    name,
    cases: [
      {
        match: argTypes =>
          argTypes.length === 0 ? [{ kind: "boolean" }] : null,
        apply: () => RTV.logical(val),
      },
    ],
  });
}

// getenv / setenv are implemented as special builtins in
// interpreterSpecialBuiltins.ts (they need the SystemAdapter from Runtime).

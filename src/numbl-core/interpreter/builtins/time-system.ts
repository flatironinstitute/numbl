/**
 * Time and system builtins: tic, toc, clock, etime, warning, computer, version,
 * ismac, ispc, isunix.
 */

import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
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

// ── warning ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "warning",
  resolve: () => {
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        if (
          args.length === 2 &&
          isRuntimeChar(args[0]) &&
          isRuntimeChar(args[1])
        ) {
          const state = toString(args[0]);
          if (state === "on" || state === "off") {
            return RTV.struct(
              new Map<string, RuntimeValue>([
                ["state", RTV.char("on")],
                ["identifier", args[1]],
              ])
            );
          }
        }
        return RTV.num(0);
      },
    };
  },
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

defineBuiltin({
  name: "computer",
  cases: [
    {
      match: argTypes => (argTypes.length === 0 ? [{ kind: "char" }] : null),
      apply: () => RTV.char("GLNXA64"),
    },
  ],
});

// ── Platform predicates ─────────────────────────────────────────────────

for (const [name, val] of [
  ["ismac", false],
  ["ispc", false],
  ["isunix", true],
] as const) {
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

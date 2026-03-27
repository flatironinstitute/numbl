/**
 * Misc builtins: substruct, odeset, peaks, now, datestr, lastwarn, verLessThan,
 * setappdata, getappdata, rmappdata, isappdata.
 */

import {
  FloatXArray,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeFunction,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeStruct } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber } from "../../runtime/convert.js";
import { toString } from "../../runtime/convert.js";
import { registerIBuiltin, getIBuiltin } from "./types.js";
import {
  mAdd,
  mSub,
  mElemMul,
  mElemDiv,
  mMul,
  mDiv,
  mLeftDiv,
  mPow,
  mElemLeftDiv,
  mEqual,
  mNotEqual,
  mLess,
  mLessEqual,
  mGreater,
  mGreaterEqual,
  mNeg,
} from "../../helpers/arithmetic.js";

// ── substruct ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "substruct",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      if (args.length < 2 || args.length % 2 !== 0)
        throw new RuntimeError(
          "substruct requires pairs of (type, subs) arguments"
        );
      const elements: RuntimeStruct[] = [];
      for (let i = 0; i < args.length; i += 2) {
        const typeArg = args[i];
        const subsArg = args[i + 1];
        if (!isRuntimeChar(typeArg) && !isRuntimeString(typeArg))
          throw new RuntimeError("substruct: type must be a string");
        const typeStr = isRuntimeChar(typeArg)
          ? typeArg.value
          : isRuntimeString(typeArg)
            ? typeArg
            : "";
        if (typeStr !== "." && typeStr !== "()" && typeStr !== "{}")
          throw new RuntimeError(
            `substruct: type must be '.', '()', or '{}', got '${typeStr}'`
          );
        let subs: RuntimeValue;
        if (typeStr === ".") {
          subs = subsArg;
        } else {
          subs = isRuntimeCell(subsArg) ? subsArg : RTV.cell([subsArg], [1, 1]);
        }
        elements.push(RTV.struct({ type: RTV.char(typeStr), subs }));
      }
      return RTV.structArray(["type", "subs"], elements);
    },
  }),
});

// ── odeset (dummy) ───────────────────────────────────────────────────────

registerIBuiltin({
  name: "odeset",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: () => RTV.struct({}),
  }),
});

// ── peaks ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "peaks",
  resolve: () => ({
    outputTypes: [{ kind: "tensor", isComplex: false }],
    apply: args => {
      const n =
        args.length > 0
          ? isRuntimeNumber(args[0])
            ? (args[0] as number)
            : toNumber(args[0])
          : 49;
      const data = new FloatXArray(n * n);
      for (let j = 0; j < n; j++) {
        const y = -3 + (6 * j) / (n - 1);
        for (let i = 0; i < n; i++) {
          const x = -3 + (6 * i) / (n - 1);
          const x2 = x * x;
          const y2 = y * y;
          const z =
            3 * Math.pow(1 - x, 2) * Math.exp(-x2 - Math.pow(y + 1, 2)) -
            10 * (x / 5 - x * x2 - Math.pow(y, 5)) * Math.exp(-x2 - y2) -
            (1 / 3) * Math.exp(-Math.pow(x + 1, 2) - y2);
          data[j * n + i] = z;
        }
      }
      return RTV.tensor(data, [n, n]);
    },
  }),
});

// ── now ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "now",
  resolve: () => ({
    outputTypes: [{ kind: "number" }],
    apply: () => {
      // MATLAB serial date number: days since Jan 0, year 0000
      // Jan 1, 1970 = MATLAB day 719529
      const matlabEpoch = 719529;
      return matlabEpoch + Date.now() / 86400000;
    },
  }),
});

// ── datestr ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "datestr",
  resolve: () => ({
    outputTypes: [{ kind: "char" }],
    apply: args => {
      if (args.length < 1)
        throw new RuntimeError("datestr requires at least 1 argument");
      const datenum = toNumber(args[0]);
      // Convert MATLAB serial date to JS Date
      const matlabEpoch = 719529;
      const ms = (datenum - matlabEpoch) * 86400000;
      const d = new Date(ms);
      const months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
      ];
      const day = String(d.getUTCDate()).padStart(2, "0");
      const mon = months[d.getUTCMonth()];
      const year = d.getUTCFullYear();
      const hh = String(d.getUTCHours()).padStart(2, "0");
      const mm = String(d.getUTCMinutes()).padStart(2, "0");
      const ss = String(d.getUTCSeconds()).padStart(2, "0");
      return RTV.char(`${day}-${mon}-${year} ${hh}:${mm}:${ss}`);
    },
  }),
});

// ── lastwarn ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "lastwarn",
  resolve: () => ({
    outputTypes: [{ kind: "char" }],
    apply: () => RTV.char(""),
  }),
});

// ── verLessThan ──────────────────────────────────────────────────────────

registerIBuiltin({
  name: "verLessThan",
  resolve: () => ({
    outputTypes: [{ kind: "boolean" }],
    apply: args => {
      if (args.length !== 2)
        throw new RuntimeError("verLessThan requires 2 arguments");
      return false; // numbl aims to be like modern MATLAB
    },
  }),
});

// ── clear / clc / clf stubs ──────────────────────────────────────────────

for (const name of ["clear", "clc", "clf"]) {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [],
      apply: () => 0,
    }),
  });
}

// ── nargin ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "nargin",
  resolve: argTypes => {
    if (argTypes.length > 1) return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => {
        if (args.length === 1 && isRuntimeFunction(args[0])) {
          const handle = args[0];
          if (handle.nargin !== undefined) return handle.nargin;
          // Infer nargin for IBuiltins by probing resolve with 1..N unknown args
          if (handle.impl === "builtin" && handle.name) {
            const ib = getIBuiltin(handle.name);
            if (ib) {
              for (let n = 1; n <= 10; n++) {
                const testArgs = Array.from({ length: n }, () => ({
                  kind: "number" as const,
                }));
                if (ib.resolve(testArgs, 1)) return n;
              }
            }
          }
          return 0;
        }
        return 0;
      },
    };
  },
});

// ── true/false constructors ─────────────────────────────────────────────

function parseShapeArgs(args: RuntimeValue[]): number[] {
  const dims: number[] = [];
  for (const a of args) {
    if (typeof a === "number") dims.push(Math.round(a));
    else if (isRuntimeTensor(a)) {
      for (let i = 0; i < a.data.length; i++) dims.push(Math.round(a.data[i]));
    } else dims.push(Math.round(toNumber(a)));
  }
  return dims;
}

for (const [name, fillVal] of [
  ["true", 1],
  ["false", 0],
] as const) {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        if (args.length === 0) return RTV.logical(fillVal === 1);
        const shape = parseShapeArgs(args);
        const rows = shape[0];
        const cols = shape[1] ?? rows;
        const t = RTV.tensor(
          fillVal
            ? new FloatXArray(rows * cols).fill(1)
            : new FloatXArray(rows * cols),
          [rows, cols]
        );
        t._isLogical = true;
        return t;
      },
    }),
  });
}

// ── Operator-name builtins ──────────────────────────────────────────────

const opMap: [string, (a: RuntimeValue, b: RuntimeValue) => RuntimeValue][] = [
  ["plus", mAdd],
  ["minus", mSub],
  ["times", mElemMul],
  ["rdivide", mElemDiv],
  ["mtimes", mMul],
  ["mrdivide", mDiv],
  ["mldivide", mLeftDiv],
  ["mpower", mPow],
  ["ldivide", mElemLeftDiv],
  ["eq", mEqual],
  ["ne", mNotEqual],
  ["lt", mLess],
  ["le", mLessEqual],
  ["gt", mGreater],
  ["ge", mGreaterEqual],
];

for (const [name, op] of opMap) {
  registerIBuiltin({
    name,
    resolve: argTypes => {
      if (argTypes.length !== 2) return null;
      return {
        outputTypes: [{ kind: "unknown" }],
        apply: args => op(args[0], args[1]),
      };
    },
  });
}

registerIBuiltin({
  name: "uminus",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => mNeg(args[0]),
    };
  },
});

registerIBuiltin({
  name: "uplus",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => args[0],
    };
  },
});

// ── Dummy/placeholder builtins ──────────────────────────────────────────

// Dummy handle functions — return a dummy handle struct
for (const name of ["groot", "gcf", "gca", "shg", "newplot", "caxis"]) {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "unknown" }],
      apply: () => RTV.dummyHandle(),
    }),
  });
}

// get(handle, propName) — return a dummy handle
registerIBuiltin({
  name: "get",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: () => RTV.dummyHandle(),
  }),
});

// set(handle, ...) — void no-op
registerIBuiltin({
  name: "set",
  resolve: () => ({
    outputTypes: [],
    apply: () => 0,
  }),
});

// Graphics stubs that return nothing
for (const name of [
  "figure",
  "hold",
  "grid",
  "close",
  "title",
  "xlabel",
  "ylabel",
  "shading",
  "subplot",
  "legend",
  "sgtitle",
  "zlabel",
  "colorbar",
  "addpath",
  "dir",
]) {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [],
      apply: () => 0,
    }),
  });
}

// ishold — returns false
registerIBuiltin({
  name: "ishold",
  resolve: () => ({
    outputTypes: [{ kind: "boolean" }],
    apply: () => RTV.logical(false),
  }),
});

// pwd — return empty string
registerIBuiltin({
  name: "pwd",
  resolve: () => ({
    outputTypes: [{ kind: "char" }],
    apply: () => RTV.char(""),
  }),
});

// mfilename — handled as a special builtin (needs runtime access to current file)

// xlim/ylim — return empty array
for (const name of ["xlim", "ylim"]) {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "tensor", isComplex: false }],
      apply: () => RTV.tensor(new FloatXArray(0), [0, 0]),
    }),
  });
}

// listfonts — return empty cell
registerIBuiltin({
  name: "listfonts",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: () => RTV.cell([], [0, 0]),
  }),
});

// Colormap name functions — return the name as a char
for (const cm of [
  "parula",
  "jet",
  "hsv",
  "hot",
  "cool",
  "spring",
  "summer",
  "autumn",
  "winter",
  "gray",
  "bone",
  "copper",
  "pink",
]) {
  registerIBuiltin({
    name: cm,
    resolve: () => ({
      outputTypes: [{ kind: "char" }],
      apply: () => RTV.char(cm),
    }),
  });
}

// ── setappdata / getappdata / rmappdata / isappdata ────────────────────

// Persistent storage keyed by object handle (number).
// Survives clear all because it's module-level state.
const appdataStore = new Map<number, Map<string, RuntimeValue>>();

/** Clear all appdata — called at the start of each executeCode to isolate runs. */
export function resetAppdataStore(): void {
  appdataStore.clear();
}

function getObjKey(obj: RuntimeValue): number {
  if (typeof obj === "number") return obj;
  if (typeof obj === "boolean") return obj ? 1 : 0;
  throw new RuntimeError(
    "setappdata/getappdata: object must be a numeric handle"
  );
}

function ensureBucket(key: number): Map<string, RuntimeValue> {
  let bucket = appdataStore.get(key);
  if (!bucket) {
    bucket = new Map();
    appdataStore.set(key, bucket);
  }
  return bucket;
}

registerIBuiltin({
  name: "setappdata",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    return {
      outputTypes: [],
      apply: args => {
        const key = getObjKey(args[0]);
        const name = toString(args[1]);
        const bucket = ensureBucket(key);
        bucket.set(name, args[2]);
        return [] as unknown as RuntimeValue;
      },
    };
  },
});

registerIBuiltin({
  name: "getappdata",
  resolve: argTypes => {
    if (argTypes.length === 2) {
      return {
        outputTypes: [{ kind: "unknown" }],
        apply: args => {
          const key = getObjKey(args[0]);
          const name = toString(args[1]);
          const bucket = appdataStore.get(key);
          if (!bucket || !bucket.has(name))
            return RTV.tensor(new FloatXArray(0), [0, 0]);
          return bucket.get(name)!;
        },
      };
    }
    if (argTypes.length === 1) {
      return {
        outputTypes: [{ kind: "struct" }],
        apply: args => {
          const key = getObjKey(args[0]);
          const bucket = appdataStore.get(key);
          if (!bucket || bucket.size === 0) return RTV.struct({});
          const fields: Record<string, RuntimeValue> = {};
          for (const [k, v] of bucket) fields[k] = v;
          return RTV.struct(fields);
        },
      };
    }
    return null;
  },
});

registerIBuiltin({
  name: "rmappdata",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [],
      apply: args => {
        const key = getObjKey(args[0]);
        const name = toString(args[1]);
        const bucket = appdataStore.get(key);
        if (bucket) bucket.delete(name);
        return [] as unknown as RuntimeValue;
      },
    };
  },
});

registerIBuiltin({
  name: "isappdata",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        const key = getObjKey(args[0]);
        const name = toString(args[1]);
        const bucket = appdataStore.get(key);
        return !!(bucket && bucket.has(name));
      },
    };
  },
});

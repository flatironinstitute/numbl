/**
 * Misc builtins: substruct, odeset, peaks, now, datestr, lastwarn, verLessThan,
 * setappdata, getappdata, rmappdata, isappdata.
 */
import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeDummyHandle,
  isRuntimeFunction,
  isRuntimeGraphicsHandle,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeStruct } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";

import { toString } from "../../runtime/convert.js";
import { getIBuiltinNargin, getIBuiltin } from "./types.js";
import { builtin } from "./builtin.js";
import { builtinError, realScalarArg } from "./args.js";
import { tryDatetimeDurationBinop } from "./datetime.js";
import { tryDecompositionDivide } from "./sparse-solvers.js";
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
import { allocFloat64Array } from "../../runtime/alloc.js";
import { PARULA_ANCHORS } from "../../plotting/colormaps.js";
import { incref, decref, getCurrentRuntime } from "../../runtime/refcount.js";
import { stripClassNameArgs } from "./array-construction.js";
import { parseShapeArgs } from "../../helpers/shape-utils.js";

// ── substruct ────────────────────────────────────────────────────────────

builtin({
  name: "substruct",
  mfile: true,
  apply: args => {
    if (args.length < 2)
      throw builtinError(
        "MATLAB:substruct:nargin",
        "SUBSTRUCT takes at least two arguments."
      );
    if (args.length % 2 !== 0)
      throw builtinError(
        "MATLAB:substruct:narginOdd",
        "SUBSTRUCT takes an even number of arguments."
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
});

// ── odeset ───────────────────────────────────────────────────────────────

builtin({
  name: "odeset",
  mfile: true,
  apply: args => {
    const fields = new Map<string, RuntimeValue>();
    if (args.length % 2 !== 0)
      throw new RuntimeError("odeset: expected name-value pairs");
    for (let i = 0; i < args.length; i += 2) {
      const key = args[i];
      if (!isRuntimeChar(key) && !isRuntimeString(key))
        throw new RuntimeError("odeset: option name must be a string");
      const name = isRuntimeChar(key) ? key.value : (key as string);
      fields.set(name, args[i + 1]);
    }
    return RTV.struct(Object.fromEntries(fields));
  },
});

// ── weboptions ──────────────────────────────────────────────────────────

const WEBOPTIONS_DEFAULTS: Record<string, RuntimeValue> = {
  CharacterEncoding: RTV.char("auto"),
  UserAgent: RTV.char("numbl"),
  Timeout: 5,
  Username: RTV.char(""),
  Password: RTV.char(""),
  KeyName: RTV.char(""),
  KeyValue: RTV.char(""),
  ContentType: RTV.char("auto"),
  MediaType: RTV.char("auto"),
  RequestMethod: RTV.char("auto"),
  ArrayFormat: RTV.char("csv"),
  CertificateFilename: RTV.char("default"),
};

builtin({
  name: "weboptions",
  mfile: true,
  apply: args => {
    const fields = new Map<string, RuntimeValue>();
    // Start with defaults
    for (const [k, v] of Object.entries(WEBOPTIONS_DEFAULTS)) {
      fields.set(k, v);
    }
    if (args.length % 2 !== 0)
      throw new RuntimeError("weboptions: expected name-value pairs");
    for (let i = 0; i < args.length; i += 2) {
      const key = args[i];
      if (!isRuntimeChar(key) && !isRuntimeString(key))
        throw new RuntimeError("weboptions: option name must be a string");
      const name = isRuntimeChar(key) ? key.value : (key as string);
      if (!(name in WEBOPTIONS_DEFAULTS))
        throw new RuntimeError(`weboptions: unknown option '${name}'`);
      fields.set(name, args[i + 1]);
    }
    return RTV.struct(Object.fromEntries(fields));
  },
});

// ── odeget ───────────────────────────────────────────────────────────────

builtin({
  name: "odeget",
  // odeget(options, name, default): the default is returned when the
  // option is missing or empty.
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const fallback = args.length >= 3 ? args[2] : undefined;
    const opts = args[0];
    const key = args[1];
    if (!isRuntimeChar(key) && !isRuntimeString(key))
      throw new RuntimeError("odeget: option name must be a string");
    const name = isRuntimeChar(key) ? key.value : (key as string);
    if (
      opts &&
      typeof opts === "object" &&
      "kind" in opts &&
      opts.kind === "struct"
    ) {
      const val = (
        opts as { kind: "struct"; fields: Map<string, RuntimeValue> }
      ).fields.get(name);
      if (val !== undefined && !(fallback !== undefined && isEmptyValue(val)))
        return val;
    }
    // Return the default, or empty if none (matches MATLAB)
    return fallback ?? RTV.tensor(allocFloat64Array(0), [0, 0]);
  },
});

/** True for an empty array value ([] , '', {}). */
function isEmptyValue(v: RuntimeValue): boolean {
  if (isRuntimeTensor(v)) return v.data.length === 0;
  if (isRuntimeChar(v)) return v.value.length === 0;
  if (isRuntimeCell(v)) return v.data.length === 0;
  return false;
}

// ── peaks ────────────────────────────────────────────────────────────────

builtin({
  name: "peaks",
  nargin: [0, 2],
  nargout: 3,
  mfile: true,
  apply: (args, nargout) => {
    if (args.length > 1)
      throw new RuntimeError("peaks: the peaks(X, Y) form is not supported");
    if (
      args.length === 1 &&
      !(
        isRuntimeNumber(args[0]) ||
        isRuntimeLogical(args[0]) ||
        (isRuntimeTensor(args[0]) && args[0].data.length === 1)
      )
    )
      throw new RuntimeError("peaks: the peaks(V) form is not supported");
    const n = args.length > 0 ? realScalarArg(args[0]) : 49;
    const data = allocFloat64Array(n * n);
    // X(:, j) varies along columns, Y(i, :) along rows (meshgrid).
    const X = allocFloat64Array(n * n);
    const Y = allocFloat64Array(n * n);
    for (let j = 0; j < n; j++) {
      const x = -3 + (6 * j) / (n - 1);
      for (let i = 0; i < n; i++) {
        const y = -3 + (6 * i) / (n - 1);
        X[j * n + i] = x;
        Y[j * n + i] = y;
        const x2 = x * x;
        const y2 = y * y;
        const z =
          3 * Math.pow(1 - x, 2) * Math.exp(-x2 - Math.pow(y + 1, 2)) -
          10 * (x / 5 - x * x2 - Math.pow(y, 5)) * Math.exp(-x2 - y2) -
          (1 / 3) * Math.exp(-Math.pow(x + 1, 2) - y2);
        data[j * n + i] = z;
      }
    }
    const Z = RTV.tensor(data, [n, n]);
    if (nargout <= 1) return Z;
    return [RTV.tensor(X, [n, n]), RTV.tensor(Y, [n, n]), Z].slice(0, nargout);
  },
});

// ── now ──────────────────────────────────────────────────────────────────

builtin({
  name: "now",
  nargin: 0,
  mfile: true,
  apply: () => {
    // MATLAB serial date number: days since Jan 0, year 0000
    // Jan 1, 1970 = MATLAB day 719529
    const matlabEpoch = 719529;
    return matlabEpoch + Date.now() / 86400000;
  },
});

// ── datestr ──────────────────────────────────────────────────────────────

builtin({
  name: "datestr",
  nargin: [1, Infinity],
  mfile: true,
  apply: args => {
    if (args.length > 1)
      throw new RuntimeError(
        "datestr: format and pivot-year arguments are not supported"
      );
    const datenum = realScalarArg(args[0]);
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
});

// ── lastwarn ─────────────────────────────────────────────────────────────

builtin({
  name: "lastwarn",
  // lastwarn(msg [, id]) sets the last warning; numbl does not track
  // warnings, so it always reports an empty message and identifier.
  nargin: [0, 2],
  nargout: 2,
  apply: (args, nargout) => {
    if (args.length > 0 && nargout === 0) return;
    if (nargout <= 1) return RTV.char("");
    return [RTV.char(""), RTV.char("")];
  },
});

// ── verLessThan ──────────────────────────────────────────────────────────

builtin({
  name: "verLessThan",
  nargin: 2,
  mfile: true,
  apply: () => false, // numbl aims to be like modern MATLAB
});

// ── usejava ──────────────────────────────────────────────────────────────

builtin({
  name: "usejava",
  nargin: 1,
  mfile: true,
  apply: () => false,
});

// ── clear / clc stubs ────────────────────────────────────────────────────
// `clear` is handled by interpreterSpecialBuiltins (it needs env access);
// the no-op IBuiltin only covers reserved-word forms (`clear functions`,
// `clear global`, …) which fall through from the special handler.

builtin({ name: "clear", nargout: 0, apply: () => {} });
builtin({ name: "clc", nargin: 0, nargout: 0, apply: () => {} });

// ── nargin ──────────────────────────────────────────────────────────────

const narginoutBadInput = () =>
  builtinError(
    "MATLAB:narginout:BadInput",
    "Argument must be either a character vector, string scalar, or a function handle."
  );
const notValidMfile = () =>
  builtinError("MATLAB:narginout:notValidMfile", "Not a valid MATLAB file.");

// nargin / nargout with no argument (inside a function) and nargin('name')
// for user functions are interpreter specials, which need the caller's
// scope. These builtins cover nargin(@f) and the remaining forms.
builtin({
  name: "nargin",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (isRuntimeFunction(v)) {
      if (v.nargin !== undefined) return v.nargin;
      if (v.impl === "builtin" && v.name) {
        const n = getIBuiltinNargin(v.name);
        if (n !== undefined) return n;
      }
      return 0;
    }
    if (isRuntimeChar(v) || isRuntimeString(v)) {
      const n = getIBuiltinNargin(toString(v));
      if (n !== undefined) return n;
      throw notValidMfile();
    }
    throw narginoutBadInput();
  },
});

/** MATLAB's nargout for a builtin: its maximum output count, or -1 when
 *  unbounded. Legacy builtins without arity metadata report -1. */
function builtinNargout(name: string): number | undefined {
  const ib = getIBuiltin(name);
  if (!ib) return undefined;
  if (!ib.direct) return -1;
  const max = ib.direct.nargout[1];
  return max === Infinity ? -1 : max;
}

builtin({
  name: "nargout",
  help: {
    signatures: ["N = nargout", "N = nargout(FUN)"],
    description:
      "Number of output arguments of the current function, or the number declared by FUN (a function handle or name). A negative value -k means the last output is varargout and k-1 outputs precede it; anonymous functions report -1.",
  },
  nargin: 1,
  apply: (args, _nargout, ctx) => {
    const v = args[0];
    if (isRuntimeFunction(v)) {
      // Anonymous functions and handles to user functions carry their
      // count; other named handles are builtins.
      if (v.nargout !== undefined) return v.nargout;
      return (v.name ? builtinNargout(v.name) : undefined) ?? -1;
    }
    if (isRuntimeChar(v) || isRuntimeString(v)) {
      const name = toString(v);
      const n = ctx.rt.declaredNargout?.(name) ?? builtinNargout(name);
      if (n !== undefined) return n;
      throw notValidMfile();
    }
    throw narginoutBadInput();
  },
});

// ── true/false constructors ─────────────────────────────────────────────

for (const [name, fillVal] of [
  ["true", 1],
  ["false", 0],
] as const) {
  builtin({
    name,
    apply: rawArgs => {
      // true/false accept a trailing 'logical' class spec; numbl ignores it.
      const args = stripClassNameArgs(rawArgs);
      if (args.length === 0) return RTV.logical(fillVal === 1);
      let shape = parseShapeArgs(args);
      if (shape.length === 1) shape = [shape[0], shape[0]];
      const data = allocFloat64Array(shape.reduce((a, b) => a * b, 1));
      if (fillVal) data.fill(1);
      const t = RTV.tensor(data, shape);
      t._isLogical = true;
      return t;
    },
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
  builtin({
    name,
    nargin: 2,
    // mpower is an M-file in MATLAB.
    mfile: name === "mpower",
    apply: args => {
      const dt = tryDatetimeDurationBinop(name, args[0], args[1]);
      if (dt !== undefined) return dt;
      if (name === "mldivide" || name === "mrdivide") {
        const dec = tryDecompositionDivide(name, args[0], args[1]);
        if (dec !== undefined) return dec;
      }
      return op(args[0], args[1]);
    },
  });
}

builtin({ name: "uminus", nargin: 1, apply: args => mNeg(args[0]) });

builtin({ name: "uplus", nargin: 1, apply: args => args[0] });

// ── Dummy/placeholder builtins ──────────────────────────────────────────

// Dummy handle functions — return a dummy handle struct
for (const name of ["groot", "gcf", "gca", "newplot"])
  builtin({ name, apply: () => RTV.dummyHandle() });

// shg brings the current figure forward; it has no output.
builtin({ name: "shg", nargin: 0, nargout: 0, mfile: true, apply: () => {} });

// ishandle / ishghandle / isgraphics — test whether the argument is a graphics
// handle. numbl represents handles either as handle objects (gcf/gca/...) or as
// numeric figure handles (`figure` returns a figure number). A numeric value is
// treated as a valid handle when it is a finite, non-negative real (0 is the
// root/groot handle, positive integers are figure numbers). Operates
// element-wise on tensors, returning a logical of the same size.
function ishandleScalar(x: number): boolean {
  return Number.isFinite(x) && x >= 0;
}
for (const name of ["ishandle", "ishghandle", "isgraphics"]) {
  builtin({
    name,
    // ishghandle(h, type) and isgraphics(h, type) also take a type name.
    nargin: name === "ishandle" ? 1 : [1, 2],
    apply: args => {
      const v = args[0];
      if (isRuntimeDummyHandle(v) || isRuntimeGraphicsHandle(v))
        return RTV.logical(true);
      if (isRuntimeNumber(v)) return RTV.logical(ishandleScalar(v));
      if (isRuntimeLogical(v)) return RTV.logical(ishandleScalar(v ? 1 : 0));
      if (isRuntimeTensor(v)) {
        const out = allocFloat64Array(v.data.length);
        for (let i = 0; i < v.data.length; i++)
          out[i] = ishandleScalar(v.data[i]) ? 1 : 0;
        const t = RTV.tensor(out, [...v.shape]);
        t._isLogical = true;
        return t;
      }
      return RTV.logical(false);
    },
  });
}

// camlight — numbl's plot viewer has no lighting model, so this is a
// stub. It accepts every argument form (position, az/el, style, and a
// leading light/axes handle) and returns a placeholder Light handle when
// one is requested (`cl = camlight(...)`). As a statement it produces no
// output, so `ans` is left untouched.
builtin({
  name: "camlight",
  mfile: true,
  apply: (_args, nargout) => (nargout >= 1 ? RTV.dummyHandle() : undefined),
});

// waitbar — numbl is headless, so the progress dialog renders nothing. All
// forms (waitbar(x), waitbar(x,msg), waitbar(x,h), waitbar(x,h,msg),
// waitbar(...,'Name',value)) are accepted; a placeholder figure handle is
// returned when one is requested (`h = waitbar(...)`). `close(h)` already
// ignores its argument, so the handle can be passed straight back.
builtin({
  name: "waitbar",
  nargin: [1, Infinity],
  mfile: true,
  apply: (_args, nargout) => (nargout >= 1 ? RTV.dummyHandle() : undefined),
});

// get(handle, propName) — return property value or dummy handle
builtin({
  name: "get",
  apply: args => {
    // get(h, 'colormap') → return default 64×3 parula matrix
    if (args.length >= 2) {
      const prop = args[1];
      const propStr = isRuntimeChar(prop)
        ? prop.value
        : isRuntimeString(prop)
          ? prop
          : "";
      if (propStr.toLowerCase() === "colormap") return parulaColormap(64);
    }
    return RTV.dummyHandle();
  },
});

// `set(handle, 'PropName', value, ...)` is registered as a runtime-closured
// special builtin in plotting/builtins.ts (it needs to push `update_trace`
// instructions so live edits reach the serialized plot viewer).

// pathdef — no-op, returns empty string
builtin({
  name: "pathdef",
  nargin: 0,
  mfile: true,
  apply: () => RTV.char(""),
});

// isnumbl — returns true (detect numbl runtime)
builtin({ name: "isnumbl", nargin: 0, apply: () => true });

// filesep — platform file separator
builtin({ name: "filesep", nargin: 0, apply: () => RTV.char("/") });

// pwd / cd / mfilename are in path-env.ts (they need the Runtime).

// xlim/ylim — return empty array
for (const name of ["xlim", "ylim"])
  builtin({
    name,
    mfile: true,
    apply: () => RTV.tensor(allocFloat64Array(0), [0, 0]),
  });

// listfonts — return empty cell
builtin({
  name: "listfonts",
  nargin: [0, 1],
  mfile: true,
  apply: () => RTV.cell([], [0, 0]),
});

// Colormap name functions — return the name as a char (so `colormap(jet)`
// works). `parula` additionally supports `parula(m)` returning an m×3 RGB
// matrix, matching MATLAB, for callers that index into the colormap.
for (const cm of [
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
  builtin({
    name: cm,
    nargin: [0, 1],
    mfile: true,
    apply: () => RTV.char(cm),
  });
}

/** Build an `m × 3` parula colormap (column-major) by interpolating the
 *  anchor table. */
function parulaColormap(m: number): RuntimeValue {
  const data = allocFloat64Array(m * 3);
  const last = PARULA_ANCHORS.length - 1;
  for (let i = 0; i < m; i++) {
    const t = m <= 1 ? 0 : (i / (m - 1)) * last;
    const k = Math.min(last - 1, Math.floor(t));
    const f = t - k;
    const a = PARULA_ANCHORS[k];
    const b = PARULA_ANCHORS[Math.min(last, k + 1)];
    for (let c = 0; c < 3; c++) {
      data[c * m + i] = a[c] + f * (b[c] - a[c]); // column-major [m, 3]
    }
  }
  return RTV.tensor(data, [m, 3]);
}

builtin({
  name: "parula",
  nargin: [0, 1],
  mfile: true,
  apply: args => {
    if (
      args.length >= 1 &&
      (isRuntimeNumber(args[0]) || isRuntimeTensor(args[0]))
    ) {
      return parulaColormap(Math.max(0, Math.round(realScalarArg(args[0]))));
    }
    return RTV.char("parula");
  },
});

// ── setappdata / getappdata / rmappdata / isappdata ────────────────────

// Persistent storage keyed by object handle (number).
// Survives clear all because it's module-level state.
const appdataStore = new Map<number, Map<string, RuntimeValue>>();

/** Clear all appdata — called at the start of each executeCode to isolate
 *  runs. Decrefs every stored value so the previous run's containers can
 *  hit rc=0 cleanly; if the previous runtime is gone (test harness), we
 *  just drop the JS references and let GC reclaim them. */
export function resetAppdataStore(): void {
  const rt = getCurrentRuntime();
  if (rt) {
    for (const bucket of appdataStore.values()) {
      for (const v of bucket.values()) decref(rt, v);
    }
  }
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

builtin({
  name: "setappdata",
  nargin: 3,
  nargout: 0,
  apply: (args, _nargout, ctx) => {
    const key = getObjKey(args[0]);
    const name = toString(args[1]);
    const bucket = ensureBucket(key);
    // Manage refcount: bucket takes ownership of the value, dropping
    // any prior occupant. Without this the bucket holds a dangling
    // reference once the caller's local goes out of scope, and a
    // later getappdata + bind would underflow on the value's
    // already-zero refcount.
    const old = bucket.get(name);
    incref(args[2]);
    bucket.set(name, args[2]);
    if (old !== undefined) decref(ctx.rt, old);
  },
});

builtin({
  name: "getappdata",
  nargin: [1, 2],
  apply: args => {
    const key = getObjKey(args[0]);
    const bucket = appdataStore.get(key);
    if (args.length === 2) {
      const name = toString(args[1]);
      if (!bucket || !bucket.has(name))
        return RTV.tensor(allocFloat64Array(0), [0, 0]);
      return bucket.get(name)!;
    }
    if (!bucket || bucket.size === 0) return RTV.struct({});
    const fields: Record<string, RuntimeValue> = {};
    for (const [k, v] of bucket) fields[k] = v;
    return RTV.struct(fields);
  },
});

builtin({
  name: "rmappdata",
  nargin: 2,
  nargout: 0,
  apply: (args, _nargout, ctx) => {
    const key = getObjKey(args[0]);
    const name = toString(args[1]);
    const bucket = appdataStore.get(key);
    if (bucket) {
      const old = bucket.get(name);
      if (bucket.delete(name) && old !== undefined) decref(ctx.rt, old);
    }
  },
});

builtin({
  name: "isappdata",
  nargin: 2,
  apply: args => {
    const key = getObjKey(args[0]);
    const name = toString(args[1]);
    const bucket = appdataStore.get(key);
    return !!(bucket && bucket.has(name));
  },
});

// ── jsondecode ──────────────────────────────────────────────────────────

function makeValidFieldName(name: string): string {
  // Replace invalid characters with underscores
  let valid = name.replace(/[^a-zA-Z0-9_]/g, "_");
  // Ensure it starts with a letter
  if (!/^[a-zA-Z]/.test(valid)) valid = "x" + valid;
  // Truncate to reasonable length
  if (valid.length > 63) valid = valid.slice(0, 63);
  return valid;
}

export function convertJsonValue(val: unknown): RuntimeValue {
  if (val === null) return NaN;
  if (typeof val === "boolean") return val;
  if (typeof val === "number") return val;
  if (typeof val === "string") return RTV.char(val);
  if (Array.isArray(val)) return convertJsonArray(val);
  if (typeof val === "object")
    return convertJsonObject(val as Record<string, unknown>);
  return NaN;
}

function convertJsonObject(obj: Record<string, unknown>): RuntimeValue {
  const fields: Record<string, RuntimeValue> = {};
  for (const [key, value] of Object.entries(obj)) {
    fields[makeValidFieldName(key)] = convertJsonValue(value);
  }
  return RTV.struct(fields);
}

function convertJsonArray(arr: unknown[]): RuntimeValue {
  if (arr.length === 0) return RTV.tensor(allocFloat64Array(0), [0, 0]);

  // Classify elements
  let allBooleans = true;
  let allNumbers = true;
  let allStrings = true;
  let allObjects = true;

  for (const el of arr) {
    if (typeof el !== "boolean" && el !== null) allBooleans = false;
    if (typeof el !== "number" && el !== null) allNumbers = false;
    if (typeof el !== "string") allStrings = false;
    if (typeof el !== "object" || el === null || Array.isArray(el))
      allObjects = false;
  }

  // Booleans also count as numbers in JSON, so prioritize booleans check
  // but only if ALL are booleans (not mixed with numbers)
  if (allBooleans && !allNumbers) {
    // Array of booleans → logical column vector
    const data = allocFloat64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i] === null ? 0 : arr[i] ? 1 : 0;
    }
    const t = RTV.tensor(data, [arr.length, 1]);
    t._isLogical = true;
    return t;
  }

  if (allNumbers) {
    // Array of numbers → double column vector (null → NaN)
    const data = allocFloat64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i] === null ? NaN : (arr[i] as number);
    }
    return RTV.tensor(data, [arr.length, 1]);
  }

  // All booleans (with no nulls that would make allNumbers true too)
  if (allBooleans) {
    const data = allocFloat64Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i] ? 1 : 0;
    }
    const t = RTV.tensor(data, [arr.length, 1]);
    t._isLogical = true;
    return t;
  }

  if (allStrings) {
    // Array of strings → cell array of char vectors
    const cells = arr.map(el => RTV.char(el as string));
    return RTV.cell(cells, [arr.length, 1]);
  }

  if (allObjects) {
    // Check if all objects have the same field names
    const objs = arr as Record<string, unknown>[];
    const fieldSets = objs.map(o =>
      Object.keys(o).map(makeValidFieldName).sort().join("\0")
    );
    const allSameFields = fieldSets.every(s => s === fieldSets[0]);

    if (allSameFields && fieldSets[0].length > 0) {
      // Same fields → struct array
      const fieldNames = Object.keys(objs[0]).map(makeValidFieldName);
      const elements = objs.map(o => {
        const fields = new Map<string, RuntimeValue>();
        for (const [key, value] of Object.entries(o)) {
          fields.set(makeValidFieldName(key), convertJsonValue(value));
        }
        return RTV.struct(fields);
      });
      return RTV.structArray(fieldNames, elements);
    }

    // Different fields → cell array of structs
    const cells = objs.map(o => convertJsonObject(o));
    return RTV.cell(cells, [arr.length, 1]);
  }

  // Mixed types → cell array
  const cells = arr.map(el => convertJsonValue(el));
  return RTV.cell(cells, [arr.length, 1]);
}

builtin({
  name: "jsondecode",
  nargin: 1,
  apply: args => {
    if (!isRuntimeChar(args[0]) && !isRuntimeString(args[0]))
      throw builtinError(
        "MATLAB:json:InvalidInput",
        "JSON text must be a character vector or a scalar non-missing string."
      );
    const txt = toString(args[0]);
    let parsed: unknown;
    try {
      parsed = JSON.parse(txt);
    } catch (e) {
      throw new RuntimeError(
        `jsondecode: invalid JSON: ${e instanceof Error ? e.message : String(e)}`
      );
    }
    return convertJsonValue(parsed);
  },
});

/**
 * Interpreter IBuiltins for string/char operations.
 *
 * Functions that involve cell arrays (strsplit, strjoin, symvar, regexp/regexpi)
 * or the parser (symvar) are left to legacy builtins since JitType has no cell kind.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  RuntimeTensor,
  RuntimeChar,
  isRuntimeChar,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeStringArray,
  stringArrayValue,
} from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import {
  RTV,
  toNumber,
  toString,
  RuntimeError,
  displayValue,
  formatDatetimeOrDuration,
} from "../../runtime/index.js";
import type { JitType } from "../../jitTypes.js";
import { registerIBuiltin } from "./types.js";
import { sprintfFormat } from "../../helpers/string.js";
import { scanFormat } from "../../helpers/scanFormat.js";
import { matlabNumToString } from "../../runtime/utils.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

// ── Type helpers ──────────────────────────────────────────────────────

function isTextType(t: JitType): boolean {
  return t.kind === "char" || t.kind === "string";
}

/** Return the same text kind as input, or null if not text. */
function preserveTextType(t: JitType): JitType | null {
  if (t.kind === "char") return { kind: "char" };
  if (t.kind === "string") return { kind: "string" };
  return null;
}

/** Apply a string function to a value, recursing into cells and mapping
 *  element-wise over string arrays. */
function applyTextFn(v: RuntimeValue, fn: (s: string) => string): RuntimeValue {
  if (isRuntimeCell(v)) {
    const out: RuntimeValue[] = new Array(v.data.length);
    for (let i = 0; i < v.data.length; i++) {
      out[i] = applyTextFn(v.data[i], fn);
    }
    return RTV.cell(out, [...v.shape]);
  }
  if (isRuntimeChar(v)) {
    // Preserve multi-row char-matrix shape (e.g. lower/upper of a padded
    // char matrix): map fn over each row, then re-stack.
    if (v.shape && (v.shape[0] ?? 1) > 1) {
      return charRowsToMatrix(valueToCharRows(v).map(fn));
    }
    return RTV.char(fn(v.value));
  }
  if (isRuntimeString(v)) return RTV.string(fn(toString(v)));
  if (isRuntimeStringArray(v)) {
    return RTV.stringArray(v.data.map(fn), [v.shape[0], v.shape[1]]);
  }
  return v;
}

/** Resolve for a simple 1-arg text→text function that preserves char/string,
 *  and maps element-wise over cell arrays of text and string arrays. */
function textPreserveResolve(fn: (s: string) => string): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 1) return null;
    const t = argTypes[0];
    if (t.kind === "cell") {
      return {
        outputTypes: [{ kind: "cell" }],
        apply: args => applyTextFn(args[0], fn),
      };
    }
    // String arrays infer as "unknown" — accept and verify at runtime.
    if (t.kind === "unknown") {
      return {
        outputTypes: [{ kind: "unknown" }],
        apply: args => {
          if (!isRuntimeStringArray(args[0]))
            throw new RuntimeError("Expected a text input");
          return applyTextFn(args[0], fn);
        },
      };
    }
    const out = preserveTextType(t);
    if (!out) return null;
    return {
      outputTypes: [out],
      apply: args => applyTextFn(args[0], fn),
    };
  };
}

// ── Text→text (type-preserving) ───────────────────────────────────────

registerIBuiltin({
  name: "lower",
  resolve: textPreserveResolve(s => s.toLowerCase()),
});

registerIBuiltin({
  name: "upper",
  resolve: textPreserveResolve(s => s.toUpperCase()),
});

registerIBuiltin({
  name: "strtrim",
  resolve: textPreserveResolve(s => s.trim()),
});

registerIBuiltin({
  name: "deblank",
  resolve: textPreserveResolve(s => s.replace(/\s+$/, "")),
});

registerIBuiltin({
  name: "reverse",
  resolve: textPreserveResolve(s => s.split("").reverse().join("")),
});

// ── strip (1-2 args) ─────────────────────────────────────────────────

registerIBuiltin({
  name: "strip",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (argTypes.length === 2 && !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        let side = "both";
        if (args.length >= 2) side = toString(args[1]).toLowerCase();
        let result: string;
        if (side === "left") result = s.replace(/^\s+/, "");
        else if (side === "right") result = s.replace(/\s+$/, "");
        else result = s.trim();
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── pad ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "pad",
  resolve: argTypes => {
    if (argTypes.length < 2 || argTypes.length > 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (argTypes[1].kind !== "number" && argTypes[1].kind !== "boolean")
      return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const n = Math.round(toNumber(args[1]));
        let side = "right";
        if (args.length >= 3) side = toString(args[2]).toLowerCase();
        if (s.length >= n)
          return isRuntimeChar(v) ? RTV.char(s) : RTV.string(s);
        const total = n - s.length;
        let result: string;
        if (side === "both") {
          const left = Math.floor(total / 2);
          const right = total - left;
          result = " ".repeat(left) + s + " ".repeat(right);
        } else if (side === "left") {
          result = " ".repeat(total) + s;
        } else {
          result = s + " ".repeat(total);
        }
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── erase ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "erase",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    const out =
      argTypes[0].kind === "unknown"
        ? { kind: "unknown" as const }
        : preserveTextType(argTypes[0]);
    if (!out) return null;
    if (!isTextType(argTypes[1])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const pat = toString(args[1]);
        return applyTextFn(args[0], s => s.split(pat).join(""));
      },
    };
  },
});

// ── strrep / replace ──────────────────────────────────────────────────

function strreplaceResolve(): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 3) return null;
    const out =
      argTypes[0].kind === "unknown" || argTypes[0].kind === "cell"
        ? ({ kind: argTypes[0].kind } as JitType)
        : preserveTextType(argTypes[0]);
    if (!out) return null;
    if (!isTextType(argTypes[1]) || !isTextType(argTypes[2])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const old = toString(args[1]);
        const rep = toString(args[2]);
        return applyTextFn(args[0], s => s.split(old).join(rep));
      },
    };
  };
}

registerIBuiltin({ name: "strrep", resolve: strreplaceResolve() });
registerIBuiltin({ name: "replace", resolve: strreplaceResolve() });

// ── regexprep ─────────────────────────────────────────────────────────

registerIBuiltin({
  name: "regexprep",
  resolve: argTypes => {
    if (argTypes.length < 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (!isTextType(argTypes[1]) || !isTextType(argTypes[2])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const str = toString(args[0]);
        const pat = toString(args[1]);
        const rep = toString(args[2]);
        // `s` (dotAll): MATLAB's `.` matches newline by default, unlike JS.
        let flags = "gs";
        for (let i = 3; i < args.length; i++) {
          const opt = toString(args[i]).toLowerCase();
          if (opt === "ignorecase") flags += "i";
        }
        const result = str.replace(new RegExp(pat, flags), rep);
        return isRuntimeChar(args[0]) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── extractBefore / extractAfter ──────────────────────────────────────

function extractResolve(extractFn: (s: string, arg: RuntimeValue) => string): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 2) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    const a1 = argTypes[1];
    if (!isTextType(a1) && a1.kind !== "number" && a1.kind !== "tensor")
      return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const result = extractFn(s, args[1]);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  };
}

registerIBuiltin({
  name: "extractBefore",
  resolve: extractResolve((s, arg) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg)) - 1;
    } else {
      const pat = toString(arg);
      pos = s.indexOf(pat);
      if (pos === -1)
        throw new RuntimeError("extractBefore: pattern not found");
    }
    return s.substring(0, pos);
  }),
});

registerIBuiltin({
  name: "extractAfter",
  resolve: extractResolve((s, arg) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg));
    } else {
      const pat = toString(arg);
      const idx = s.indexOf(pat);
      if (idx === -1) throw new RuntimeError("extractAfter: pattern not found");
      pos = idx + pat.length;
    }
    return s.substring(pos);
  }),
});

// ── extractBetween ────────────────────────────────────────────────────

registerIBuiltin({
  name: "extractBetween",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const a = args[1];
        const b = args[2];
        let startPos: number;
        let endPos: number;
        if (isRuntimeNumber(a) || (isRuntimeTensor(a) && a.data.length === 1)) {
          startPos = Math.round(toNumber(a)) - 1;
        } else {
          const pat = toString(a);
          const idx = s.indexOf(pat);
          if (idx === -1)
            throw new RuntimeError("extractBetween: start pattern not found");
          startPos = idx + pat.length;
        }
        if (isRuntimeNumber(b) || (isRuntimeTensor(b) && b.data.length === 1)) {
          endPos = Math.round(toNumber(b));
        } else {
          const pat = toString(b);
          const idx = s.indexOf(pat, startPos);
          if (idx === -1)
            throw new RuntimeError("extractBetween: end pattern not found");
          endPos = idx;
        }
        const result = s.substring(startPos, endPos);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── insertBefore / insertAfter ────────────────────────────────────────

function insertResolve(
  insertFn: (s: string, arg: RuntimeValue, ins: string) => string
): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const ins = toString(args[2]);
        const result = insertFn(s, args[1], ins);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  };
}

registerIBuiltin({
  name: "insertBefore",
  resolve: insertResolve((s, arg, ins) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg)) - 1;
    } else {
      const pat = toString(arg);
      pos = s.indexOf(pat);
      if (pos === -1) throw new RuntimeError("insertBefore: pattern not found");
    }
    return s.substring(0, pos) + ins + s.substring(pos);
  }),
});

registerIBuiltin({
  name: "insertAfter",
  resolve: insertResolve((s, arg, ins) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg));
    } else {
      const pat = toString(arg);
      const idx = s.indexOf(pat);
      if (idx === -1) throw new RuntimeError("insertAfter: pattern not found");
      pos = idx + pat.length;
    }
    return s.substring(0, pos) + ins + s.substring(pos);
  }),
});

// ── Text→boolean functions ────────────────────────────────────────────

function isText(v: RuntimeValue): boolean {
  return isRuntimeChar(v) || isRuntimeString(v);
}

/** Element-wise strcmp helper supporting cell arrays. */
/** View a string array as a cell of strings (for elementwise text ops). */
function stringArrayToCell(v: RuntimeValue): RuntimeValue {
  if (!isRuntimeStringArray(v)) return v;
  return RTV.cell(
    v.data.map(s => RTV.string(s)),
    [v.shape[0], v.shape[1]]
  );
}

function strcmpApply(
  args: RuntimeValue[],
  cmp: (a: string, b: string) => boolean
): RuntimeValue {
  const a = stringArrayToCell(args[0]);
  const b = stringArrayToCell(args[1]);
  const aIsCell = isRuntimeCell(a);
  const bIsCell = isRuntimeCell(b);

  if (!aIsCell && !bIsCell) {
    // Scalar comparison
    if (!isText(a) || !isText(b)) return RTV.logical(false);
    return RTV.logical(cmp(toString(a), toString(b)));
  }

  // At least one cell array — element-wise comparison
  if (aIsCell && bIsCell) {
    // Both cells — must be same size
    const cellA = a as import("../../runtime/types.js").RuntimeCell;
    const cellB = b as import("../../runtime/types.js").RuntimeCell;
    const len = cellA.data.length;
    const result = allocFloat64Array(len);
    for (let i = 0; i < len; i++) {
      const ai = cellA.data[i];
      const bi = cellB.data[i];
      result[i] =
        isText(ai) && isText(bi) && cmp(toString(ai), toString(bi)) ? 1 : 0;
    }
    return new RuntimeTensor(result, cellA.shape.slice(), undefined, true);
  }

  // One cell, one scalar
  const cell = (
    aIsCell ? a : b
  ) as import("../../runtime/types.js").RuntimeCell;
  const scalar = aIsCell ? b : a;
  const len = cell.data.length;
  const result = allocFloat64Array(len);
  for (let i = 0; i < len; i++) {
    const elem = cell.data[i];
    const [s1, s2] = aIsCell ? [elem, scalar] : [scalar, elem];
    result[i] =
      isText(s1) && isText(s2) && cmp(toString(s1), toString(s2)) ? 1 : 0;
  }
  return new RuntimeTensor(result, cell.shape.slice(), undefined, true);
}

registerIBuiltin({
  name: "strcmp",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => strcmpApply(args, (a, b) => a === b),
    };
  },
});

registerIBuiltin({
  name: "strcmpi",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args =>
        strcmpApply(args, (a, b) => a.toLowerCase() === b.toLowerCase()),
    };
  },
});

registerIBuiltin({
  name: "strncmp",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        if (!isText(args[0]) || !isText(args[1])) return RTV.logical(false);
        const n = Math.round(toNumber(args[2]));
        const s1 = toString(args[0]);
        const s2 = toString(args[1]);
        return RTV.logical(s1.substring(0, n) === s2.substring(0, n));
      },
    };
  },
});

registerIBuiltin({
  name: "strncmpi",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        if (!isText(args[0]) || !isText(args[1])) return RTV.logical(false);
        const n = Math.round(toNumber(args[2]));
        const s1 = toString(args[0]).toLowerCase();
        const s2 = toString(args[1]).toLowerCase();
        return RTV.logical(s1.substring(0, n) === s2.substring(0, n));
      },
    };
  },
});

// ── isstrprop ─────────────────────────────────────────────────────────

const RE_ALPHA = /\p{L}/u;
const RE_ALPHANUM = /[\p{L}\p{N}]/u;
const RE_CNTRL = /\p{Cc}/u;
const RE_GRAPHIC = /[\p{L}\p{N}\p{P}\p{S}\p{M}]/u;
const RE_LOWER = /\p{Ll}/u;
const RE_PUNCT = /\p{P}/u;
const RE_UPPER = /\p{Lu}/u;
const RE_WSPACE = /\s/u;

/** Per-codepoint predicate for an isstrprop category, or null if unknown. */
function isstrpropPredicate(
  category: string
): ((cp: number) => boolean) | null {
  switch (category) {
    case "alpha":
      return cp => RE_ALPHA.test(String.fromCodePoint(cp));
    case "alphanum":
      return cp => RE_ALPHANUM.test(String.fromCodePoint(cp));
    case "cntrl":
      return cp => RE_CNTRL.test(String.fromCodePoint(cp));
    case "digit":
      return cp => cp >= 48 && cp <= 57;
    case "graphic":
      return cp => RE_GRAPHIC.test(String.fromCodePoint(cp));
    case "lower":
      return cp => RE_LOWER.test(String.fromCodePoint(cp));
    case "print":
      return cp => cp === 32 || RE_GRAPHIC.test(String.fromCodePoint(cp));
    case "punct":
      return cp => RE_PUNCT.test(String.fromCodePoint(cp));
    case "wspace":
      return cp => RE_WSPACE.test(String.fromCodePoint(cp));
    case "upper":
      return cp => RE_UPPER.test(String.fromCodePoint(cp));
    case "xdigit":
      return cp =>
        (cp >= 48 && cp <= 57) ||
        (cp >= 65 && cp <= 70) ||
        (cp >= 97 && cp <= 102);
    default:
      return null;
  }
}

/** Convert a string to per-character code points (handles surrogate pairs). */
function stringCodePoints(s: string): number[] {
  const out: number[] = [];
  for (const ch of s) out.push(ch.codePointAt(0)!);
  return out;
}

/** Build a logical tensor by applying a predicate to each code point of `s`. */
function logicalRowFromString(
  s: string,
  pred: (cp: number) => boolean,
  shape?: number[]
): RuntimeValue {
  const cps = stringCodePoints(s);
  const data = allocFloat64Array(cps.length);
  for (let i = 0; i < cps.length; i++) data[i] = pred(cps[i]) ? 1 : 0;
  return new RuntimeTensor(data, shape ?? [1, cps.length], undefined, true);
}

/** Build a logical tensor by applying a predicate to each element of a numeric tensor. */
function logicalFromNumericTensor(
  data: ArrayLike<number>,
  shape: number[],
  pred: (cp: number) => boolean
): RuntimeValue {
  const out = allocFloat64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = pred(Math.round(data[i])) ? 1 : 0;
  }
  return new RuntimeTensor(out, [...shape], undefined, true);
}

function isstrpropApply(args: RuntimeValue[]): RuntimeValue {
  const v = args[0];
  const category = toString(args[1]);
  const pred = isstrpropPredicate(category);
  if (!pred) {
    throw new RuntimeError(`isstrprop: unknown category '${category}'`);
  }

  // Optional name-value pair: 'ForceCellOutput', tf
  let forceCell = false;
  if (args.length >= 3) {
    const flagName = toString(args[2]).toLowerCase();
    if (flagName !== "forcecelloutput") {
      throw new RuntimeError(`isstrprop: unknown name '${toString(args[2])}'`);
    }
    if (args.length < 4) {
      throw new RuntimeError("isstrprop: 'ForceCellOutput' requires a value");
    }
    forceCell = toNumber(args[3]) !== 0;
  }

  // Cell array of chars/strings → cell of logical row vectors
  if (isRuntimeCell(v)) {
    const out: RuntimeValue[] = new Array(v.data.length);
    for (let i = 0; i < v.data.length; i++) {
      const elem = v.data[i];
      const s = isText(elem) ? toString(elem) : "";
      out[i] = logicalRowFromString(s, pred);
    }
    return RTV.cell(out, [...v.shape]);
  }

  let result: RuntimeValue;
  if (isRuntimeString(v)) {
    result = logicalRowFromString(toString(v), pred);
  } else if (isRuntimeChar(v)) {
    const c = v as import("../../runtime/types.js").RuntimeChar;
    result = logicalRowFromString(
      c.value,
      pred,
      c.shape ? [...c.shape] : undefined
    );
  } else if (isRuntimeTensor(v)) {
    result = logicalFromNumericTensor(v.data, v.shape, pred);
  } else if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
    const cp = Math.round(toNumber(v));
    const data = allocFloat64Array(1);
    data[0] = pred(cp) ? 1 : 0;
    result = new RuntimeTensor(data, [1, 1], undefined, true);
  } else {
    throw new RuntimeError("isstrprop: unsupported input type");
  }

  if (forceCell) return RTV.cell([result], [1, 1]);
  return result;
}

registerIBuiltin({
  name: "isstrprop",
  help: {
    signatures: [
      "TF = isstrprop(str, category)",
      "TF = isstrprop(str, category, 'ForceCellOutput', tf)",
    ],
    description:
      "Test which characters in str belong to a category (alpha, alphanum, cntrl, digit, graphic, lower, print, punct, upper, wspace, xdigit). Returns a logical array, or a cell of logical vectors when str is a cell array or ForceCellOutput is true. Numeric input is treated as Unicode code points.",
  },
  resolve: argTypes => {
    if (argTypes.length < 2 || argTypes.length > 4) return null;
    if (!isTextType(argTypes[1])) return null;
    if (argTypes.length >= 3 && !isTextType(argTypes[2])) return null;
    if (argTypes.length === 4) {
      const k = argTypes[3].kind;
      if (k !== "number" && k !== "boolean") return null;
    }
    const t0 = argTypes[0];
    if (
      t0.kind !== "char" &&
      t0.kind !== "string" &&
      t0.kind !== "number" &&
      t0.kind !== "boolean" &&
      t0.kind !== "tensor" &&
      t0.kind !== "cell"
    ) {
      return null;
    }
    const isCell = t0.kind === "cell";
    return {
      outputTypes: isCell
        ? [{ kind: "cell" }]
        : [{ kind: "tensor", isComplex: false, isLogical: true }],
      apply: args => isstrpropApply(args),
    };
  },
});

registerIBuiltin({
  name: "isspace",
  help: {
    signatures: ["TF = isspace(str)"],
    description:
      "Return a logical array the same size as str, true where the character is whitespace. Numeric input is treated as Unicode code points.",
  },
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    if (
      k !== "char" &&
      k !== "string" &&
      k !== "number" &&
      k !== "boolean" &&
      k !== "tensor"
    ) {
      return null;
    }
    const pred = (cp: number) => RE_WSPACE.test(String.fromCodePoint(cp));
    return {
      outputTypes: [{ kind: "tensor", isComplex: false, isLogical: true }],
      apply: args => {
        const v = args[0];
        if (isRuntimeChar(v)) {
          return logicalRowFromString(
            v.value,
            pred,
            v.shape ? [...v.shape] : undefined
          );
        }
        if (isRuntimeString(v)) return logicalRowFromString(toString(v), pred);
        if (isRuntimeTensor(v))
          return logicalFromNumericTensor(v.data, v.shape, pred);
        // scalar number / boolean → code point
        return logicalFromNumericTensor([toNumber(v)], [1, 1], pred);
      },
    };
  },
});

/** Normalize a pattern argument (scalar text, cellstr, or string array) to a
 *  list of pattern strings. */
function patternList(pat: RuntimeValue): string[] {
  if (isRuntimeCell(pat)) return pat.data.map(e => toString(e));
  if (isRuntimeStringArray(pat)) return pat.data.slice();
  return [toString(pat)];
}

/** Apply a per-element text predicate over a subject that may be scalar
 *  text, a cellstr, or a string array. */
function textSubjectApply(
  subject: RuntimeValue,
  perElem: (s: string) => boolean
): RuntimeValue {
  if (isRuntimeStringArray(subject)) {
    return new RuntimeTensor(
      allocFloat64Array(subject.data.map(s => (perElem(s) ? 1 : 0))),
      [subject.shape[0], subject.shape[1]],
      undefined,
      true
    );
  }
  if (isRuntimeCell(subject)) {
    return new RuntimeTensor(
      allocFloat64Array(subject.data.map(e => (perElem(toString(e)) ? 1 : 0))),
      subject.shape.slice(),
      undefined,
      true
    );
  }
  return RTV.logical(perElem(toString(subject)));
}

function subjectTypeOk(t: JitType): boolean {
  return isTextType(t) || t.kind === "cell" || t.kind === "unknown";
}

registerIBuiltin({
  name: "contains",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    if (!subjectTypeOk(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        const pats = patternList(args[1]);
        return textSubjectApply(args[0], s => pats.some(p => s.includes(p)));
      },
    };
  },
});

function resolveStartsEndsWith(
  name: string,
  testFn: (s: string, pat: string) => boolean
) {
  registerIBuiltin({
    name,
    resolve: argTypes => {
      if (argTypes.length < 2) return null;
      if (!subjectTypeOk(argTypes[0])) return null;
      return {
        outputTypes: [{ kind: "boolean" }],
        apply: args => {
          let ic = false;
          for (let i = 2; i + 1 < args.length; i += 2) {
            if (toString(args[i]).toLowerCase() === "ignorecase") {
              ic = !!toNumber(args[i + 1]);
            }
          }
          let pats = patternList(args[1]);
          if (ic) pats = pats.map(p => p.toLowerCase());
          return textSubjectApply(args[0], s0 => {
            const s = ic ? s0.toLowerCase() : s0;
            return pats.some(p => testFn(s, p));
          });
        },
      };
    },
  });
}

resolveStartsEndsWith("startsWith", (s, p) => s.startsWith(p));
resolveStartsEndsWith("endsWith", (s, p) => s.endsWith(p));

// ── Text→number functions ─────────────────────────────────────────────

registerIBuiltin({
  name: "strlength",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0]) && argTypes[0].kind !== "unknown") return null;
    return {
      outputTypes: [{ kind: "number", sign: "nonneg" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeString(v)) return RTV.num(v.length);
        if (isRuntimeChar(v)) return RTV.num(v.value.length);
        if (isRuntimeStringArray(v)) {
          return RTV.tensor(allocFloat64Array(v.data.map(s => s.length)), [
            v.shape[0],
            v.shape[1],
          ]);
        }
        throw new RuntimeError("strlength: argument must be a string or char");
      },
    };
  },
});

/** Parse a single string to a double, handling commas and complex numbers. */
function str2doubleScalar(s: string): number {
  s = s.trim();
  if (s === "") return NaN;

  // Strip commas used as thousands separators
  s = s.replace(/,/g, "");

  // Try complex: e.g. "3+4i", "3-4j", "2i", "-1.5j"
  // Full form: real +/- imag*i/j
  const complexFull =
    /^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*([+-]\s*(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*[ij]$/i;
  const mFull = complexFull.exec(s);
  if (mFull) {
    const real = Number(mFull[1]);
    const imag = Number(mFull[2].replace(/\s/g, ""));
    if (!isNaN(real) && !isNaN(imag)) return real; // MATLAB str2double returns real part... actually returns complex
  }

  // Pure imaginary: e.g. "4i", "-3j", "+2.5i"
  const pureImag = /^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*[ij]$/i;
  const mImag = pureImag.exec(s);
  if (mImag) {
    // For now, return NaN for pure imaginary (complex not yet fully supported in str2double output)
    // MATLAB returns a complex number; we return NaN since our output type is number
  }

  const n = Number(s);
  return isNaN(n) ? NaN : n;
}

registerIBuiltin({
  name: "str2double",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const t = argTypes[0];
    // Accept text or cell array
    if (isTextType(t)) {
      return {
        outputTypes: [{ kind: "number" }],
        apply: args => RTV.num(str2doubleScalar(toString(args[0]))),
      };
    }
    if (t.kind === "cell") {
      return {
        outputTypes: [{ kind: "tensor", isComplex: false }],
        apply: args => {
          const cell = args[0];
          if (
            typeof cell !== "object" ||
            cell === null ||
            (cell as { kind?: string }).kind !== "cell"
          )
            throw new RuntimeError("str2double: expected cell array");
          const c = cell as import("../../runtime/types.js").RuntimeCell;
          const data = allocFloat64Array(c.data.length);
          for (let i = 0; i < c.data.length; i++) {
            const el = c.data[i];
            if (isRuntimeString(el) || isRuntimeChar(el)) {
              data[i] = str2doubleScalar(toString(el));
            } else {
              data[i] = NaN;
            }
          }
          return RTV.tensor(data, c.shape.slice());
        },
      };
    }
    if (t.kind === "unknown") {
      return {
        outputTypes: [{ kind: "tensor", isComplex: false }],
        apply: args => {
          const v = args[0];
          if (!isRuntimeStringArray(v))
            throw new RuntimeError("str2double: expected text input");
          return RTV.tensor(
            allocFloat64Array(v.data.map(s => str2doubleScalar(s))),
            [v.shape[0], v.shape[1]]
          );
        },
      };
    }
    return null;
  },
});

// NOTE: the real str2num is the special builtin in
// interpreterSpecialBuiltins.ts, which evaluates the text as `[<text>]`
// (handling vectors / matrices / arithmetic). It intercepts before this
// IBuiltin, which only stays registered for builtin-listing/type presence
// and the degenerate fallback.
registerIBuiltin({
  name: "str2num",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    // Output can be number or empty tensor — conservatively say number
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => {
        const s = toString(args[0]);
        const n = Number(s);
        return Number.isNaN(n)
          ? RTV.tensor(allocFloat64Array(0), [0, 0])
          : RTV.num(n);
      },
    };
  },
});

registerIBuiltin({
  name: "count",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [{ kind: "number", sign: "nonneg" }],
      apply: args => {
        const s = toString(args[0]);
        const pat = toString(args[1]);
        if (pat.length === 0) return RTV.num(s.length + 1);
        let n = 0;
        let idx = 0;
        while ((idx = s.indexOf(pat, idx)) !== -1) {
          n++;
          idx += pat.length;
        }
        return RTV.num(n);
      },
    };
  },
});

registerIBuiltin({
  name: "hex2dec",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => RTV.num(parseInt(toString(args[0]), 16)),
    };
  },
});

registerIBuiltin({
  name: "bin2dec",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => RTV.num(parseInt(toString(args[0]), 2)),
    };
  },
});

registerIBuiltin({
  name: "strfind",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [{ kind: "tensor", isComplex: false, ndim: 2 }],
      apply: args => {
        const s = toString(args[0]);
        const pattern = toString(args[1]);
        const indices: number[] = [];
        let pos = 0;
        while (pos <= s.length - pattern.length) {
          const idx = s.indexOf(pattern, pos);
          if (idx === -1) break;
          indices.push(idx + 1);
          pos = idx + 1;
        }
        if (indices.length === 0) {
          return RTV.tensor(allocFloat64Array(0), [1, 0]);
        }
        const data = allocFloat64Array(indices.length);
        indices.forEach((v, i) => (data[i] = v));
        return RTV.tensor(data, [1, indices.length]);
      },
    };
  },
});

// ── Number/tensor→char functions ──────────────────────────────────────

registerIBuiltin({
  name: "blanks",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind !== "number" && argTypes[0].kind !== "boolean")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const n = Math.round(toNumber(args[0]));
        return RTV.char(" ".repeat(Math.max(0, n)));
      },
    };
  },
});

registerIBuiltin({
  name: "dec2hex",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (argTypes[0].kind !== "number" && argTypes[0].kind !== "boolean")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        let n = Math.round(toNumber(args[0]));
        if (n < 0) {
          // Two's complement: find smallest byte-aligned width
          let bits = 8;
          while (bits <= 52 && -(1 << (bits - 1)) > n) bits += 8;
          n = (1 << bits) + n;
        }
        let hex = n.toString(16).toUpperCase();
        if (args.length === 2) {
          const minDigits = Math.round(toNumber(args[1]));
          while (hex.length < minDigits) hex = "0" + hex;
        }
        return RTV.char(hex);
      },
    };
  },
});

registerIBuiltin({
  name: "dec2bin",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (argTypes[0].kind !== "number" && argTypes[0].kind !== "boolean")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        let n = Math.round(toNumber(args[0]));
        if (n < 0) {
          let bits = 8;
          while (bits <= 52 && -(1 << (bits - 1)) > n) bits += 8;
          n = (1 << bits) + n;
        }
        let bin = n.toString(2);
        if (args.length === 2) {
          const minDigits = Math.round(toNumber(args[1]));
          while (bin.length < minDigits) bin = "0" + bin;
        }
        return RTV.char(bin);
      },
    };
  },
});

registerIBuiltin({
  name: "int2str",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        const roundInt = (x: number) => Math.sign(x) * Math.round(Math.abs(x));
        if (isRuntimeTensor(v)) {
          const rows = v.shape[0] || 1;
          const cols = v.shape.length >= 2 ? v.shape[1] : v.data.length;
          const rowStrs: string[] = [];
          for (let r = 0; r < rows; r++) {
            const vals: string[] = [];
            for (let c = 0; c < cols; c++) {
              vals.push(String(roundInt(v.data[r + c * rows])));
            }
            rowStrs.push(vals.join("  "));
          }
          return RTV.char(rowStrs.join("\n"));
        }
        return RTV.char(String(roundInt(toNumber(v))));
      },
    };
  },
});

// ── num2str ───────────────────────────────────────────────────────────

const numStr = matlabNumToString;

registerIBuiltin({
  name: "num2str",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
      return null;
    // Second arg can be format string or precision number
    if (
      argTypes.length === 2 &&
      !isTextType(argTypes[1]) &&
      argTypes[1].kind !== "number"
    )
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        if (
          args.length >= 2 &&
          (isRuntimeString(args[1]) || isRuntimeChar(args[1]))
        ) {
          return RTV.char(
            sprintfFormat(isRuntimeString(args[1]) ? args[1] : args[1].value, [
              v,
            ])
          );
        }
        const precision =
          args.length >= 2 && isRuntimeNumber(args[1])
            ? (args[1] as number)
            : undefined;
        const fmtNum = (x: number): string => {
          if (precision !== undefined) {
            let s = x.toPrecision(precision);
            if (s.includes(".")) {
              const eIdx = s.search(/[eE]/);
              if (eIdx === -1) {
                s = s.replace(/\.?0+$/, "");
              } else {
                const mantissa = s.slice(0, eIdx).replace(/\.?0+$/, "");
                let expPart = s.slice(eIdx);
                expPart = expPart.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
                s = mantissa + expPart;
              }
            }
            return s;
          }
          return numStr(x);
        };
        if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
          return RTV.char(fmtNum(toNumber(v)));
        }
        if (isRuntimeTensor(v)) {
          const nRows = v.shape[0] || 1;
          const nCols = v.data.length / nRows;
          const strs: string[][] = [];
          for (let r = 0; r < nRows; r++) {
            const row: string[] = [];
            for (let c = 0; c < nCols; c++) {
              row.push(fmtNum(v.data[c * nRows + r]));
            }
            strs.push(row);
          }
          const colWidths: number[] = [];
          for (let c = 0; c < nCols; c++) {
            let maxW = 0;
            for (let r = 0; r < nRows; r++) {
              if (strs[r][c].length > maxW) maxW = strs[r][c].length;
            }
            colWidths.push(maxW);
          }
          const rowStrs = strs.map(row =>
            row.map((s, c) => s.padStart(colWidths[c])).join("  ")
          );
          if (nRows === 1) return RTV.char(rowStrs[0]);
          const rowWidth = rowStrs[0].length;
          const paddedRows = rowStrs.map(r => r.padEnd(rowWidth));
          return new RuntimeChar(paddedRows.join(""), [nRows, rowWidth]);
        }
        return RTV.char(String(toNumber(v)));
      },
    };
  },
});

// ── mat2str ───────────────────────────────────────────────────────────

registerIBuiltin({
  name: "mat2str",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    if (
      a.kind !== "number" &&
      a.kind !== "boolean" &&
      a.kind !== "tensor" &&
      a.kind !== "complex_or_number"
    )
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        const prec = args.length >= 2 ? Math.round(toNumber(args[1])) : -1;
        const fmt = (n: number): string => {
          // MATLAB spells these Inf / -Inf / NaN (not JS's "Infinity").
          if (!Number.isFinite(n))
            return Number.isNaN(n) ? "NaN" : n > 0 ? "Inf" : "-Inf";
          return prec >= 0 ? Number(n.toPrecision(prec)).toString() : String(n);
        };
        // Complex element: MATLAB writes `re+imi` / `re-imi` (always an
        // explicit imaginary part for a complex-typed value, e.g. `0+1i`).
        const fmtC = (re: number, im: number): string =>
          fmt(re) + (im < 0 ? "-" : "+") + fmt(Math.abs(im)) + "i";
        if (isRuntimeComplexNumber(v)) return RTV.char(fmtC(v.re, v.im));
        if (isRuntimeNumber(v)) return RTV.char(fmt(v));
        if (isRuntimeTensor(v)) {
          const im = v.imag;
          const cell = (k: number): string =>
            im !== undefined ? fmtC(v.data[k], im[k]) : fmt(v.data[k]);
          const nRows = v.shape[0] || 1;
          const nCols = v.data.length / nRows;
          if (nRows === 1 && nCols === 1) return RTV.char(cell(0));
          const rows: string[] = [];
          for (let r = 0; r < nRows; r++) {
            const elems: string[] = [];
            for (let c = 0; c < nCols; c++) {
              elems.push(cell(c * nRows + r));
            }
            rows.push(elems.join(" "));
          }
          return RTV.char("[" + rows.join(";") + "]");
        }
        return RTV.char(String(toNumber(v)));
      },
    };
  },
});

// ── Type conversion: char, string ─────────────────────────────────────

/** Collect the char rows of a char matrix / string / cellstr, flattening a
 *  cell array element-wise (column-major). */
function collectRows(v: RuntimeValue): string[] {
  if (isRuntimeCell(v)) {
    const rows: string[] = [];
    for (const el of v.data) rows.push(...valueToCharRows(el));
    return rows;
  }
  return valueToCharRows(v);
}

// strmatch (legacy): find rows of a text array that begin with (or, with
// 'exact', equal) a pattern. Ported from MATLAB's strmatch.m.
registerIBuiltin({
  name: "strmatch",
  help: {
    signatures: [
      "x = strmatch(str, strarray)",
      "x = strmatch(str, strarray, 'exact')",
    ],
    description:
      "(Legacy) Find rows of STRARRAY that begin with STR, or that equal STR when 'exact' is given. Returns a column vector of matching row indices.",
  },
  resolve: argTypes => {
    if (argTypes.length < 2 || argTypes.length > 3) return null;
    if (!isTextType(argTypes[0])) return null;
    const sa = argTypes[1];
    if (sa.kind !== "char" && sa.kind !== "string" && sa.kind !== "cell")
      return null;
    return {
      outputTypes: [{ kind: "tensor", isComplex: false }],
      apply: args => {
        const str = toString(args[0]);
        let rows = collectRows(args[1]);
        const n = rows.length === 0 ? 0 : Math.max(...rows.map(r => r.length));
        rows = rows.map(r => r.padEnd(n, " "));
        const exactMatch = args.length === 3;
        let s = str;
        let len = s.length;
        if (len > n) {
          return RTV.tensor(allocFloat64Array(0), [0, 1]);
        }
        if (exactMatch && len < n) {
          // Pad pattern with nulls if the last column holds a null, else
          // spaces (matches MATLAB's strmatch.m).
          const useNull = rows.some(r => r.charCodeAt(n - 1) === 0);
          s = s.padEnd(n, useNull ? "\0" : " ");
          len = n;
        }
        const matches: number[] = [];
        for (let r = 0; r < rows.length; r++) {
          let ok = true;
          for (let i = 0; i < len; i++) {
            if (rows[r][i] !== s[i]) {
              ok = false;
              break;
            }
          }
          if (ok) matches.push(r + 1);
        }
        return RTV.tensor(allocFloat64Array(matches), [matches.length, 1]);
      },
    };
  },
});

/** Extract the character rows of a value (one entry per row), for char() of
 *  a cell array. A char matrix yields its rows; a string/scalar yields one
 *  row; a numeric array yields one row of char codes per matrix row. */
function valueToCharRows(v: RuntimeValue): string[] {
  if (isRuntimeChar(v)) {
    const cols = v.shape ? (v.shape[1] ?? v.value.length) : v.value.length;
    const numRows = v.shape ? (v.shape[0] ?? 1) : 1;
    if (numRows <= 1) return [v.value];
    const out: string[] = [];
    for (let r = 0; r < numRows; r++)
      out.push(v.value.slice(r * cols, (r + 1) * cols));
    return out;
  }
  if (isRuntimeString(v)) return [v];
  if (isRuntimeNumber(v)) return [String.fromCharCode(Math.round(v))];
  if (isRuntimeTensor(v)) {
    const rows = v.shape.length >= 2 ? (v.shape[0] ?? 1) : 1;
    const cols = v.shape.length >= 2 ? (v.shape[1] ?? 0) : v.data.length;
    const out: string[] = [];
    for (let r = 0; r < rows; r++) {
      let s = "";
      for (let c = 0; c < cols; c++)
        s += String.fromCharCode(Math.round(v.data[c * rows + r]));
      out.push(s);
    }
    return out;
  }
  throw new RuntimeError("char: unsupported cell element type");
}

/** Stack char rows into a row-major RuntimeChar matrix, right-padding each
 *  row with spaces to the widest row. */
function charRowsToMatrix(rows: string[]): RuntimeChar {
  if (rows.length === 0) return RTV.char("");
  const width = Math.max(...rows.map(r => r.length));
  const padded = rows.map(r => r.padEnd(width, " "));
  if (rows.length === 1) return RTV.char(padded[0]);
  return new RuntimeChar(padded.join(""), [rows.length, width]);
}

registerIBuiltin({
  name: "char",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (
      a.kind !== "char" &&
      a.kind !== "string" &&
      a.kind !== "number" &&
      a.kind !== "tensor" &&
      a.kind !== "cell" &&
      a.kind !== "unknown" &&
      // datetime/duration only — other classes keep their own char methods
      // (or the generic "does not support these argument types" error).
      !(
        a.kind === "class_instance" &&
        (a.className === "datetime" || a.className === "duration")
      )
    )
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeChar(v)) return v;
        // char(datetime) / char(duration): the display text (honoring a
        // datetime's Format property).
        if (isRuntimeClassInstance(v)) {
          const s = formatDatetimeOrDuration(v);
          if (s !== null) return RTV.char(s);
        }
        if (isRuntimeString(v)) return RTV.char(v);
        if (isRuntimeNumber(v))
          return RTV.char(String.fromCharCode(Math.round(v)));
        if (isRuntimeTensor(v)) {
          const chars: string[] = [];
          for (let i = 0; i < v.data.length; i++) {
            chars.push(String.fromCharCode(Math.round(v.data[i])));
          }
          return RTV.char(chars.join(""));
        }
        // char(cellArray): each element becomes one or more rows, all
        // right-padded with spaces to the widest row. Matches MATLAB's
        // conversion of a cellstr to a padded char matrix.
        if (isRuntimeCell(v)) {
          const rows: string[] = [];
          for (const el of v.data) rows.push(...valueToCharRows(el));
          return charRowsToMatrix(rows);
        }
        // char(stringArray): one row per element (column-major order),
        // right-padded to the widest element.
        if (isRuntimeStringArray(v)) {
          return charRowsToMatrix(v.data.slice());
        }
        throw new RuntimeError("char: unsupported arguments");
      },
    };
  },
});

registerIBuiltin({
  name: "string",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    const scalarText =
      a.kind === "char" ||
      a.kind === "string" ||
      a.kind === "number" ||
      a.kind === "boolean";
    return {
      outputTypes: [scalarText ? { kind: "string" } : { kind: "unknown" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeString(v) || isRuntimeStringArray(v)) return v;
        if (isRuntimeChar(v)) {
          const rows = v.shape ? v.shape[0] : 1;
          if (rows <= 1) return RTV.string(v.value);
          // Char matrix: one string per row (m x 1), pad spaces preserved.
          const width = v.shape![1];
          const out: string[] = [];
          for (let r = 0; r < rows; r++) {
            out.push(v.value.slice(r * width, (r + 1) * width));
          }
          return RTV.stringArray(out, [rows, 1]);
        }
        if (isRuntimeNumber(v)) return RTV.string(numStr(v));
        if (isRuntimeLogical(v)) return RTV.string(v ? "true" : "false");
        if (isRuntimeTensor(v)) {
          // Elementwise conversion, same shape ([] -> 0x0 empty string array).
          const rows = v.shape.length >= 2 ? v.shape[0] : 1;
          const cols =
            v.data.length === 0
              ? (v.shape[1] ?? 0)
              : v.data.length / (rows || 1);
          const out: string[] = [];
          for (let i = 0; i < v.data.length; i++) {
            out.push(
              v._isLogical ? (v.data[i] ? "true" : "false") : numStr(v.data[i])
            );
          }
          return stringArrayValue(out, [rows, cols]);
        }
        if (isRuntimeCell(v)) {
          const rows = v.shape.length >= 2 ? v.shape[0] : 1;
          const cols = v.data.length / (rows || 1);
          const out = v.data.map(el => {
            const rv = ensureRuntimeValue(el);
            if (isRuntimeChar(rv)) return rv.value;
            if (isRuntimeString(rv)) return rv;
            if (isRuntimeNumber(rv)) return numStr(rv);
            if (isRuntimeLogical(rv)) return rv ? "true" : "false";
            throw new RuntimeError(
              "string: cell elements must be text or numbers"
            );
          });
          return stringArrayValue(out, [rows, cols]);
        }
        return RTV.string(displayValue(v));
      },
    };
  },
});

registerIBuiltin({
  name: "strings",
  help: {
    signatures: ["strings", "strings(n)", "strings(m,n)", "strings([m n])"],
    description:
      'Create a string array with every element set to "" (empty string).',
  },
  resolve: argTypes => {
    if (argTypes.some(t => t.kind !== "number" && t.kind !== "tensor"))
      return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const dims: number[] = [];
        for (const a of args) {
          const rv = ensureRuntimeValue(a);
          if (isRuntimeTensor(rv)) {
            for (const d of rv.data) dims.push(Math.max(0, Math.round(d)));
          } else {
            dims.push(Math.max(0, Math.round(toNumber(rv))));
          }
        }
        if (dims.length === 0) dims.push(1, 1);
        if (dims.length === 1) dims.push(dims[0]);
        const m = dims[0];
        const n = dims.slice(1).reduce((a, b) => a * b, 1);
        return stringArrayValue(new Array<string>(m * n).fill(""), [m, n]);
      },
    };
  },
});

registerIBuiltin({
  name: "newline",
  help: {
    signatures: ["newline"],
    description: "Newline character, char(10).",
  },
  resolve: argTypes => {
    if (argTypes.length !== 0) return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: () => RTV.char("\n"),
    };
  },
});

// ── split / join (string arrays) ──────────────────────────────────────

registerIBuiltin({
  name: "split",
  help: {
    signatures: ["split(str)", "split(str, delimiter)"],
    description:
      "Split text at whitespace (or the given delimiter(s)) into a string array. " +
      "A scalar input yields a column of parts.",
  },
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    if (!isTextType(a) && a.kind !== "cell" && a.kind !== "unknown")
      return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const delims = args.length >= 2 ? patternList(args[1]) : null;
        const splitOne = (s: string): string[] => {
          if (delims === null) {
            const trimmed = s.trim();
            return trimmed.length === 0 ? [""] : trimmed.split(/\s+/);
          }
          const escaped = delims
            .map(d => d.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
            .join("|");
          return s.split(new RegExp(escaped));
        };
        const v = args[0];
        const subjects: string[] = isRuntimeStringArray(v)
          ? v.data.slice()
          : isRuntimeCell(v)
            ? v.data.map(e => toString(e))
            : [toString(v)];
        const partsPer = subjects.map(splitOne);
        const nParts = partsPer[0].length;
        if (partsPer.some(p => p.length !== nParts)) {
          throw new RuntimeError(
            "split: all elements must split into the same number of parts"
          );
        }
        if (subjects.length === 1) {
          return stringArrayValue(partsPer[0], [nParts, 1]);
        }
        // Vector input of m elements -> m x nParts (column-major storage).
        const m = subjects.length;
        const out: string[] = new Array(m * nParts);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < nParts; j++) {
            out[j * m + i] = partsPer[i][j];
          }
        }
        return stringArrayValue(out, [m, nParts]);
      },
    };
  },
});

registerIBuiltin({
  name: "join",
  help: {
    signatures: ["join(str)", "join(str, delimiter)"],
    description:
      "Combine the elements of a string array into single strings, " +
      "separated by a space (or the given delimiter).",
  },
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    if (!isTextType(a) && a.kind !== "cell" && a.kind !== "unknown")
      return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const delim = args.length >= 2 ? toString(args[1]) : " ";
        const v = args[0];
        if (isRuntimeString(v)) return v;
        if (isRuntimeChar(v)) return RTV.string(v.value);
        const elems: string[] = isRuntimeStringArray(v)
          ? v.data
          : isRuntimeCell(v)
            ? v.data.map(e => toString(e))
            : [toString(v)];
        const shape: [number, number] = isRuntimeStringArray(v)
          ? v.shape
          : [1, elems.length];
        const [m, n] = shape;
        if (m === 1 || n === 1) {
          return RTV.string(elems.join(delim));
        }
        // Matrix: join along dim 2 -> m x 1 column.
        const out: string[] = [];
        for (let r = 0; r < m; r++) {
          const parts: string[] = [];
          for (let c = 0; c < n; c++) parts.push(elems[c * m + r]);
          out.push(parts.join(delim));
        }
        return stringArrayValue(out, [m, 1]);
      },
    };
  },
});

registerIBuiltin({
  name: "cellstr",
  help: {
    signatures: ["cellstr(A)"],
    description:
      "Convert a string array, char array, or cell array of text to a " +
      "cell array of character vectors.",
  },
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (!isTextType(a) && a.kind !== "cell" && a.kind !== "unknown")
      return null;
    return {
      outputTypes: [{ kind: "cell" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeCell(v)) {
          return RTV.cell(
            v.data.map(e => RTV.char(toString(e))),
            [...v.shape]
          );
        }
        if (isRuntimeStringArray(v)) {
          return RTV.cell(
            v.data.map(s => RTV.char(s)),
            [v.shape[0], v.shape[1]]
          );
        }
        if (isRuntimeString(v)) return RTV.cell([RTV.char(v)], [1, 1]);
        if (isRuntimeChar(v)) {
          const rows = v.shape ? v.shape[0] : 1;
          if (rows <= 1) {
            // MATLAB deblanks each row when converting char to cellstr.
            return RTV.cell([RTV.char(v.value.replace(/\s+$/, ""))], [1, 1]);
          }
          const width = v.shape![1];
          const out: RuntimeValue[] = [];
          for (let r = 0; r < rows; r++) {
            out.push(
              RTV.char(
                v.value.slice(r * width, (r + 1) * width).replace(/\s+$/, "")
              )
            );
          }
          return RTV.cell(out, [rows, 1]);
        }
        throw new RuntimeError("cellstr: unsupported argument");
      },
    };
  },
});

registerIBuiltin({
  name: "ismissing",
  help: {
    signatures: ["ismissing(A)"],
    description:
      "Logical array marking missing elements. numbl has no <missing> " +
      "value, so string inputs return all-false.",
  },
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeStringArray(v)) {
          const t = RTV.tensor(allocFloat64Array(v.data.length), [
            v.shape[0],
            v.shape[1],
          ]);
          t._isLogical = true;
          return t;
        }
        if (isRuntimeTensor(v)) {
          const t = RTV.tensor(
            allocFloat64Array(v.data.map(x => (isNaN(x) ? 1 : 0))),
            [...v.shape]
          );
          t._isLogical = true;
          return t;
        }
        if (isRuntimeNumber(v)) return RTV.logical(isNaN(v));
        return RTV.logical(false);
      },
    };
  },
});

// ── strcat ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "strcat",
  resolve: argTypes => {
    if (argTypes.length < 1) return null;
    // Fall back for any unknown/cell args
    if (argTypes.some(t => t.kind === "unknown")) return null;
    // Must have at least one text arg
    if (!argTypes.some(t => isTextType(t))) return null;
    // If any string arg, output is string; otherwise char
    const hasString = argTypes.some(t => t.kind === "string");
    const outType: JitType = hasString ? { kind: "string" } : { kind: "char" };
    return {
      outputTypes: [outType],
      apply: args => {
        const anyString = args.some(a => isRuntimeString(a));
        if (anyString) {
          return RTV.string(args.map(a => toString(a)).join(""));
        }
        const parts = args.map(a => toString(a).replace(/[ \t\v\n\r\f]+$/, ""));
        return RTV.char(parts.join(""));
      },
    };
  },
});

// ── sprintf ───────────────────────────────────────────────────────────

registerIBuiltin({
  name: "sprintf",
  resolve: argTypes => {
    if (argTypes.length < 1) return null;
    if (!isTextType(argTypes[0])) return null;
    // Output type matches format arg type
    const isChar = argTypes[0].kind === "char";
    const outType: JitType = isChar ? { kind: "char" } : { kind: "string" };
    return {
      outputTypes: [outType],
      apply: args => {
        const fmt = toString(args[0]);
        // A string-array argument supplies one text arg per element
        // (the format cycles over them, like numeric arrays).
        const rest = args
          .slice(1)
          .flatMap(a =>
            isRuntimeStringArray(a) ? a.data.map(s => RTV.string(s)) : [a]
          );
        const result = sprintfFormat(fmt, rest);
        return isChar ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── sscanf ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "sscanf",
  resolve: (argTypes, nargout) => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    const outputTypes: JitType[] = [{ kind: "number" }];
    if (nargout >= 2) outputTypes.push({ kind: "number" });
    if (nargout >= 3) outputTypes.push({ kind: "char" }); // errmsg
    if (nargout >= 4) outputTypes.push({ kind: "number" }); // nextindex
    return {
      outputTypes,
      apply: (args, nout) => {
        const str = toString(args[0]);
        const fmt = toString(args[1]);
        const maxCount = args.length >= 3 ? toNumber(args[2]) : Infinity;
        const {
          results,
          consumed: strPos,
          matchFailure,
        } = scanFormat(str, fmt, maxCount);

        const vals =
          results.length === 1
            ? RTV.num(results[0])
            : RTV.tensor(allocFloat64Array(results), [results.length, 1]);
        if (nout >= 2) {
          const out: RuntimeValue[] = [vals, RTV.num(results.length)];
          if (nout >= 3) {
            // errmsg: empty on success, non-empty when scanning stopped on a
            // matching failure (MATLAB semantics).
            out.push(
              RTV.char(matchFailure ? "Matching failure in format." : "")
            );
          }
          if (nout >= 4) {
            // nextindex: 1-based position of the next unscanned character.
            out.push(RTV.num(strPos + 1));
          }
          return out;
        }
        return vals;
      },
    };
  },
});

// ── strtok ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "strtok",
  resolve: (argTypes, nargout) => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (!isTextType(argTypes[0])) return null;
    if (argTypes.length === 2 && !isTextType(argTypes[1])) return null;
    const outputTypes: JitType[] = [{ kind: "char" }];
    if (nargout >= 2) outputTypes.push({ kind: "char" });
    return {
      outputTypes,
      apply: (args, nout) => {
        const s = toString(args[0]);
        const delims = args.length >= 2 ? toString(args[1]) : " \t\n\r\f\v";
        let start = 0;
        while (start < s.length && delims.includes(s[start])) start++;
        if (start >= s.length) {
          if (nout <= 1) return RTV.char("");
          return [RTV.char(""), RTV.char("")];
        }
        let end = start;
        while (end < s.length && !delims.includes(s[end])) end++;
        const token = s.substring(start, end);
        if (nout <= 1) return RTV.char(token);
        return [RTV.char(token), RTV.char(s.substring(end))];
      },
    };
  },
});

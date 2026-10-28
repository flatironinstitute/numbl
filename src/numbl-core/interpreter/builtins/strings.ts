/**
 * Interpreter builtins for string/char operations.
 *
 * strsplit, strjoin, regexp/regexpi, and symvar live in string-extras.ts.
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
  isRuntimeSparseMatrix,
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
import { builtin, type BuiltinApply } from "./builtin.js";
import { builtinError, numericArg, realScalarArg } from "./args.js";
import {
  className,
  isText,
  isTextOrCell,
  mustBeTextError,
  unsupportedClassError,
  unsupportedFormError,
} from "./text-args.js";
import { sprintfFormat } from "../../helpers/string.js";
import { scanFormat } from "../../helpers/scanFormat.js";
import { matlabNumToString } from "../../runtime/utils.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

// ── Argument helpers ──────────────────────────────────────────────────

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

/** A text result of the same kind (char or string) as the input `v`. */
function sameTextKind(v: RuntimeValue, s: string): RuntimeValue {
  return isRuntimeChar(v) ? RTV.char(s) : RTV.string(s);
}

/** The text of a char or string-scalar subject (first) argument. String
 *  arrays and cell arrays are not implemented for these functions. */
function scalarTextSubject(name: string, v: RuntimeValue): string {
  if (isText(v)) return toString(v);
  if (isRuntimeStringArray(v) || isRuntimeCell(v))
    throw unsupportedClassError(name, v);
  throw mustBeTextError();
}

/** The text of a char or string-scalar argument other than the subject. */
function textArg(name: string, v: RuntimeValue): string {
  if (isText(v)) return toString(v);
  throw unsupportedClassError(name, v);
}

/** A 1-arg text→text function that preserves char/string and maps
 *  element-wise over cell arrays of text and string arrays. */
function textMap(
  fn: (s: string) => string,
  badInput: (v: RuntimeValue) => RuntimeError
): BuiltinApply {
  return args => {
    if (!isTextOrCell(args[0])) throw badInput(args[0]);
    return applyTextFn(args[0], fn);
  };
}

// ── Text→text (type-preserving) ───────────────────────────────────────

// MATLAB's lower/upper return non-text input unchanged.
builtin({
  name: "lower",
  nargin: 1,
  apply: args => applyTextFn(args[0], s => s.toLowerCase()),
});

builtin({
  name: "upper",
  nargin: 1,
  apply: args => applyTextFn(args[0], s => s.toUpperCase()),
});

builtin({
  name: "strtrim",
  nargin: 1,
  apply: textMap(
    s => s.trim(),
    () =>
      builtinError(
        "MATLAB:strtrim:InputClass",
        "Input should be a string, character array, or a cell array of character arrays."
      )
  ),
});

builtin({
  name: "deblank",
  nargin: 1,
  apply: textMap(
    s => s.replace(/\s+$/, ""),
    v => unsupportedClassError("deblank", v)
  ),
});

builtin({
  name: "reverse",
  nargin: 1,
  mfile: true,
  apply: textMap(s => s.split("").reverse().join(""), mustBeTextError),
});

// ── strip ─────────────────────────────────────────────────────────────

/** Remove leading (side 'left'), trailing ('right') or both runs of
 *  characters satisfying `test`. */
function stripSides(
  s: string,
  side: string,
  test: (c: string) => boolean
): string {
  let a = 0;
  let b = s.length;
  if (side !== "right") while (a < b && test(s[a])) a++;
  if (side !== "left") while (b > a && test(s[b - 1])) b--;
  return s.slice(a, b);
}

// strip(str), strip(str, side), strip(str, stripCharacter),
// strip(str, side, stripCharacter)
builtin({
  name: "strip",
  nargin: [1, 3],
  mfile: true,
  apply: args => {
    const v = args[0];
    if (!isTextOrCell(v)) throw mustBeTextError();
    let side = "both";
    let ch: string | null = null;
    for (let i = 1; i < args.length; i++) {
      const opt = isText(args[i]) ? toString(args[i]) : null;
      const lc = opt?.toLowerCase();
      if (i === 1 && (lc === "left" || lc === "right" || lc === "both"))
        side = lc;
      else if (opt !== null && opt.length === 1) ch = opt;
      else
        throw builtinError(
          "MATLAB:string:OptionMustBeOneOf",
          "Option must be 'left', 'right', or 'both', or it must be exactly one character."
        );
    }
    const test =
      ch === null ? (c: string) => /\s/.test(c) : (c: string) => c === ch;
    return applyTextFn(v, s => stripSides(s, side, test));
  },
});

// ── pad ───────────────────────────────────────────────────────────────

// pad(str), pad(str, numberOfCharacters), pad(..., side),
// pad(..., padCharacter)
builtin({
  name: "pad",
  nargin: [1, 4],
  mfile: true,
  apply: args => {
    const v = args[0];
    const s = scalarTextSubject("pad", v);
    let i = 1;
    let n = s.length;
    if (i < args.length && !isText(args[i])) {
      n = Math.round(realScalarArg(args[i], "Number of characters"));
      i++;
    }
    let side = "right";
    if (i < args.length) {
      const lc = textArg("pad", args[i]).toLowerCase();
      if (lc === "left" || lc === "right" || lc === "both") {
        side = lc;
        i++;
      }
    }
    let ch = " ";
    if (i < args.length) {
      const c = textArg("pad", args[i]);
      if (c.length !== 1)
        throw new RuntimeError("pad: padCharacter must be a single character");
      ch = c;
      i++;
    }
    if (i < args.length)
      throw new RuntimeError(`pad: invalid argument at position ${i + 1}`);
    if (s.length >= n) return sameTextKind(v, s);
    const total = n - s.length;
    let result: string;
    if (side === "both") {
      const left = Math.floor(total / 2);
      const right = total - left;
      result = ch.repeat(left) + s + ch.repeat(right);
    } else if (side === "left") {
      result = ch.repeat(total) + s;
    } else {
      result = s + ch.repeat(total);
    }
    return sameTextKind(v, result);
  },
});

// ── erase ─────────────────────────────────────────────────────────────

builtin({
  name: "erase",
  nargin: 2,
  mfile: true,
  apply: args => {
    if (!isTextOrCell(args[0])) throw mustBeTextError();
    const p = args[1];
    if (!isText(p)) {
      if (isRuntimeStringArray(p) || isRuntimeCell(p))
        throw unsupportedFormError("erase", "a pattern list");
      throw builtinError(
        "MATLAB:string:MatchMustBeStringCharOrCellArrayOfChars",
        "Text to replace must be a text or pattern array."
      );
    }
    const pat = toString(p);
    return applyTextFn(args[0], s => s.split(pat).join(""));
  },
});

// ── strrep / replace ──────────────────────────────────────────────────

function strreplaceApply(
  name: string,
  badSubject: (v: RuntimeValue) => RuntimeError
): BuiltinApply {
  return args => {
    if (!isTextOrCell(args[0])) throw badSubject(args[0]);
    const old = textArg(name, args[1]);
    const rep = textArg(name, args[2]);
    return applyTextFn(args[0], s => s.split(old).join(rep));
  };
}

builtin({
  name: "strrep",
  nargin: 3,
  apply: strreplaceApply("strrep", v => unsupportedClassError("strrep", v)),
});

builtin({
  name: "replace",
  nargin: 3,
  mfile: true,
  apply: strreplaceApply("replace", mustBeTextError),
});

// ── regexprep ─────────────────────────────────────────────────────────

builtin({
  name: "regexprep",
  nargin: [3, Infinity],
  apply: args => {
    if (!isTextOrCell(args[0]))
      throw builtinError(
        "MATLAB:REGEXP:invalidInputs",
        "The 'STRING' input must be either a char row vector, a cell array of char row vectors, or a string array."
      );
    const pat = textArg("regexprep", args[1]);
    const rep = textArg("regexprep", args[2]);
    // `s` (dotAll): MATLAB's `.` matches newline by default, unlike JS.
    let flags = "gs";
    for (let i = 3; i < args.length; i++) {
      const opt = toString(args[i]).toLowerCase();
      if (opt === "ignorecase") flags += "i";
    }
    const re = new RegExp(pat, flags);
    return applyTextFn(args[0], s => s.replace(re, rep));
  },
});

// ── extractBefore / extractAfter ──────────────────────────────────────

/** A position (numeric scalar) argument, or null for a text pattern. */
function positionArg(name: string, arg: RuntimeValue): number | null {
  if (isRuntimeNumber(arg) || (isRuntimeTensor(arg) && arg.data.length === 1))
    return Math.round(toNumber(arg));
  if (isText(arg)) return null;
  throw unsupportedClassError(name, arg);
}

function extractApply(
  name: string,
  extractFn: (s: string, pos: number | null, pat: string) => string
): BuiltinApply {
  return args => {
    const v = args[0];
    const s = scalarTextSubject(name, v);
    const pos = positionArg(name, args[1]);
    const pat = pos === null ? toString(args[1]) : "";
    return sameTextKind(v, extractFn(s, pos, pat));
  };
}

builtin({
  name: "extractBefore",
  nargin: 2,
  mfile: true,
  apply: extractApply("extractBefore", (s, p, pat) => {
    let pos: number;
    if (p !== null) {
      pos = p - 1;
    } else {
      pos = s.indexOf(pat);
      if (pos === -1)
        throw new RuntimeError("extractBefore: pattern not found");
    }
    return s.substring(0, pos);
  }),
});

builtin({
  name: "extractAfter",
  nargin: 2,
  mfile: true,
  apply: extractApply("extractAfter", (s, p, pat) => {
    let pos: number;
    if (p !== null) {
      pos = p;
    } else {
      const idx = s.indexOf(pat);
      if (idx === -1) throw new RuntimeError("extractAfter: pattern not found");
      pos = idx + pat.length;
    }
    return s.substring(pos);
  }),
});

// ── extractBetween ────────────────────────────────────────────────────

builtin({
  name: "extractBetween",
  nargin: [3, Infinity],
  mfile: true,
  apply: args => {
    if (args.length > 3)
      throw unsupportedFormError("extractBetween", "name-value options");
    const v = args[0];
    const s = scalarTextSubject("extractBetween", v);
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
    return sameTextKind(v, s.substring(startPos, endPos));
  },
});

// ── insertBefore / insertAfter ────────────────────────────────────────

function insertApply(
  name: string,
  insertFn: (s: string, arg: RuntimeValue, ins: string) => string
): BuiltinApply {
  return args => {
    const v = args[0];
    const s = scalarTextSubject(name, v);
    const ins = toString(args[2]);
    return sameTextKind(v, insertFn(s, args[1], ins));
  };
}

builtin({
  name: "insertBefore",
  nargin: 3,
  mfile: true,
  apply: insertApply("insertBefore", (s, arg, ins) => {
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

builtin({
  name: "insertAfter",
  nargin: 3,
  mfile: true,
  apply: insertApply("insertAfter", (s, arg, ins) => {
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

builtin({
  name: "strcmp",
  nargin: 2,
  apply: args => strcmpApply(args, (a, b) => a === b),
});

builtin({
  name: "strcmpi",
  nargin: 2,
  apply: args =>
    strcmpApply(args, (a, b) => a.toLowerCase() === b.toLowerCase()),
});

builtin({
  name: "strncmp",
  nargin: 3,
  apply: args => {
    if (!isText(args[0]) || !isText(args[1])) return RTV.logical(false);
    const n = Math.round(toNumber(args[2]));
    const s1 = toString(args[0]);
    const s2 = toString(args[1]);
    return RTV.logical(s1.substring(0, n) === s2.substring(0, n));
  },
});

builtin({
  name: "strncmpi",
  nargin: 3,
  apply: args => {
    if (!isText(args[0]) || !isText(args[1])) return RTV.logical(false);
    const n = Math.round(toNumber(args[2]));
    const s1 = toString(args[0]).toLowerCase();
    const s2 = toString(args[1]).toLowerCase();
    return RTV.logical(s1.substring(0, n) === s2.substring(0, n));
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
  if (!isText(args[1]))
    throw builtinError(
      "MATLAB:isstrprop:InvalidQualifierType",
      "Invalid data type in second argument."
    );
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
    result = logicalRowFromString(
      v.value,
      pred,
      v.shape ? [...v.shape] : undefined
    );
  } else if (isRuntimeTensor(v)) {
    result = logicalFromNumericTensor(v.data, v.shape, pred);
  } else if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
    const cp = Math.round(toNumber(v));
    const data = allocFloat64Array(1);
    data[0] = pred(cp) ? 1 : 0;
    result = new RuntimeTensor(data, [1, 1], undefined, true);
  } else {
    throw unsupportedClassError("isstrprop", v);
  }

  if (forceCell) return RTV.cell([result], [1, 1]);
  return result;
}

builtin({
  name: "isstrprop",
  help: {
    signatures: [
      "TF = isstrprop(str, category)",
      "TF = isstrprop(str, category, 'ForceCellOutput', tf)",
    ],
    description:
      "Test which characters in str belong to a category (alpha, alphanum, cntrl, digit, graphic, lower, print, punct, upper, wspace, xdigit). Returns a logical array, or a cell of logical vectors when str is a cell array or ForceCellOutput is true. Numeric input is treated as Unicode code points.",
  },
  nargin: [2, 4],
  apply: args => isstrpropApply(args),
});

builtin({
  name: "isspace",
  help: {
    signatures: ["TF = isspace(str)"],
    description:
      "Return a logical array the same size as str, true where the character is whitespace. Numeric input is treated as Unicode code points.",
  },
  nargin: 1,
  apply: args => {
    const pred = (cp: number) => RE_WSPACE.test(String.fromCodePoint(cp));
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
    if (isRuntimeNumber(v) || isRuntimeLogical(v))
      return logicalFromNumericTensor([toNumber(v)], [1, 1], pred);
    throw unsupportedClassError("isspace", v);
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

/** The 'IgnoreCase' value from trailing name-value pairs (other names are
 *  ignored). */
function ignoreCaseOption(args: RuntimeValue[], start: number): boolean {
  let ic = false;
  for (let i = start; i + 1 < args.length; i += 2) {
    if (toString(args[i]).toLowerCase() === "ignorecase") {
      ic = !!toNumber(args[i + 1]);
    }
  }
  return ic;
}

// contains / startsWith / endsWith (str, pat, 'IgnoreCase', tf)
function defineTextMatch(
  name: string,
  testFn: (s: string, pat: string) => boolean
): void {
  builtin({
    name,
    nargin: [2, Infinity],
    apply: args => {
      if (!isTextOrCell(args[0])) throw mustBeTextError();
      const ic = ignoreCaseOption(args, 2);
      let pats = patternList(args[1]);
      if (ic) pats = pats.map(p => p.toLowerCase());
      return textSubjectApply(args[0], s0 => {
        const s = ic ? s0.toLowerCase() : s0;
        return pats.some(p => testFn(s, p));
      });
    },
  });
}

defineTextMatch("contains", (s, p) => s.includes(p));
defineTextMatch("startsWith", (s, p) => s.startsWith(p));
defineTextMatch("endsWith", (s, p) => s.endsWith(p));

// ── Text→number functions ─────────────────────────────────────────────

builtin({
  name: "strlength",
  nargin: 1,
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
    if (isRuntimeCell(v)) throw unsupportedClassError("strlength", v);
    throw mustBeTextError();
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

builtin({
  name: "str2double",
  nargin: 1,
  mfile: true,
  apply: args => {
    const v = args[0];
    if (isText(v)) return RTV.num(str2doubleScalar(toString(v)));
    if (isRuntimeCell(v)) {
      const data = allocFloat64Array(v.data.length);
      for (let i = 0; i < v.data.length; i++) {
        const el = v.data[i];
        data[i] = isText(el) ? str2doubleScalar(toString(el)) : NaN;
      }
      return RTV.tensor(data, v.shape.slice());
    }
    if (isRuntimeStringArray(v)) {
      return RTV.tensor(
        allocFloat64Array(v.data.map(s => str2doubleScalar(s))),
        [v.shape[0], v.shape[1]]
      );
    }
    throw unsupportedClassError("str2double", v);
  },
});

// NOTE: the real str2num is the special builtin in
// interpreterSpecialBuiltins.ts, which evaluates the text as `[<text>]`
// (handling vectors / matrices / arithmetic). It intercepts before this
// builtin, which only stays registered for builtin-listing/type presence
// and the degenerate fallback.
builtin({
  name: "str2num",
  nargin: [1, 3],
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    if (args.length > 1)
      throw unsupportedFormError("str2num", "the 'Evaluation' option");
    if (!isText(args[0]))
      throw builtinError(
        "MATLAB:str2num:InvalidArgument",
        "Input must be a character vector or string scalar."
      );
    const n = Number(toString(args[0]));
    const ok = !Number.isNaN(n);
    const x = ok ? RTV.num(n) : RTV.tensor(allocFloat64Array(0), [0, 0]);
    return nargout >= 2 ? [x, RTV.logical(ok)] : x;
  },
});

builtin({
  name: "count",
  nargin: [2, Infinity],
  apply: args => {
    const s0 = scalarTextSubject("count", args[0]);
    const p = args[1];
    if (!isText(p)) {
      if (isRuntimeStringArray(p) || isRuntimeCell(p))
        throw unsupportedFormError("count", "a pattern list");
      throw builtinError(
        "MATLAB:string:MustBeTextOrPattern",
        "Search term must be a text or pattern array."
      );
    }
    const ic = ignoreCaseOption(args, 2);
    const s = ic ? s0.toLowerCase() : s0;
    const pat = ic ? toString(p).toLowerCase() : toString(p);
    if (pat.length === 0) return RTV.num(s.length + 1);
    let n = 0;
    let idx = 0;
    while ((idx = s.indexOf(pat, idx)) !== -1) {
      n++;
      idx += pat.length;
    }
    return RTV.num(n);
  },
});

builtin({
  name: "hex2dec",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (!isText(v)) {
      if (isRuntimeStringArray(v) || isRuntimeCell(v))
        throw unsupportedClassError("hex2dec", v);
      throw builtinError(
        "MATLAB:hex2dec:InvalidCharacters",
        "Hexadecimal text must consist of characters 0-9 and A-F."
      );
    }
    return RTV.num(parseInt(toString(v), 16));
  },
});

builtin({
  name: "bin2dec",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (!isText(v)) {
      if (isRuntimeStringArray(v) || isRuntimeCell(v))
        throw unsupportedClassError("bin2dec", v);
      throw builtinError(
        "MATLAB:bin2dec:InputMustBeString",
        "Input argument must be a character vector, string, or cell array of character vectors."
      );
    }
    return RTV.num(parseInt(toString(v), 2));
  },
});

builtin({
  name: "strfind",
  nargin: [2, 4],
  apply: args => {
    if (args.length > 2)
      throw unsupportedFormError("strfind", "the 'ForceCellOutput' option");
    const s = textArg("strfind", args[0]);
    const pattern = textArg("strfind", args[1]);
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
});

// ── Number/tensor→char functions ──────────────────────────────────────

function isNumericValue(v: RuntimeValue): boolean {
  return (
    isRuntimeNumber(v) ||
    isRuntimeLogical(v) ||
    isRuntimeTensor(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeSparseMatrix(v)
  );
}

builtin({
  name: "blanks",
  nargin: 1,
  mfile: true,
  apply: args => {
    const a = args[0];
    if (!isNumericValue(a))
      throw builtinError(
        "MATLAB:repmat:nonNumericReplications",
        "Replication factors must be numeric."
      );
    if (isRuntimeTensor(a) && a.data.length !== 1)
      throw builtinError(
        "MATLAB:NonScalarInput",
        "Size inputs must be scalar."
      );
    const n = Math.round(realScalarArg(a));
    return RTV.char(" ".repeat(Math.max(0, n)));
  },
});

/** The scalar decimal input of dec2hex / dec2bin. */
function decimalArg(
  name: string,
  v: RuntimeValue,
  badType: () => RuntimeError
): number {
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return toNumber(v);
  if (isRuntimeTensor(v) && !v.imag) {
    if (v.data.length === 1) return v.data[0];
    throw unsupportedFormError(name, "array input");
  }
  if (isRuntimeChar(v)) throw unsupportedClassError(name, v);
  if (isNumericValue(v))
    throw unsupportedFormError(name, "complex or sparse input");
  throw badType();
}

builtin({
  name: "dec2hex",
  nargin: [1, 2],
  apply: args => {
    let n = Math.round(
      decimalArg("dec2hex", args[0], () =>
        builtinError(
          "MATLAB:dec2hex:FirstArgIsInvalid",
          "First argument must contain finite integers."
        )
      )
    );
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
});

builtin({
  name: "dec2bin",
  nargin: [1, 2],
  apply: args => {
    let n = Math.round(
      decimalArg("dec2bin", args[0], () =>
        builtinError("MATLAB:dec2bin:InvalidDecimalArg", "D must be numeric.")
      )
    );
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
});

builtin({
  name: "int2str",
  nargin: 1,
  mfile: true,
  apply: args => {
    // char input is converted to its character codes, as in MATLAB.
    const v = numericArg(args[0]);
    if (
      isRuntimeComplexNumber(v) ||
      isRuntimeSparseMatrix(v) ||
      (isRuntimeTensor(v) && v.imag)
    )
      throw unsupportedFormError("int2str", "complex or sparse input");
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
    return RTV.char(String(roundInt(v)));
  },
});

// ── num2str ───────────────────────────────────────────────────────────

const numStr = matlabNumToString;

builtin({
  name: "num2str",
  nargin: [1, 2],
  mfile: true,
  apply: args => {
    const v = args[0];
    // MATLAB returns char input unchanged.
    if (isRuntimeChar(v)) return v;
    if (
      isRuntimeComplexNumber(v) ||
      isRuntimeSparseMatrix(v) ||
      (isRuntimeTensor(v) && v.imag)
    )
      throw unsupportedFormError("num2str", "complex or sparse input");
    if (!isRuntimeNumber(v) && !isRuntimeLogical(v) && !isRuntimeTensor(v))
      throw builtinError(
        "MATLAB:num2str:nonNumericInput",
        "Input to num2str must be numeric."
      );
    let precision: number | undefined;
    if (args.length >= 2) {
      const a1 = args[1];
      if (isText(a1)) return RTV.char(sprintfFormat(toString(a1), [v]));
      if (
        !isRuntimeNumber(a1) &&
        !(isRuntimeTensor(a1) && a1.data.length === 1 && !a1._isLogical)
      )
        throw builtinError(
          "MATLAB:num2str:invalidSecondArgument",
          "Second argument to num2str must be a character vector, string scalar, or numeric."
        );
      precision = toNumber(a1);
    }
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
  },
});

// ── mat2str ───────────────────────────────────────────────────────────

builtin({
  name: "mat2str",
  nargin: [1, 3],
  mfile: true,
  apply: args => {
    if (args.length > 2)
      throw unsupportedFormError("mat2str", "the 'class' option");
    const v = args[0];
    if (isText(v) || isRuntimeStringArray(v))
      throw unsupportedClassError("mat2str", v);
    if (isRuntimeSparseMatrix(v))
      throw unsupportedFormError("mat2str", "sparse input");
    if (
      !isRuntimeNumber(v) &&
      !isRuntimeLogical(v) &&
      !isRuntimeTensor(v) &&
      !isRuntimeComplexNumber(v)
    )
      throw builtinError(
        "MATLAB:mat2str:NumericInput",
        "Input matrix must be a numeric array, character array, or string array."
      );
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
    const fmtL = (x: number): string => (x ? "true" : "false");
    if (isRuntimeComplexNumber(v)) return RTV.char(fmtC(v.re, v.im));
    if (isRuntimeNumber(v)) return RTV.char(fmt(v));
    if (isRuntimeLogical(v)) return RTV.char(fmtL(v ? 1 : 0));
    const im = v.imag;
    const cell = (k: number): string =>
      im !== undefined
        ? fmtC(v.data[k], im[k])
        : v._isLogical
          ? fmtL(v.data[k])
          : fmt(v.data[k]);
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
builtin({
  name: "strmatch",
  help: {
    signatures: [
      "x = strmatch(str, strarray)",
      "x = strmatch(str, strarray, 'exact')",
    ],
    description:
      "(Legacy) Find rows of STRARRAY that begin with STR, or that equal STR when 'exact' is given. Returns a column vector of matching row indices.",
  },
  nargin: [2, 3],
  mfile: true,
  apply: args => {
    const str = textArg("strmatch", args[0]);
    if (!isText(args[1]) && !isRuntimeCell(args[1]))
      throw unsupportedClassError("strmatch", args[1]);
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

function charConversionError(v: RuntimeValue): RuntimeError {
  return builtinError(
    "MATLAB:invalidConversion",
    `Conversion to char from ${className(v)} is not possible.`
  );
}

// char(), char(A), char(A1, ..., An)
builtin({
  name: "char",
  apply: args => {
    if (args.length === 0) return RTV.char("");
    if (args.length > 1) {
      // Each argument becomes one or more rows (an empty one a blank row),
      // right-padded to the widest row.
      const rows: string[] = [];
      for (const a of args) {
        if (isRuntimeCell(a))
          throw builtinError(
            "MATLAB:character:InvalidInput",
            "Inputs must be character arrays."
          );
        if (isRuntimeStringArray(a)) rows.push(...a.data);
        else if (
          isText(a) ||
          isRuntimeNumber(a) ||
          (isRuntimeTensor(a) && !a.imag)
        )
          rows.push(...valueToCharRows(a));
        else throw charConversionError(a);
      }
      return charRowsToMatrix(rows);
    }
    const v = args[0];
    if (isRuntimeChar(v)) return v;
    // char(datetime) / char(duration): the display text (honoring a
    // datetime's Format property).
    if (isRuntimeClassInstance(v)) {
      const s = formatDatetimeOrDuration(v);
      if (s !== null) return RTV.char(s);
    }
    if (isRuntimeString(v)) return RTV.char(v);
    if (isRuntimeNumber(v)) return RTV.char(String.fromCharCode(Math.round(v)));
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
    throw charConversionError(v);
  },
});

builtin({
  name: "string",
  nargin: [0, 1],
  apply: args => {
    if (args.length === 0) return RTV.string("");
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
        v.data.length === 0 ? (v.shape[1] ?? 0) : v.data.length / (rows || 1);
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
        throw new RuntimeError("string: cell elements must be text or numbers");
      });
      return stringArrayValue(out, [rows, cols]);
    }
    return RTV.string(displayValue(v));
  },
});

builtin({
  name: "strings",
  help: {
    signatures: ["strings", "strings(n)", "strings(m,n)", "strings([m n])"],
    description:
      'Create a string array with every element set to "" (empty string).',
  },
  apply: args => {
    const dims: number[] = [];
    for (const rv of args) {
      if (isRuntimeTensor(rv)) {
        for (const d of rv.data) dims.push(Math.max(0, Math.round(d)));
      } else if (isRuntimeNumber(rv) || isRuntimeLogical(rv)) {
        dims.push(Math.max(0, Math.round(toNumber(rv))));
      } else {
        throw builtinError(
          "MATLAB:class:MustBeNumeric",
          "Array must be numeric or logical."
        );
      }
    }
    if (dims.length === 0) dims.push(1, 1);
    if (dims.length === 1) dims.push(dims[0]);
    const m = dims[0];
    const n = dims.slice(1).reduce((a, b) => a * b, 1);
    return stringArrayValue(new Array<string>(m * n).fill(""), [m, n]);
  },
});

builtin({
  name: "newline",
  help: {
    signatures: ["newline"],
    description: "Newline character, char(10).",
  },
  nargin: 0,
  apply: () => RTV.char("\n"),
});

// ── split / join (string arrays) ──────────────────────────────────────

builtin({
  name: "split",
  help: {
    signatures: ["split(str)", "split(str, delimiter)"],
    description:
      "Split text at whitespace (or the given delimiter(s)) into a string array. " +
      "A scalar input yields a column of parts.",
  },
  nargin: [1, 3],
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    if (!isTextOrCell(args[0])) throw mustBeTextError();
    if (args.length > 2)
      throw unsupportedFormError("split", "the dim argument");
    if (nargout > 1) throw unsupportedFormError("split", "the matches output");
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
});

builtin({
  name: "join",
  help: {
    signatures: ["join(str)", "join(str, delimiter)"],
    description:
      "Combine the elements of a string array into single strings, " +
      "separated by a space (or the given delimiter).",
  },
  nargin: [1, 3],
  mfile: true,
  apply: args => {
    if (!isTextOrCell(args[0])) throw mustBeTextError();
    if (args.length > 2) throw unsupportedFormError("join", "the dim argument");
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
});

builtin({
  name: "cellstr",
  help: {
    signatures: ["cellstr(A)"],
    description:
      "Convert a string array, char array, or cell array of text to a " +
      "cell array of character vectors.",
  },
  nargin: 1,
  mfile: true,
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
    throw builtinError(
      "MATLAB:invalidConversion",
      `Conversion to cellstr from ${className(v)} is not possible.`
    );
  },
});

builtin({
  name: "ismissing",
  help: {
    signatures: ["ismissing(A)"],
    description:
      "Logical array marking missing elements. numbl has no <missing> " +
      "value, so string inputs return all-false.",
  },
  nargin: [1, 2],
  mfile: true,
  apply: args => {
    if (args.length > 1)
      throw unsupportedFormError("ismissing", "the indicator argument");
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
});

// ── strcat ────────────────────────────────────────────────────────────

builtin({
  name: "strcat",
  nargin: [1, Infinity],
  mfile: true,
  apply: args => {
    if (args.some(a => isRuntimeCell(a) || isRuntimeStringArray(a)))
      throw unsupportedFormError("strcat", "cell array or string array input");
    if (!args.some(isText))
      throw unsupportedFormError("strcat", "input without text arguments");
    const anyString = args.some(a => isRuntimeString(a));
    if (anyString) {
      return RTV.string(args.map(a => toString(a)).join(""));
    }
    const parts = args.map(a => toString(a).replace(/[ \t\v\n\r\f]+$/, ""));
    return RTV.char(parts.join(""));
  },
});

// ── sprintf ───────────────────────────────────────────────────────────

builtin({
  name: "sprintf",
  nargin: [1, Infinity],
  nargout: 2,
  apply: (args, nargout) => {
    const f = args[0];
    if (!isText(f))
      throw builtinError("MATLAB:badformat_mx", "Invalid format.");
    const fmt = toString(f);
    // A string-array argument supplies one text arg per element
    // (the format cycles over them, like numeric arrays).
    const rest = args
      .slice(1)
      .flatMap(a =>
        isRuntimeStringArray(a) ? a.data.map(s => RTV.string(s)) : [a]
      );
    const result = sprintfFormat(fmt, rest);
    const out = isRuntimeChar(f) ? RTV.char(result) : RTV.string(result);
    // The second output (errmsg) is empty: formatting errors throw.
    return nargout >= 2 ? [out, RTV.char("")] : out;
  },
});

// ── sscanf ────────────────────────────────────────────────────────────

builtin({
  name: "sscanf",
  nargin: [2, 3],
  nargout: 4,
  apply: (args, nout) => {
    if (!isText(args[0]))
      throw builtinError(
        "MATLAB:badstringarg1_mx",
        "First argument must be a text scalar."
      );
    if (!isText(args[1]))
      throw builtinError("MATLAB:badformat_mx", "Invalid format.");
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
        out.push(RTV.char(matchFailure ? "Matching failure in format." : ""));
      }
      if (nout >= 4) {
        // nextindex: 1-based position of the next unscanned character.
        out.push(RTV.num(strPos + 1));
      }
      return out;
    }
    return vals;
  },
});

// ── strtok ────────────────────────────────────────────────────────────

builtin({
  name: "strtok",
  nargin: [1, 2],
  nargout: 2,
  mfile: true,
  apply: (args, nout) => {
    const s = textArg("strtok", args[0]);
    const delims =
      args.length >= 2 ? textArg("strtok", args[1]) : " \t\n\r\f\v";
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
});

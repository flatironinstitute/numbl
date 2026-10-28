/**
 * Validation builtins: mustBe* family.
 */

import type { RuntimeChar, RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeStringArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
} from "../../runtime/types.js";
import { toNumber, RuntimeError } from "../../runtime/index.js";
import { RTV } from "../../runtime/constructors.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { Runtime } from "../../runtime/runtime.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import { MATLAB_KEYWORDS } from "./introspection.js";
import { valueShape } from "../../runtime/utils.js";

function numericElements(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v as number];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  return [];
}

function numelVal(v: RuntimeValue): number {
  return valueShape(v).reduce((a, b) => a * b, 1);
}

/** Define a mustBe* validator: an M-file function in MATLAB, with no
 *  outputs. `check` throws when the value is invalid. */
function validator(
  name: string,
  nargin: number | [number, number],
  check: (args: RuntimeValue[], rt: Runtime) => void
): void {
  builtin({
    name,
    nargin,
    nargout: 0,
    mfile: true,
    apply: (args, _nargout, ctx) => check(args, ctx.rt),
  });
}

/** The error a mustBe* validator raises: `MATLAB:validators:<name>`. */
function fail(name: string, message: string): RuntimeError {
  return builtinError(`MATLAB:validators:${name}`, message);
}

/** Real and imaginary parts of a numeric, logical (or, with `allowChar`,
 *  char) value, or null for any other value. */
function numericParts(
  v: RuntimeValue,
  allowChar = false
): { re: ArrayLike<number>; im?: ArrayLike<number> } | null {
  if (isRuntimeNumber(v)) return { re: [v] };
  if (isRuntimeLogical(v)) return { re: [v ? 1 : 0] };
  if (isRuntimeTensor(v)) return { re: v.data, im: v.imag };
  if (isRuntimeComplexNumber(v)) return { re: [v.re], im: [v.im] };
  if (isRuntimeSparseMatrix(v)) {
    const re = new Float64Array(v.m * v.n);
    const im = v.pi ? new Float64Array(v.m * v.n) : undefined;
    for (let j = 0; j < v.n; j++) {
      for (let k = v.jc[j]; k < v.jc[j + 1]; k++) {
        re[j * v.m + v.ir[k]] = v.pr[k];
        if (im) im[j * v.m + v.ir[k]] = v.pi![k];
      }
    }
    return { re, im };
  }
  if (allowChar && isRuntimeChar(v)) {
    return { re: Array.from(v.value, c => c.charCodeAt(0)) };
  }
  return null;
}

/** Numeric parts, raising mustBeNumericOrLogical for anything else. */
function requireNumeric(v: RuntimeValue): {
  re: ArrayLike<number>;
  im?: ArrayLike<number>;
} {
  const p = numericParts(v);
  if (!p)
    throw fail("mustBeNumericOrLogical", "Value must be numeric or logical.");
  return p;
}

function isLogicalValue(v: RuntimeValue): boolean {
  return isRuntimeLogical(v) || (isRuntimeTensor(v) && !!v._isLogical);
}

/** An element-wise test on the real parts of a numeric or logical value. */
function elemValidator(
  name: string,
  test: (re: number, im: number) => boolean,
  message: string,
  allowChar = false
): void {
  validator(name, 1, ([v]) => {
    if (typeof v === "number") {
      if (!test(v, 0)) throw fail(name, message);
      return;
    }
    const p = allowChar ? numericParts(v, true) : requireNumeric(v);
    if (!p)
      throw fail("mustBeNumericOrLogical", "Value must be numeric or logical.");
    for (let i = 0; i < p.re.length; i++) {
      if (!test(p.re[i], p.im ? p.im[i] : 0)) throw fail(name, message);
    }
  });
}

elemValidator("mustBePositive", re => re > 0, "Value must be positive.");
elemValidator("mustBeNonnegative", re => re >= 0, "Value must be nonnegative.");
elemValidator(
  "mustBeNonpositive",
  re => re <= 0,
  "Value must not be positive."
);
elemValidator("mustBeNegative", re => re < 0, "Value must be negative.");
elemValidator(
  "mustBeNonzero",
  (re, im) => re !== 0 || im !== 0,
  "Value must not be zero."
);
elemValidator(
  "mustBeInteger",
  (re, im) => im === 0 && Number.isFinite(re) && Math.floor(re) === re,
  "Value must be integer."
);
elemValidator(
  "mustBeFinite",
  (re, im) => Number.isFinite(re) && Number.isFinite(im),
  "Value must be finite.",
  true
);
elemValidator(
  "mustBeNonNan",
  (re, im) => !Number.isNaN(re) && !Number.isNaN(im),
  "Value must not be NaN.",
  true
);

validator("mustBeReal", 1, ([v]) => {
  const p = numericParts(v);
  if (p?.im && Array.from(p.im).some(x => x !== 0))
    throw fail("mustBeReal", "Value must be real.");
});

validator("mustBeNumeric", 1, ([v]) => {
  if (!numericParts(v) || isLogicalValue(v))
    throw fail("mustBeNumeric", "Value must be numeric.");
});

validator("mustBeNumericOrLogical", 1, ([v]) => {
  requireNumeric(v);
});

validator("mustBeFloat", 1, ([v]) => {
  if (!numericParts(v) || isLogicalValue(v))
    throw fail("mustBeFloat", "Value must be a floating-point array.");
});

validator("mustBeNonempty", 1, ([v]) => {
  if (numelVal(v) === 0)
    throw fail("mustBeNonempty", "Value must not be empty.");
});

validator("mustBeNonsparse", 1, ([v]) => {
  if (isRuntimeSparseMatrix(v))
    throw fail("mustBeNonsparse", "Value must not be sparse.");
});

validator("mustBeScalarOrEmpty", 1, ([v]) => {
  const n = numelVal(v);
  if (n !== 0 && n !== 1)
    throw fail("mustBeScalarOrEmpty", "Value must be scalar or empty.");
});

// mustBeVector(v, 'allow-all-empties'): any empty is then accepted.
validator("mustBeVector", [1, 2], ([v, opt]) => {
  const shape = valueShape(v);
  if (opt !== undefined) {
    const o = textValue(opt);
    if (o?.toLowerCase() !== "allow-all-empties")
      throw builtinError(
        "MATLAB:validatorUsage:invalidOption",
        "Second input to function 'mustBeVector' must be 'allow-all-empties'."
      );
    if (numelVal(v) === 0) return;
  }
  if (shape.length !== 2 || (shape[0] !== 1 && shape[1] !== 1))
    throw fail(
      "mustBeVector",
      "Value must be a 1-by-n vector or an n-by-1 vector."
    );
});

validator("mustBeRow", 1, ([v]) => {
  const shape = valueShape(v);
  if (shape.length !== 2 || shape[0] !== 1)
    throw fail("mustBeRow", "Value must be a row vector.");
});

validator("mustBeColumn", 1, ([v]) => {
  const shape = valueShape(v);
  if (shape.length !== 2 || shape[1] !== 1)
    throw fail("mustBeColumn", "Value must be a column vector.");
});

validator("mustBeMatrix", 1, ([v]) => {
  if (valueShape(v).length !== 2)
    throw fail("mustBeMatrix", "Value must be a matrix.");
});

// ── Text validators ─────────────────────────────────────────────────────

const TEXT_MESSAGE =
  "Value must be a character vector, string array, or cell array of character vectors.";

/** A char row vector or string scalar as a JS string, else null. */
function textValue(v: RuntimeValue): string | null {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) {
    const shape = valueShape(v);
    return shape.length === 2 && shape[0] <= 1 ? v.value : null;
  }
  return null;
}

/** The strings of a text value (char row, string array or cellstr), or
 *  null when the value is not text. */
function textElements(v: RuntimeValue): string[] | null {
  const t = textValue(v);
  if (t !== null) return [t];
  if (isRuntimeStringArray(v)) return v.data;
  if (isRuntimeCell(v)) {
    const out: string[] = [];
    for (const e of v.data) {
      const ev = ensureRuntimeValue(e);
      if (!isRuntimeChar(ev)) return null;
      const s = textValue(ev);
      if (s === null) return null;
      out.push(s);
    }
    return out;
  }
  return null;
}

validator("mustBeText", 1, ([v]) => {
  if (textElements(v) === null) throw fail("mustBeText", TEXT_MESSAGE);
});

validator("mustBeTextScalar", 1, ([v]) => {
  if (textValue(v) === null)
    throw fail(
      "mustBeTextScalar",
      "Value must be a character vector or string scalar."
    );
});

validator("mustBeNonzeroLengthText", 1, ([v]) => {
  const els = textElements(v);
  if (els === null) throw fail("mustBeNonzeroLengthText", TEXT_MESSAGE);
  if (els.length === 0 || els.some(s => s.length === 0))
    throw fail(
      "mustBeNonzeroLengthText",
      "Value must be text with one or more characters."
    );
});

validator("mustBeValidVariableName", 1, ([v]) => {
  const els = textElements(v);
  if (els === null) throw fail("mustBeValidVariableName", TEXT_MESSAGE);
  const bad = els.filter(
    s => !/^[A-Za-z][A-Za-z0-9_]{0,62}$/.test(s) || MATLAB_KEYWORDS.has(s)
  );
  if (bad.length > 0)
    throw fail(
      "mustBeValidVariableName",
      `These are not valid variable names: ${bad.map(s => `'${s}'`).join(", ")}.`
    );
});

// ── Comparison validators ───────────────────────────────────────────────

/** Format a bound for a comparison validator's message (MATLAB's num2str). */
function formatBound(n: number): string {
  if (Number.isInteger(n)) return String(n);
  if (!Number.isFinite(n))
    return Number.isNaN(n) ? "NaN" : n > 0 ? "Inf" : "-Inf";
  return String(parseFloat(n.toPrecision(5)));
}

function comparisonValidator(
  name: string,
  test: (x: number, c: number) => boolean,
  phrase: string
): void {
  validator(name, 2, ([v, c]) => {
    const cp = numericParts(c);
    if (!cp || cp.re.length !== 1)
      throw builtinError(
        "MATLAB:validatorUsage:nonScalarSecondInput",
        `Second input to function '${name}' must be a scalar.`
      );
    const bound = cp.re[0];
    const p = requireNumeric(v);
    for (let i = 0; i < p.re.length; i++) {
      if (!test(p.re[i], bound))
        throw fail(name, `Value must be ${phrase} ${formatBound(bound)}.`);
    }
  });
}

comparisonValidator("mustBeGreaterThan", (x, c) => x > c, "greater than");
comparisonValidator(
  "mustBeGreaterThanOrEqual",
  (x, c) => x >= c,
  "greater than or equal to"
);
comparisonValidator("mustBeLessThan", (x, c) => x < c, "less than");
comparisonValidator(
  "mustBeLessThanOrEqual",
  (x, c) => x <= c,
  "less than or equal to"
);

// mustBeInRange(value, lower, upper[, boundflag1[, boundflag2]])
validator("mustBeInRange", [3, 5], ([v, lo, hi, ...flags]) => {
  const scalar = (x: RuntimeValue, which: string): number => {
    const p = numericParts(x);
    if (!p || p.re.length !== 1)
      throw builtinError(
        "MATLAB:validatorUsage:nonScalarInput",
        `${which} input to function 'mustBeInRange' must be a scalar.`
      );
    return p.re[0];
  };
  const lower = scalar(lo, "Second");
  const upper = scalar(hi, "Third");
  let lowerOpen = false;
  let upperOpen = false;
  for (const f of flags) {
    switch (textValue(f)?.toLowerCase()) {
      case "inclusive":
        break;
      case "exclusive":
        lowerOpen = upperOpen = true;
        break;
      case "exclude-lower":
        lowerOpen = true;
        break;
      case "exclude-upper":
        upperOpen = true;
        break;
      default:
        throw builtinError(
          "MATLAB:validatorUsage:invalidBoundFlag",
          "Bound flag must be 'inclusive', 'exclusive', 'exclude-lower', or 'exclude-upper'."
        );
    }
  }
  const p = requireNumeric(v);
  const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(15));
  for (let i = 0; i < p.re.length; i++) {
    const x = p.re[i];
    const ok =
      (lowerOpen ? x > lower : x >= lower) &&
      (upperOpen ? x < upper : x <= upper);
    if (!ok) {
      throw fail(
        "mustBeInRange",
        `Value must be greater than ${lowerOpen ? "" : "or equal to "}${fmt(lower)}, and less than ${upperOpen ? "" : "or equal to "}${fmt(upper)}.`
      );
    }
  }
});

// ── Membership and class ────────────────────────────────────────────────

validator("mustBeMember", 2, ([v, set]) => {
  const setText = isRuntimeChar(set) ? null : textElements(set);
  const vText = textElements(v);
  if (setText !== null || (isRuntimeChar(set) && vText !== null)) {
    // Text membership (case-sensitive). A char set is matched character-
    // wise, like ismember.
    const members = setText ?? Array.from((set as RuntimeChar).value);
    const items =
      setText === null && isRuntimeChar(v) ? Array.from(v.value) : vText;
    if (items !== null && items.every(s => members.includes(s))) return;
    const shown = setText ?? [(set as RuntimeChar).value];
    throw fail(
      "mustBeMember",
      `Value must be a member of this set:\n${shown.map(s => `    '${s}'\n`).join("")}`
    );
  }
  const sp = numericParts(set, true);
  const vp = numericParts(v, true);
  if (!sp) {
    throw fail(
      "mustBeMemberGenericText",
      "Value must be members of the required set."
    );
  }
  const members = Array.from(sp.re);
  if (vp && Array.from(vp.re).every(x => members.includes(x))) return;
  throw fail(
    "mustBeMember",
    `Value must be a member of this set:\n${members.map(n => `    ${formatBound(n)}\n`).join("")}`
  );
});

/** Class names given as char, string array or cellstr. */
function classNamesArg(v: RuntimeValue, name: string): string[] {
  const els = textElements(v);
  if (els === null)
    throw builtinError(
      "MATLAB:validatorUsage:invalidClassNames",
      `Second input to function '${name}' must be a string array, character vector, or cell array of character vectors.`
    );
  return els;
}

validator("mustBeA", 2, ([v, classes], rt) => {
  const names = classNamesArg(classes, "mustBeA");
  for (const c of names) {
    if (rt.isa(v, RTV.char(c))) return;
  }
  const q = names.map(n => `'${n}'`);
  const list =
    q.length <= 2
      ? q.join(" or ")
      : `${q.slice(0, -1).join(", ")}, or ${q[q.length - 1]}`;
  throw fail("mustBeA", `Value must be one of these types: ${list}.`);
});

validator("mustBeUnderlyingType", 2, ([v, types], rt) => {
  const names = classNamesArg(types, "mustBeUnderlyingType");
  for (const c of names) {
    if (rt.isa(v, RTV.char(c))) return;
  }
  throw fail(
    "mustBeUnderlyingType",
    `Value must have one of these underlying types: ${names.join(", ")}.`
  );
});

// ── validateattributes ─────────────────────────────────────────────────

function getValClassName(v: RuntimeValue): string {
  if (typeof v === "number") return "double";
  if (typeof v === "boolean") return "logical";
  if (typeof v === "string") return "string";
  const obj = v as { kind: string; _isLogical?: boolean; className?: string };
  switch (obj.kind) {
    case "tensor":
      return obj._isLogical ? "logical" : "double";
    case "complex_number":
      return "double";
    case "char":
      return "char";
    case "cell":
      return "cell";
    case "struct":
    case "struct_array":
      return "struct";
    case "function":
      return "function_handle";
    case "class_instance":
    case "class_instance_array":
      return obj.className ?? "unknown";
    case "sparse_matrix":
      return "double";
    case "dictionary":
      return "dictionary";
    case "string_array":
      return "string";
    default:
      return "unknown";
  }
}

function matchesOneClass(v: RuntimeValue, cls: string): boolean {
  const vc = getValClassName(v);
  if (vc === cls) return true;
  if (cls === "numeric" && vc === "double") return true;
  return false;
}

function extractStr(v: RuntimeValue): string {
  if (typeof v === "string") return v;
  if (isRuntimeChar(v)) return v.value;
  throw new RuntimeError("Expected character vector or string");
}

function checkColumnMonotonicity(
  A: RuntimeValue,
  shape: number[],
  cmp: (prev: number, curr: number) => boolean,
  desc: string,
  errFn: (msg: string) => never
): void {
  if (isRuntimeComplexNumber(A)) errFn(`be ${desc}.`);
  const elems = numericElements(A);
  if (elems.length <= 1) return;
  for (const e of elems) {
    if (Number.isNaN(e)) errFn(`be ${desc}.`);
  }
  const nrows = shape[0] ?? 1;
  const ncols = Math.floor(elems.length / nrows);
  for (let c = 0; c < ncols; c++) {
    for (let r = 1; r < nrows; r++) {
      if (!cmp(elems[c * nrows + r - 1], elems[c * nrows + r])) {
        errFn(`be ${desc}.`);
      }
    }
  }
}

builtin({
  name: "validateattributes",
  nargin: [3, 6],
  nargout: 0,
  apply: args => {
    const A = args[0];
    const classesArg = args[1];
    const attrsArg = args[2];

    // Parse optional arguments
    let funcName: string | undefined;
    let varName: string | undefined;
    let argIndex: number | undefined;

    if (args.length === 4) {
      if (isRuntimeNumber(args[3])) {
        argIndex = args[3] as number;
      } else {
        funcName = extractStr(args[3]);
      }
    } else if (args.length === 5) {
      funcName = extractStr(args[3]);
      varName = extractStr(args[4]);
    } else if (args.length >= 6) {
      funcName = extractStr(args[3]);
      varName = extractStr(args[4]);
      argIndex = toNumber(args[5]);
    }

    if (funcName === "") funcName = undefined;
    if (varName === "") varName = undefined;

    // Build error subject
    let subject = "input";
    if (argIndex !== undefined && varName) {
      subject = `input number ${argIndex}, ${varName},`;
    } else if (argIndex !== undefined) {
      subject = `input number ${argIndex}`;
    } else if (varName) {
      subject = `input '${varName}'`;
    }

    const errFn = (msg: string): never => {
      throw new RuntimeError(`Expected ${subject} to ${msg}`);
    };

    // --- Class check ---
    let classNames: string[] = [];
    if (isRuntimeCell(classesArg)) {
      classNames = classesArg.data.map(extractStr);
    } else if (isRuntimeChar(classesArg)) {
      classNames = [classesArg.value];
    } else if (isRuntimeString(classesArg)) {
      classNames = [classesArg as string];
    }

    if (classNames.length > 0) {
      if (!classNames.some(cls => matchesOneClass(A, cls))) {
        const vc = getValClassName(A);
        errFn(
          `be one of these types:\n\n${classNames.join(", ")}\n\nInstead its type was ${vc}.`
        );
      }
    }

    // --- Attribute check ---
    if (!isRuntimeCell(attrsArg)) {
      throw new RuntimeError(
        "validateattributes: attributes must be a cell array"
      );
    }

    const shape = valueShape(A);
    // Effective ndims: remove trailing singletons, min 2
    let effNdims = shape.length;
    while (effNdims > 2 && shape[effNdims - 1] === 1) effNdims--;
    effNdims = Math.max(effNdims, 2);

    const nrows = shape[0] ?? 1;
    const ncols = shape.length >= 2 ? shape[1] : 1;
    const numel = shape.reduce((a, b) => a * b, 1);
    const attrData = attrsArg.data;

    let ai = 0;
    while (ai < attrData.length) {
      const attrName = extractStr(attrData[ai]);
      ai++;

      switch (attrName) {
        // ── Shape / size ──
        case "2d":
          if (effNdims > 2) errFn("be two-dimensional.");
          break;
        case "3d":
          if (effNdims > 3) errFn("have 3 or fewer dimensions.");
          break;
        case "column":
          if (effNdims > 2 || ncols !== 1) errFn("be a column vector.");
          break;
        case "row":
          if (effNdims > 2 || nrows !== 1) errFn("be a row vector.");
          break;
        case "scalar":
          if (numel !== 1) errFn("be scalar.");
          break;
        case "scalartext": {
          let ok = false;
          if (isRuntimeString(A)) ok = true;
          else if (isRuntimeChar(A)) {
            const cs = A.shape ?? [1, A.value.length];
            if (cs.length <= 2 && cs[0] === 1) ok = true;
          }
          if (!ok) errFn("be a character vector or string scalar.");
          break;
        }
        case "vector": {
          const isVec =
            numel <= 1 || (effNdims <= 2 && (nrows === 1 || ncols === 1));
          if (!isVec) errFn("be a vector.");
          break;
        }
        case "size": {
          if (ai >= attrData.length)
            throw new RuntimeError(
              "validateattributes: 'size' requires a value"
            );
          const sizeVal = attrData[ai];
          ai++;
          let expectedSize: number[];
          if (isRuntimeTensor(sizeVal)) {
            expectedSize = Array.from(sizeVal.data);
          } else if (isRuntimeNumber(sizeVal)) {
            expectedSize = [sizeVal as number];
          } else {
            throw new RuntimeError(
              "validateattributes: 'size' value must be numeric"
            );
          }
          for (let j = 0; j < expectedSize.length; j++) {
            if (Number.isNaN(expectedSize[j])) continue;
            const actual = j < shape.length ? shape[j] : 1;
            if (actual !== expectedSize[j]) {
              const expStr = expectedSize
                .map(x => (Number.isNaN(x) ? ":" : String(x)))
                .join("x");
              errFn(
                `be of size ${expStr} when it is actually size ${shape.join("x")}.`
              );
            }
          }
          break;
        }
        case "numel": {
          if (ai >= attrData.length)
            throw new RuntimeError(
              "validateattributes: 'numel' requires a value"
            );
          const n = toNumber(attrData[ai]);
          ai++;
          if (numel !== n) errFn(`have ${n} elements.`);
          break;
        }
        case "ncols": {
          if (ai >= attrData.length)
            throw new RuntimeError(
              "validateattributes: 'ncols' requires a value"
            );
          const n = toNumber(attrData[ai]);
          ai++;
          if (ncols !== n) errFn(`have ${n} columns.`);
          break;
        }
        case "nrows": {
          if (ai >= attrData.length)
            throw new RuntimeError(
              "validateattributes: 'nrows' requires a value"
            );
          const n = toNumber(attrData[ai]);
          ai++;
          if (nrows !== n) errFn(`have ${n} rows.`);
          break;
        }
        case "ndims": {
          if (ai >= attrData.length)
            throw new RuntimeError(
              "validateattributes: 'ndims' requires a value"
            );
          const n = toNumber(attrData[ai]);
          ai++;
          if (effNdims !== n) errFn(`be ${n}-dimensional.`);
          break;
        }
        case "square":
          if (effNdims !== 2 || nrows !== ncols) errFn("be square.");
          break;
        case "diag": {
          if (effNdims > 2 || nrows !== ncols) errFn("be diagonal.");
          const diagElems = numericElements(A);
          for (let r = 0; r < nrows; r++) {
            for (let c = 0; c < ncols; c++) {
              if (r !== c && diagElems[c * nrows + r] !== 0)
                errFn("be diagonal.");
            }
          }
          break;
        }
        case "nonempty":
          if (numel === 0) errFn("be nonempty.");
          break;
        case "nonsparse":
          if (isRuntimeSparseMatrix(A)) errFn("be nonsparse.");
          break;

        // ── Range ──
        case ">": {
          if (ai >= attrData.length)
            throw new RuntimeError("validateattributes: '>' requires a value");
          const n = toNumber(attrData[ai]);
          ai++;
          if (isRuntimeComplexNumber(A))
            errFn(`be an array with all of the values > ${n}.`);
          for (const e of numericElements(A)) {
            if (!(e > n)) errFn(`be an array with all of the values > ${n}.`);
          }
          break;
        }
        case ">=": {
          if (ai >= attrData.length)
            throw new RuntimeError("validateattributes: '>=' requires a value");
          const n = toNumber(attrData[ai]);
          ai++;
          if (isRuntimeComplexNumber(A))
            errFn(`be an array with all of the values >= ${n}.`);
          for (const e of numericElements(A)) {
            if (!(e >= n)) errFn(`be an array with all of the values >= ${n}.`);
          }
          break;
        }
        case "<": {
          if (ai >= attrData.length)
            throw new RuntimeError("validateattributes: '<' requires a value");
          const n = toNumber(attrData[ai]);
          ai++;
          if (isRuntimeComplexNumber(A))
            errFn(`be an array with all of the values < ${n}.`);
          for (const e of numericElements(A)) {
            if (!(e < n)) errFn(`be an array with all of the values < ${n}.`);
          }
          break;
        }
        case "<=": {
          if (ai >= attrData.length)
            throw new RuntimeError("validateattributes: '<=' requires a value");
          const n = toNumber(attrData[ai]);
          ai++;
          if (isRuntimeComplexNumber(A))
            errFn(`be an array with all of the values <= ${n}.`);
          for (const e of numericElements(A)) {
            if (!(e <= n)) errFn(`be an array with all of the values <= ${n}.`);
          }
          break;
        }
        case "finite":
          for (const e of numericElements(A)) {
            if (!Number.isFinite(e)) errFn("be finite.");
          }
          break;
        case "nonnan":
          for (const e of numericElements(A)) {
            if (Number.isNaN(e)) errFn("have no NaN values.");
          }
          break;

        // ── Type / value ──
        case "binary":
          for (const e of numericElements(A)) {
            if (e !== 0 && e !== 1) errFn("be binary.");
          }
          break;
        case "even":
          for (const e of numericElements(A)) {
            if (!Number.isFinite(e) || Math.floor(e) !== e || e % 2 !== 0)
              errFn("be even.");
          }
          break;
        case "odd":
          for (const e of numericElements(A)) {
            if (
              !Number.isFinite(e) ||
              Math.floor(e) !== e ||
              Math.abs(e) % 2 !== 1
            )
              errFn("be odd.");
          }
          break;
        case "integer":
          if (isRuntimeComplexNumber(A)) errFn("be integer.");
          for (const e of numericElements(A)) {
            if (!Number.isFinite(e) || Math.floor(e) !== e)
              errFn("be integer.");
          }
          break;
        case "real":
          if (isRuntimeComplexNumber(A) && (A as { im: number }).im !== 0)
            errFn("be real.");
          if (isRuntimeTensor(A) && A.imag) {
            for (const v of A.imag) {
              if (v !== 0) errFn("be real.");
            }
          }
          break;
        case "nonnegative":
          if (isRuntimeComplexNumber(A)) errFn("be nonnegative.");
          for (const e of numericElements(A)) {
            if (e < 0) errFn("be nonnegative.");
          }
          break;
        case "nonzero":
          for (const e of numericElements(A)) {
            if (e === 0) errFn("be nonzero.");
          }
          break;
        case "positive":
          if (isRuntimeComplexNumber(A)) errFn("be positive.");
          for (const e of numericElements(A)) {
            if (e <= 0) errFn("be positive.");
          }
          break;

        // ── Monotonicity (column-wise) ──
        case "increasing":
          checkColumnMonotonicity(
            A,
            shape,
            (a, b) => b > a,
            "strictly increasing",
            errFn
          );
          break;
        case "decreasing":
          checkColumnMonotonicity(
            A,
            shape,
            (a, b) => b < a,
            "strictly decreasing",
            errFn
          );
          break;
        case "nondecreasing":
          checkColumnMonotonicity(
            A,
            shape,
            (a, b) => b >= a,
            "nondecreasing",
            errFn
          );
          break;
        case "nonincreasing":
          checkColumnMonotonicity(
            A,
            shape,
            (a, b) => b <= a,
            "nonincreasing",
            errFn
          );
          break;

        default:
          throw new RuntimeError(
            `validateattributes: unknown attribute '${attrName}'`
          );
      }
    }
  },
});

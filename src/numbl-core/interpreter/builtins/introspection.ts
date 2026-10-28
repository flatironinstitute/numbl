/**
 * Introspection builtins: size, length, numel, ndims, isempty, isscalar,
 * isvector, isrow, iscolumn, ismatrix, islogical, isnumeric, isfloat,
 * isinteger, ischar, isstring, iscell, isstruct, issparse, class, isa,
 * superclasses, properties, ismethod.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeClassInstance,
  isRuntimeFunction,
  isRuntimeDummyHandle,
  isRuntimeGraphicsHandle,
  isRuntimeDictionary,
  isRuntimeClassInstanceArray,
  isRuntimeStringArray,
  RuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import type { Runtime } from "../../runtime/runtime.js";
import {
  MEXCEPTION_CLASS,
  MEXCEPTION_PROPERTIES,
} from "../../runtime/mexception.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toString } from "../../runtime/convert.js";
import { makeTensor } from "./types.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { incref } from "../../runtime/refcount.js";
import { valueShape } from "../../runtime/utils.js";

// ── Helpers ──────────────────────────────────────────────────────────────

/** A one-argument predicate builtin. */
function predicate(name: string, test: (v: RuntimeValue) => boolean): void {
  builtin({ name, nargin: 1, apply: args => test(args[0]) });
}

// ── Type predicates ──────────────────────────────────────────────────────

predicate("isnumeric", v => {
  if (typeof v === "number") return true;
  if (isRuntimeTensor(v) && !v._isLogical) return true;
  if (isRuntimeComplexNumber(v)) return true;
  if (isRuntimeSparseMatrix(v)) return true;
  if (isRuntimeClassInstance(v) && v._builtinData !== undefined) return true;
  return false;
});

predicate("isfloat", v => {
  if (typeof v === "number") return true;
  if (isRuntimeTensor(v) && !v._isLogical) return true;
  if (isRuntimeComplexNumber(v)) return true;
  if (isRuntimeSparseMatrix(v)) return true;
  return false;
});

predicate("isinteger", () => false);

predicate("islogical", v => {
  if (typeof v === "boolean") return true;
  if (isRuntimeTensor(v) && v._isLogical === true) return true;
  return false;
});

predicate("ischar", v => isRuntimeChar(v));

// Legacy alias for ischar (removed from modern MATLAB docs but still works).
predicate("isstr", v => isRuntimeChar(v));

predicate("isstring", v => isRuntimeString(v) || isRuntimeStringArray(v));

// MATLAB's reserved words (`iskeyword` with no args). Note true/false are
// functions, not keywords; spmd is included.
export const MATLAB_KEYWORDS: ReadonlySet<string> = new Set([
  "break",
  "case",
  "catch",
  "classdef",
  "continue",
  "else",
  "elseif",
  "end",
  "for",
  "function",
  "global",
  "if",
  "otherwise",
  "parfor",
  "persistent",
  "return",
  "spmd",
  "switch",
  "try",
  "while",
]);

/** Extract a single-row text value (char row vector or scalar string),
 *  or null if the value is not such text. */
function singleRowText(v: RuntimeValue): string | null {
  if (isRuntimeChar(v)) {
    if (v.shape && v.shape.length >= 1 && v.shape[0] > 1) return null;
    return v.value;
  }
  if (isRuntimeString(v)) return v;
  return null;
}

predicate("isvarname", v => {
  const s = singleRowText(v);
  if (s === null || s.length === 0 || s.length > 63) return false;
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(s)) return false;
  return !MATLAB_KEYWORDS.has(s);
});

builtin({
  name: "iskeyword",
  nargin: [0, 1],
  mfile: true,
  apply: args => {
    // iskeyword() with no args returns the list of keywords (column cell).
    if (args.length === 0) {
      const names = [...MATLAB_KEYWORDS].sort();
      return RTV.cell(
        names.map(n => RTV.char(n)),
        [names.length, 1]
      );
    }
    const s = singleRowText(args[0]);
    return RTV.logical(s !== null && MATLAB_KEYWORDS.has(s));
  },
});

predicate("iscell", v => isRuntimeCell(v));

predicate("isstruct", v => isRuntimeStruct(v) || isRuntimeStructArray(v));

predicate("issparse", v => isRuntimeSparseMatrix(v));

predicate(
  "isobject",
  v => isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)
);

builtin({
  name: "isprop",
  help: {
    signatures: ["tf = isprop(obj, PropertyName)"],
    description:
      "Return logical 1 where PropertyName is a property of object obj, else 0. The result has the same size as obj. Only class objects have properties: structs (even with that field), numeric, char and other built-in types always return false. Methods are not properties.",
  },
  nargin: [2, Infinity],
  apply: (args, _nargout, ctx) => {
    const v = args[0];
    const nameArg = args[1];
    // PropertyName must be a single char row vector or string scalar;
    // anything else (cell, string array, numeric, ...) means "no match".
    let propName: string | null = null;
    if (isRuntimeChar(nameArg)) propName = nameArg.value;
    else if (isRuntimeString(nameArg)) propName = nameArg;
    // Object arrays are homogeneous, so every element gives the same
    // answer. isprop reports every property, whatever its access: stored
    // ones (including dynamic properties) and Constant / Dependent ones.
    let answer = false;
    if (propName !== null) {
      if (isRuntimeClassInstance(v))
        answer =
          v.fields.has(propName) ||
          !!ctx.rt.classTable?.findProperty(v.className, propName);
      else if (isRuntimeClassInstanceArray(v))
        answer =
          v.elements.length > 0 &&
          (v.elements[0].fields.has(propName) ||
            !!ctx.rt.classTable?.findProperty(v.className, propName));
    }
    const shape = valueShape(v);
    const n = shape.reduce((a, b) => a * b, 1);
    if (n === 1) return RTV.logical(answer);
    const data = allocFloat64Array(n);
    if (answer) data.fill(1);
    return new RuntimeTensor(data, shape, undefined, true);
  },
});

builtin({
  name: "addprop",
  help: {
    signatures: ["p = addprop(obj, PropertyName)"],
    description:
      "Add a dynamic property named PropertyName to the dynamicprops (handle) object obj. The property can then be read and assigned via obj.PropertyName. Returns a meta.DynamicProperty describing the new property.",
  },
  nargin: 2,
  apply: args => {
    const obj = args[0];
    if (!isRuntimeClassInstance(obj))
      throw new RuntimeError(
        "addprop: first argument must be a dynamicprops object"
      );
    const name = toString(args[1]);
    // Add the property (default empty value) if not already present. For a
    // handle class this persists on the shared instance; reads/writes via
    // obj.(name) then resolve against the field map. The default value is
    // increfed so the field map holds a balanced reference (a later
    // bindField on assignment decrefs the old value).
    if (!obj.fields.has(name)) {
      const empty = RTV.tensor(allocFloat64Array(0), [0, 0]);
      incref(empty);
      obj.fields.set(name, empty);
    }
    // Minimal meta.DynamicProperty: enough for `p.Name` and for setting
    // accessor fields (e.g. p.SetMethod = @...) on the returned handle.
    return RTV.classInstance(
      "meta.DynamicProperty",
      ["Name"],
      true,
      new Map([["Name", RTV.char(name)]])
    );
  },
});

// ── Shape predicates ─────────────────────────────────────────────────────

// isscalar is numel(x)==1 for ANY type (cells, structs, strings, ...),
// not just numeric arrays.
predicate("isscalar", v => valueShape(v).reduce((a, b) => a * b, 1) === 1);

predicate("isempty", v => {
  if (typeof v === "number" || typeof v === "boolean") return false;
  if (isRuntimeComplexNumber(v)) return false;
  if (isRuntimeSparseMatrix(v)) return v.m === 0 || v.n === 0;
  if (isRuntimeTensor(v)) return v.data.length === 0;
  if (isRuntimeCell(v)) return v.data.length === 0;
  if (isRuntimeChar(v)) return v.value.length === 0;
  if (isRuntimeString(v)) return false;
  if (isRuntimeStructArray(v)) return v.elements.length === 0;
  if (isRuntimeClassInstanceArray(v)) return v.elements.length === 0;
  if (isRuntimeStringArray(v)) return v.data.length === 0;
  return false;
});

predicate("isvector", v => {
  const shape = valueShape(v);
  return shape.length === 2 && (shape[0] === 1 || shape[1] === 1);
});

predicate("isrow", v => {
  const shape = valueShape(v);
  return shape.length === 2 && shape[0] === 1;
});

predicate("iscolumn", v => {
  const shape = valueShape(v);
  return shape.length === 2 && shape[1] === 1;
});

predicate("ismatrix", v => valueShape(v).length <= 2);

// ── Shape queries ────────────────────────────────────────────────────────

builtin({
  name: "numel",
  nargin: [1, Infinity],
  apply: args => {
    if (args.length > 1)
      throw new RuntimeError(
        "numel: the numel(A, index1, ..., indexN) form is not supported"
      );
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return 1;
    if (isRuntimeComplexNumber(v)) return 1;
    if (isRuntimeTensor(v)) return v.data.length;
    if (isRuntimeSparseMatrix(v)) return v.m * v.n;
    if (isRuntimeCell(v)) return v.data.length;
    if (isRuntimeChar(v)) return v.value.length;
    if (isRuntimeString(v)) return 1;
    if (isRuntimeStructArray(v)) return v.elements.length;
    if (isRuntimeClassInstanceArray(v)) return v.elements.length;
    if (isRuntimeStringArray(v)) return v.data.length;
    return 1;
  },
});

builtin({
  name: "length",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return 1;
    if (isRuntimeComplexNumber(v)) return 1;
    if (isRuntimeTensor(v))
      return v.data.length === 0 ? 0 : Math.max(...v.shape);
    if (isRuntimeSparseMatrix(v)) return Math.max(v.m, v.n);
    if (isRuntimeCell(v)) return v.data.length === 0 ? 0 : Math.max(...v.shape);
    if (isRuntimeChar(v)) {
      const s =
        v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
      return s.length === 0 ? 0 : Math.max(...s);
    }
    if (isRuntimeString(v)) return 1;
    if (isRuntimeStructArray(v))
      return v.elements.length === 0 ? 0 : Math.max(...v.shape);
    if (isRuntimeClassInstanceArray(v))
      return v.elements.length === 0 ? 0 : Math.max(...v.shape);
    if (isRuntimeStringArray(v))
      return v.data.length === 0 ? 0 : Math.max(...v.shape);
    return 1;
  },
});

builtin({
  name: "ndims",
  nargin: 1,
  apply: args => {
    return Math.max(2, valueShape(args[0]).length);
  },
});

const sizeDimError = () =>
  builtinError(
    "MATLAB:size:invalidDim",
    "Dimension argument must be a positive integer scalar or a vector of positive integers. When an empty dimension argument is specified, it must be a 0-by-0, 1-by-0, or 0-by-1 array."
  );

/** The dimensions requested by size(A, dim), size(A, vecdim) or
 *  size(A, dim1, dim2, ...): positive integers. */
function sizeDims(args: RuntimeValue[]): number[] {
  const dims: number[] = [];
  const add = (x: number) => {
    if (!(Number.isInteger(x) && x >= 1)) throw sizeDimError();
    dims.push(x);
  };
  for (const a of args) {
    if (typeof a === "number") add(a);
    else if (typeof a === "boolean") add(a ? 1 : 0);
    else if (isRuntimeTensor(a) && !a.imag) {
      if (args.length > 1 && a.data.length !== 1) throw sizeDimError();
      if (a.data.length === 0 && a.shape.length > 2) throw sizeDimError();
      for (let i = 0; i < a.data.length; i++) add(a.data[i]);
    } else throw sizeDimError();
  }
  return dims;
}

builtin({
  name: "size",
  nargin: [1, Infinity],
  nargout: [0, Infinity],
  apply: (args, nargout) => {
    const shape = valueShape(args[0]);
    if (args.length === 1) {
      if (nargout <= 1)
        return makeTensor(allocFloat64Array(shape), undefined, [
          1,
          shape.length,
        ]);
      // [d1, ..., dn] = size(A): the last output is the product of the
      // remaining dimensions.
      const result: RuntimeValue[] = [];
      for (let i = 0; i < nargout - 1; i++)
        result.push(i < shape.length ? shape[i] : 1);
      let rest = 1;
      for (let k = nargout - 1; k < shape.length; k++) rest *= shape[k];
      result.push(rest);
      return result;
    }
    const sizes = sizeDims(args.slice(1)).map(d =>
      d <= shape.length ? shape[d - 1] : 1
    );
    if (nargout <= 1)
      return sizes.length === 1
        ? sizes[0]
        : makeTensor(allocFloat64Array(sizes), undefined, [1, sizes.length]);
    if (nargout !== sizes.length)
      throw builtinError(
        "MATLAB:size:NumOutNotEqualNumDims",
        "Incorrect number of output arguments. Number of output arguments must equal the number of input dimension arguments."
      );
    return sizes;
  },
});

// ── class() ──────────────────────────────────────────────────────────────

/** MATLAB's class name of a value. */
function classNameOf(v: RuntimeValue): string {
  if (typeof v === "number") return "double";
  if (typeof v === "boolean") return "logical";
  if (isRuntimeComplexNumber(v)) return "double";
  if (isRuntimeTensor(v)) return v._isLogical ? "logical" : "double";
  if (isRuntimeSparseMatrix(v)) return "double";
  if (isRuntimeString(v)) return "string";
  if (isRuntimeStringArray(v)) return "string";
  if (isRuntimeChar(v)) return "char";
  if (isRuntimeStruct(v) || isRuntimeStructArray(v)) return "struct";
  if (isRuntimeCell(v)) return "cell";
  if (isRuntimeDictionary(v)) return "dictionary";
  if (isRuntimeClassInstance(v)) return v.className;
  if (isRuntimeClassInstanceArray(v)) return v.className;
  if (isRuntimeFunction(v)) return "function_handle";
  if (isRuntimeDummyHandle(v)) return "dummy_handle";
  if (isRuntimeGraphicsHandle(v)) {
    const handleClass: Record<string, string> = {
      contour: "matlab.graphics.chart.primitive.Contour",
      patch: "matlab.graphics.primitive.Patch",
      quiver3: "matlab.graphics.chart.primitive.Quiver",
    };
    return handleClass[v._traceType] ?? "matlab.graphics.primitive.Surface";
  }
  return "unknown";
}

builtin({
  name: "class",
  nargin: [1, Infinity],
  apply: args => {
    if (args.length === 1) return RTV.char(classNameOf(args[0]));
    // Old-style (pre-classdef) constructor form: class(structData,
    // 'ClassName') builds a value-type instance whose fields are the
    // struct's fields. The inheritance form class(s, 'Name', parent, ...)
    // is not supported.
    if (args.length > 2)
      throw new RuntimeError(
        "class: the class(s, 'name', parent, ...) form is not supported"
      );
    const s = args[0];
    const nameVal = args[1];
    if (!isRuntimeStruct(s))
      throw new RuntimeError("class: first argument must be a scalar struct");
    if (!isRuntimeChar(nameVal) && !isRuntimeString(nameVal))
      throw new RuntimeError("class: class name must be a character vector");
    const className = isRuntimeChar(nameVal) ? nameVal.value : nameVal;
    const fieldNames = [...s.fields.keys()];
    return RTV.classInstance(className, fieldNames, false, s.fields);
  },
});

// ── superiorto / inferiorto ───────────────────────────────────────────────
// Old-style (pre-classdef) class-precedence declarations. Called inside a
// constructor to rank the class being defined relative to other classes for
// function dispatch. numbl already dispatches to class methods ahead of
// builtins when an argument is a class instance, so these are accepted as
// no-ops (they take one or more class-name strings and return nothing).

for (const name of ["superiorto", "inferiorto"]) {
  builtin({
    name,
    help: {
      signatures: [`${name}('Class1', 'Class2', ...)`],
      description:
        name === "superiorto"
          ? "Establish superior class relationship (old-style class precedence). Accepted as a no-op."
          : "Establish inferior class relationship (old-style class precedence). Accepted as a no-op.",
    },
    nargin: [1, Infinity],
    nargout: 0,
    apply: args => {
      for (const a of args)
        if (!isRuntimeChar(a) && !isRuntimeString(a))
          throw new RuntimeError(
            `${name}: arguments must be class names (character vectors or strings)`
          );
    },
  });
}

// ── fieldnames / fields ──────────────────────────────────────────────────

function fieldnamesApply(v: RuntimeValue, rt: Runtime): RuntimeValue {
  // N×1 cell of char vectors (cellstr), matching MATLAB.
  if (isRuntimeStructArray(v)) {
    const names = v.fieldNames;
    return RTV.cell(
      names.map(n => RTV.char(n)),
      [names.length, 1]
    );
  }
  // An object of a classdef class: its public, non-hidden properties.
  const table = rt.classTable;
  if (
    (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)) &&
    table?.get(v.className)?.ast
  ) {
    const names = table.publicProperties(v.className).map(p => p.name);
    // A subclass of MException also has MException's properties.
    if (table.ancestors(v.className).includes(MEXCEPTION_CLASS))
      for (const pn of MEXCEPTION_PROPERTIES)
        if (!names.includes(pn)) names.push(pn);
    return RTV.cell(
      names.map(n => RTV.char(n)),
      [names.length, 1]
    );
  }
  if (!isRuntimeStruct(v) && !isRuntimeClassInstance(v))
    throw builtinError(
      "MATLAB:fieldnames:InvalidInput",
      `Invalid input argument of type '${classNameOf(v)}'. Input must be a structure or a Java or COM object.`
    );
  const names = [...v.fields.keys()];
  return RTV.cell(
    names.map(n => RTV.char(n)),
    [names.length, 1]
  );
}

builtin({
  name: "fieldnames",
  // fieldnames(obj, '-full'): the option only adds detail for Java and COM
  // objects, so it is accepted and ignored.
  nargin: [1, 2],
  apply: (args, _nargout, ctx) => {
    if (
      args.length === 2 &&
      !(
        (isRuntimeChar(args[1]) || isRuntimeString(args[1])) &&
        toString(args[1]) === "-full"
      )
    )
      throw new RuntimeError("fieldnames: the second argument must be '-full'");
    return fieldnamesApply(args[0], ctx.rt);
  },
});
builtin({
  name: "fields",
  nargin: 1,
  apply: (args, _nargout, ctx) => fieldnamesApply(args[0], ctx.rt),
});

// ── fileparts ────────────────────────────────────────────────────────────

builtin({
  name: "fileparts",
  nargin: 1,
  nargout: 3,
  mfile: true,
  apply: (args, nargout) => {
    const v = args[0];
    if (!isRuntimeString(v) && !isRuntimeChar(v))
      throw builtinError(
        "MATLAB:fileparts:MustBeChar",
        "Input must be a row vector of characters, or a string scalar, or a cellstr, or a string matrix."
      );
    const p = isRuntimeString(v) ? v : v.value;
    const lastSep = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    const dir = lastSep >= 0 ? p.slice(0, lastSep) : "";
    const rest = lastSep >= 0 ? p.slice(lastSep + 1) : p;
    const dotIdx = rest.lastIndexOf(".");
    const name = dotIdx >= 0 ? rest.slice(0, dotIdx) : rest;
    const ext = dotIdx >= 0 ? rest.slice(dotIdx) : "";
    // A string input gives string outputs, as in MATLAB.
    const out = (t: string): RuntimeValue =>
      isRuntimeString(v) ? RTV.string(t) : RTV.char(t);
    const parts = [out(dir), out(name), out(ext)];
    return nargout <= 1 ? parts[0] : parts.slice(0, nargout);
  },
});

// ── fullfile ─────────────────────────────────────────────────────────────

/** Join path parts the way MATLAB's fullfile does on Unix: empty parts are
 *  dropped, and repeated separators and `/./` collapse, except for the
 *  leading separators. */
function joinPath(parts: string[]): string {
  const nonEmpty = parts.filter(p => p.length > 0);
  if (nonEmpty.length === 0) return "";
  const joined = nonEmpty.reduce(
    (acc, p) => acc.replace(/\/+$/, "") + "/" + p.replace(/^\/+/, "")
  );
  const lead = nonEmpty[0].match(/^\/*/)![0];
  let rest = joined.slice(lead.length).replace(/\/{2,}/g, "/");
  while (rest.includes("/./")) rest = rest.replace("/./", "/");
  return lead + rest;
}

builtin({
  name: "fullfile",
  mfile: true,
  apply: args => {
    if (args.length === 0)
      throw builtinError(
        "MATLAB:narginchk:notEnoughInputs",
        "Not enough input arguments."
      );
    const text = (a: RuntimeValue): string => {
      if (isRuntimeString(a)) return a;
      if (isRuntimeChar(a)) return a.value;
      throw new RuntimeError(
        "fullfile: arguments must be character vectors, strings or cell arrays of them"
      );
    };
    const cells = args.filter(isRuntimeCell);
    if (cells.length === 0) {
      const out = joinPath(args.map(text));
      return args.some(a => isRuntimeString(a)) ? out : RTV.char(out);
    }
    const n = cells[0].data.length;
    if (cells.some(c => c.data.length !== n))
      throw new RuntimeError(
        "fullfile: cell array arguments must have the same number of elements"
      );
    const out: RuntimeValue[] = [];
    for (let i = 0; i < n; i++) {
      const parts = args.map(a =>
        isRuntimeCell(a) ? text(a.data[i]) : text(a)
      );
      out.push(RTV.char(joinPath(parts)));
    }
    return RTV.cell(out, [...cells[0].shape]);
  },
});

// ── isa, superclasses, properties, ismethod ──────────────────────────────

builtin({
  name: "isa",
  nargin: 2,
  apply: (args, _nargout, ctx) => ctx.rt.isa(args[0], args[1]),
});

/** The class an object or a class name (char/string) refers to. */
function referencedClass(v: RuntimeValue): string | undefined {
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v))
    return v.className;
  if (isRuntimeChar(v) || isRuntimeString(v)) return toString(v);
  return undefined;
}

/** A column cell of string names. */
const nameColumn = (names: string[]) =>
  RTV.cell(
    names.map(n => RTV.string(n)),
    [names.length, 1]
  );

// superclasses(obj) / superclasses('ClassName'): the names of the class's
// ancestors (not the class itself); 0x1 for a class without superclasses
// or a built-in type.
builtin({
  name: "superclasses",
  nargin: 1,
  apply: (args, _nargout, ctx) => {
    const className = referencedClass(args[0]);
    return nameColumn(className ? ctx.rt.classAncestors(className) : []);
  },
});

// properties(obj) / properties('ClassName'): the public properties; a 0x1
// cell for anything that is not a class.
builtin({
  name: "properties",
  nargin: 1,
  scope: true,
  apply: (args, nargout, { scope, rt }) => {
    const className = referencedClass(args[0]);
    const names = className
      ? scope.classPublicProperties(className)
      : undefined;
    if (!names) return RTV.cell([], [0, 1]);
    if (nargout === 0 && isRuntimeClassInstance(args[0])) {
      // properties(obj) without an output lists the names.
      rt.output(
        `\nProperties for class ${className}:\n\n` +
          names.map(n => `    ${n}\n`).join("") +
          "\n"
      );
      return undefined;
    }
    return nameColumn(names);
  },
});

// methods(obj) / methods('ClassName'): the public, non-hidden methods,
// sorted; listed in columns without an output.
builtin({
  name: "methods",
  nargin: 1,
  scope: true,
  apply: (args, nargout, { scope, rt }) => {
    const className = referencedClass(args[0]);
    const found = className ? scope.classPublicMethods(className) : undefined;
    const names = [...(found ?? [])].sort();
    if (nargout > 0)
      return RTV.cell(
        names.map(n => RTV.char(n)),
        [names.length, 1]
      );
    if (!found) return undefined;
    const width = Math.max(...names.map(n => n.length)) + 2;
    const perLine = Math.max(1, Math.floor(80 / width));
    const lines: string[] = [];
    for (let i = 0; i < names.length; i += perLine)
      lines.push(
        names
          .slice(i, i + perLine)
          .map(n => n.padEnd(width))
          .join("")
      );
    rt.output(`\nMethods for class ${className}:\n\n${lines.join("\n")}\n\n`);
    return undefined;
  },
});

// metaclass(obj): the meta.class object of the value's class (as ?Name).
builtin({
  name: "metaclass",
  nargin: 1,
  scope: true,
  apply: (args, _nargout, { scope }) => scope.metaClass(classNameOf(args[0])),
});

// ismethod(obj, name): whether `name` is a public method of the object's
// class (a char/string first argument names the class). Built-in types are
// not modelled and give false.
builtin({
  name: "ismethod",
  nargin: 2,
  mfile: true,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    const className = referencedClass(args[0]);
    const nameArg = args[1];
    let methodName: string | undefined;
    if (isRuntimeChar(nameArg) || isRuntimeString(nameArg))
      methodName = toString(nameArg);
    else if (isRuntimeCell(nameArg) && nameArg.data.length === 1) {
      const el = nameArg.data[0];
      if (isRuntimeChar(el) || isRuntimeString(el)) methodName = toString(el);
    }
    if (!className || methodName === undefined) return RTV.logical(false);
    const names = scope.classPublicMethods(className);
    return RTV.logical(names !== undefined && names.has(methodName));
  },
});

// Internal: the runtime representation of a value, for tests.
builtin({
  name: "__inferred_type_str",
  nargin: 1,
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return RTV.string("Number");
    if (typeof v === "boolean") return RTV.string("Boolean");
    if (isRuntimeTensor(v)) return RTV.string("Tensor");
    if (isRuntimeCell(v)) return RTV.string("Cell");
    if (isRuntimeClassInstance(v))
      return RTV.string(`ClassInstance(${v.className})`);
    if (isRuntimeChar(v)) return RTV.string("Char");
    if (isRuntimeString(v)) return RTV.string("String");
    if (isRuntimeFunction(v)) return RTV.string("Function");
    return RTV.string("Unknown");
  },
});

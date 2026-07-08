/**
 * Introspection builtins: size, length, numel, ndims, isempty, isscalar,
 * isvector, isrow, iscolumn, ismatrix, islogical, isnumeric, isfloat,
 * isinteger, ischar, isstring, iscell, isstruct, issparse, class.
 */

import {
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  type RuntimeChar,
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
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toString } from "../../runtime/convert.js";
import type { JitType } from "../../jitTypes.js";
import { defineBuiltin, type BuiltinCase, makeTensor } from "./types.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { incref } from "../../runtime/refcount.js";

// ── Helpers ──────────────────────────────────────────────────────────────

/** Case that accepts any known type and returns logical. */
function anyToLogicalCase(
  applyFn: (args: RuntimeValue[]) => RuntimeValue | RuntimeValue[]
): BuiltinCase {
  return {
    match: argTypes => {
      if (argTypes.length !== 1) return null;
      return [{ kind: "boolean" }];
    },
    apply: applyFn,
  };
}

/** Get the effective 2D+ shape of a runtime value. */
function getShape(v: RuntimeValue): number[] {
  if (typeof v === "number" || typeof v === "boolean") return [1, 1];
  if (isRuntimeComplexNumber(v)) return [1, 1];
  if (isRuntimeTensor(v))
    return v.shape.length >= 2 ? v.shape : [1, ...v.shape];
  if (isRuntimeChar(v))
    return v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
  if (isRuntimeString(v)) return [1, 1];
  if (isRuntimeSparseMatrix(v)) return [v.m, v.n];
  if (isRuntimeCell(v)) return v.shape;
  if (isRuntimeStructArray(v)) return [1, v.elements.length];
  if (isRuntimeClassInstanceArray(v)) return [...v.shape];
  if (isRuntimeStringArray(v)) return [...v.shape];
  return [1, 1];
}

// ── Type predicates ──────────────────────────────────────────────────────

defineBuiltin({
  name: "isnumeric",
  cases: [
    anyToLogicalCase(args => {
      const v = args[0];
      if (typeof v === "number") return true;
      if (isRuntimeTensor(v) && !v._isLogical) return true;
      if (isRuntimeComplexNumber(v)) return true;
      if (isRuntimeSparseMatrix(v)) return true;
      if (isRuntimeClassInstance(v) && v._builtinData !== undefined)
        return true;
      return false;
    }),
  ],
});

defineBuiltin({
  name: "isfloat",
  cases: [
    anyToLogicalCase(args => {
      const v = args[0];
      if (typeof v === "number") return true;
      if (isRuntimeTensor(v) && !v._isLogical) return true;
      if (isRuntimeComplexNumber(v)) return true;
      if (isRuntimeSparseMatrix(v)) return true;
      return false;
    }),
  ],
});

defineBuiltin({
  name: "isinteger",
  cases: [anyToLogicalCase(() => false)],
});

defineBuiltin({
  name: "islogical",
  cases: [
    anyToLogicalCase(args => {
      const v = args[0];
      if (typeof v === "boolean") return true;
      if (isRuntimeTensor(v) && v._isLogical === true) return true;
      return false;
    }),
  ],
});

defineBuiltin({
  name: "ischar",
  cases: [anyToLogicalCase(args => isRuntimeChar(args[0]))],
});

// Legacy alias for ischar (removed from modern MATLAB docs but still works).
defineBuiltin({
  name: "isstr",
  cases: [anyToLogicalCase(args => isRuntimeChar(args[0]))],
});

defineBuiltin({
  name: "isstring",
  cases: [
    anyToLogicalCase(
      args => isRuntimeString(args[0]) || isRuntimeStringArray(args[0])
    ),
  ],
});

// MATLAB's reserved words (`iskeyword` with no args). Note true/false are
// functions, not keywords; spmd is included.
const MATLAB_KEYWORDS: ReadonlySet<string> = new Set([
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

defineBuiltin({
  name: "isvarname",
  cases: [
    anyToLogicalCase(args => {
      const s = singleRowText(args[0]);
      if (s === null || s.length === 0 || s.length > 63) return false;
      if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(s)) return false;
      return !MATLAB_KEYWORDS.has(s);
    }),
  ],
});

defineBuiltin({
  name: "iskeyword",
  cases: [
    {
      // iskeyword() with no args returns the list of keywords (column cell).
      match: argTypes => {
        if (argTypes.length === 0) return [{ kind: "cell" }];
        if (argTypes.length === 1) return [{ kind: "boolean" }];
        return null;
      },
      apply: args => {
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
    },
  ],
});

defineBuiltin({
  name: "iscell",
  cases: [anyToLogicalCase(args => isRuntimeCell(args[0]))],
});

defineBuiltin({
  name: "isstruct",
  cases: [
    anyToLogicalCase(
      args => isRuntimeStruct(args[0]) || isRuntimeStructArray(args[0])
    ),
  ],
});

defineBuiltin({
  name: "issparse",
  cases: [anyToLogicalCase(args => isRuntimeSparseMatrix(args[0]))],
});

defineBuiltin({
  name: "isobject",
  cases: [
    anyToLogicalCase(
      args =>
        isRuntimeClassInstance(args[0]) || isRuntimeClassInstanceArray(args[0])
    ),
  ],
});

defineBuiltin({
  name: "isprop",
  help: {
    signatures: ["tf = isprop(obj, PropertyName)"],
    description:
      "Return logical 1 where PropertyName is a property of object obj, else 0. The result has the same size as obj. Only class objects have properties: structs (even with that field), numeric, char and other built-in types always return false. Methods are not properties.",
  },
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "boolean" }];
      },
      apply: args => {
        const v = args[0];
        const nameArg = args[1];
        // PropertyName must be a single char row vector or string scalar;
        // anything else (cell, string array, numeric, ...) means "no match".
        let propName: string | null = null;
        if (isRuntimeChar(nameArg)) propName = nameArg.value;
        else if (isRuntimeString(nameArg)) propName = nameArg;
        // Object arrays are homogeneous, so every element gives the same
        // answer. A class instance's field map holds all declared properties
        // (public and private), which is exactly what isprop reports.
        let answer = false;
        if (propName !== null) {
          if (isRuntimeClassInstance(v)) answer = v.fields.has(propName);
          else if (isRuntimeClassInstanceArray(v))
            answer =
              v.elements.length > 0 && v.elements[0].fields.has(propName);
        }
        const shape = getShape(v);
        const n = shape.reduce((a, b) => a * b, 1);
        if (n === 1) return RTV.logical(answer);
        const data = allocFloat64Array(n);
        if (answer) data.fill(1);
        return new RuntimeTensor(data, shape, undefined, true);
      },
    },
  ],
});

defineBuiltin({
  name: "addprop",
  help: {
    signatures: ["p = addprop(obj, PropertyName)"],
    description:
      "Add a dynamic property named PropertyName to the dynamicprops (handle) object obj. The property can then be read and assigned via obj.PropertyName. Returns a meta.DynamicProperty describing the new property.",
  },
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "unknown" }];
      },
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
    },
  ],
});

// ── Shape predicates ─────────────────────────────────────────────────────

defineBuiltin({
  name: "isscalar",
  cases: [
    // isscalar is numel(x)==1 for ANY type (cells, structs, strings, ...),
    // not just numeric arrays.
    anyToLogicalCase(
      args => getShape(args[0]).reduce((a, b) => a * b, 1) === 1
    ),
  ],
});

defineBuiltin({
  name: "isempty",
  cases: [
    anyToLogicalCase(args => {
      const v = args[0];
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
    }),
  ],
});

defineBuiltin({
  name: "isvector",
  cases: [
    anyToLogicalCase(args => {
      const shape = getShape(args[0]);
      return shape.filter(d => d > 1).length <= 1;
    }),
  ],
});

defineBuiltin({
  name: "isrow",
  cases: [
    anyToLogicalCase(args => {
      const shape = getShape(args[0]);
      return shape.length === 2 && shape[0] === 1;
    }),
  ],
});

defineBuiltin({
  name: "iscolumn",
  cases: [
    anyToLogicalCase(args => {
      const shape = getShape(args[0]);
      return shape.length === 2 && shape[1] === 1;
    }),
  ],
});

defineBuiltin({
  name: "ismatrix",
  cases: [anyToLogicalCase(args => getShape(args[0]).length <= 2)],
});

// ── Shape queries ────────────────────────────────────────────────────────

defineBuiltin({
  name: "numel",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "number", sign: "nonneg" as const }];
      },
      apply: args => {
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
    },
  ],
});

defineBuiltin({
  name: "length",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "number", sign: "nonneg" as const }];
      },
      apply: args => {
        const v = args[0];
        if (typeof v === "number" || typeof v === "boolean") return 1;
        if (isRuntimeComplexNumber(v)) return 1;
        if (isRuntimeTensor(v))
          return v.data.length === 0 ? 0 : Math.max(...v.shape);
        if (isRuntimeSparseMatrix(v)) return Math.max(v.m, v.n);
        if (isRuntimeCell(v))
          return v.data.length === 0 ? 0 : Math.max(...v.shape);
        if (isRuntimeChar(v)) {
          const s =
            v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
          return s.length === 0 ? 0 : Math.max(...s);
        }
        if (isRuntimeString(v)) return 1;
        if (isRuntimeStructArray(v)) return v.elements.length;
        if (isRuntimeClassInstanceArray(v))
          return v.elements.length === 0 ? 0 : Math.max(...v.shape);
        if (isRuntimeStringArray(v))
          return v.data.length === 0 ? 0 : Math.max(...v.shape);
        return 1;
      },
    },
  ],
});

defineBuiltin({
  name: "ndims",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "number", sign: "nonneg" as const }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeTensor(v)) return Math.max(2, v.shape.length);
        return 2;
      },
    },
  ],
});

defineBuiltin({
  name: "size",
  cases: [
    // size(A, dim)
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const dimKind = argTypes[1].kind;
        if (dimKind !== "number" && dimKind !== "boolean") return null;
        return [{ kind: "number", sign: "nonneg" as const }];
      },
      apply: args => {
        const v = args[0];
        const shape = getShape(v);
        const dim =
          typeof args[1] === "number" ? Math.round(args[1]) : args[1] ? 1 : 0;
        return dim > 0 && dim <= shape.length ? shape[dim - 1] : 1;
      },
    },
    // size(A) — single-output returns a 1×ndims tensor of sizes.
    // [d1,d2,...] = size(A) — multi-output returns each dim as a scalar;
    // the trailing dim aggregates sizes of any remaining dims (MATLAB rule),
    // but for the type-level signature we just report one nonneg number per
    // requested output. The apply path pads with 1s for excess outputs.
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1) return null;
        if (nargout > 1) {
          const out: JitType[] = [];
          for (let i = 0; i < nargout; i++) {
            out.push({ kind: "number", sign: "nonneg" });
          }
          return out;
        }
        const a = argTypes[0];
        const ndims = a.kind === "tensor" && a.shape ? a.shape.length : 2;
        return [
          {
            kind: "tensor" as const,
            isComplex: false,
            shape: [1, ndims],
            nonneg: true,
          },
        ];
      },
      apply: (args, nargout) => {
        const v = args[0];
        const shape = getShape(v);
        if (nargout > 1) {
          const result: RuntimeValue[] = [];
          for (let i = 0; i < nargout; i++) {
            result.push(i < shape.length ? shape[i] : 1);
          }
          return result;
        }
        return makeTensor(allocFloat64Array(shape), undefined, [
          1,
          shape.length,
        ]);
      },
    },
  ],
});

// ── class() ──────────────────────────────────────────────────────────────

/** Helper to create a RuntimeChar value. */
function mkChar(value: string): RuntimeChar {
  return RTV.char(value);
}

defineBuiltin({
  name: "class",
  cases: [
    // Old-style (pre-classdef) constructor form: class(structData, 'ClassName')
    // builds a value-type instance whose fields are the struct's fields. The
    // optional class(s,'Name',parent,...) inheritance form is not supported
    // (returns null → "unsupported argument types").
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        if (argTypes[0].kind !== "struct") return null;
        const k = argTypes[1].kind;
        if (k !== "char" && k !== "string") return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const s = args[0];
        if (!isRuntimeStruct(s))
          throw new RuntimeError(
            "class: first argument must be a scalar struct"
          );
        const nameVal = args[1];
        const className = isRuntimeChar(nameVal)
          ? nameVal.value
          : isRuntimeString(nameVal)
            ? nameVal
            : String(nameVal);
        const fieldNames = [...s.fields.keys()];
        return RTV.classInstance(className, fieldNames, false, s.fields);
      },
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "char" }];
      },
      apply: args => {
        const v = args[0];
        if (typeof v === "number") return mkChar("double");
        if (typeof v === "boolean") return mkChar("logical");
        if (isRuntimeComplexNumber(v)) return mkChar("double");
        if (isRuntimeTensor(v))
          return mkChar(v._isLogical ? "logical" : "double");
        if (isRuntimeSparseMatrix(v)) return mkChar("double");
        if (isRuntimeString(v)) return mkChar("string");
        if (isRuntimeStringArray(v)) return mkChar("string");
        if (isRuntimeChar(v)) return mkChar("char");
        if (isRuntimeStruct(v) || isRuntimeStructArray(v))
          return mkChar("struct");
        if (isRuntimeCell(v)) return mkChar("cell");
        if (isRuntimeDictionary(v)) return mkChar("dictionary");
        if (isRuntimeClassInstance(v)) return mkChar(v.className);
        if (isRuntimeClassInstanceArray(v)) return mkChar(v.className);
        if (isRuntimeFunction(v)) return mkChar("function_handle");
        if (isRuntimeDummyHandle(v)) return mkChar("dummy_handle");
        if (isRuntimeGraphicsHandle(v)) {
          const handleClass: Record<string, string> = {
            contour: "matlab.graphics.chart.primitive.Contour",
            quiver3: "matlab.graphics.chart.primitive.Quiver",
          };
          return mkChar(
            handleClass[v._traceType] ?? "matlab.graphics.primitive.Surface"
          );
        }
        return mkChar("unknown");
      },
    },
  ],
});

// ── superiorto / inferiorto ───────────────────────────────────────────────
// Old-style (pre-classdef) class-precedence declarations. Called inside a
// constructor to rank the class being defined relative to other classes for
// function dispatch. numbl already dispatches to class methods ahead of
// builtins when an argument is a class instance, so these are accepted as
// no-ops (they take one or more class-name strings and return nothing).

for (const name of ["superiorto", "inferiorto"]) {
  defineBuiltin({
    name,
    help: {
      signatures: [`${name}('Class1', 'Class2', ...)`],
      description:
        name === "superiorto"
          ? "Establish superior class relationship (old-style class precedence). Accepted as a no-op."
          : "Establish inferior class relationship (old-style class precedence). Accepted as a no-op.",
    },
    cases: [
      {
        match: argTypes => {
          for (const t of argTypes) {
            if (t.kind !== "char" && t.kind !== "string") return null;
          }
          return [];
        },
        apply: () => undefined as unknown as RuntimeValue,
      },
    ],
  });
}

// ── fieldnames / fields ──────────────────────────────────────────────────

function fieldnamesApply(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 1)
    throw new RuntimeError("fieldnames requires 1 argument");
  const v = args[0];
  if (isRuntimeStructArray(v)) {
    const names = v.fieldNames;
    return RTV.cell(
      names.map(n => RTV.string(n)),
      [names.length, 1]
    );
  }
  if (!isRuntimeStruct(v) && !isRuntimeClassInstance(v))
    throw new RuntimeError("fieldnames: argument must be a struct");
  const names = [...v.fields.keys()];
  return RTV.cell(
    names.map(n => RTV.string(n)),
    [names.length, 1]
  );
}

const fieldnamesCase: BuiltinCase = {
  match: argTypes => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    if (k !== "struct" && k !== "class_instance" && k !== "unknown")
      return null;
    return [{ kind: "unknown" }];
  },
  apply: args => fieldnamesApply(args),
};

defineBuiltin({ name: "fieldnames", cases: [fieldnamesCase] });
defineBuiltin({ name: "fields", cases: [fieldnamesCase] });

// ── fileparts ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "fileparts",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1) return null;
        const k = argTypes[0].kind;
        if (k !== "string" && k !== "char" && k !== "unknown") return null;
        const charType: JitType = { kind: "char" };
        return Array(Math.max(nargout, 1)).fill(charType);
      },
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("fileparts requires 1 argument");
        const v = args[0];
        if (!isRuntimeString(v) && !isRuntimeChar(v))
          throw new RuntimeError("fileparts: argument must be a string");
        const p = isRuntimeString(v) ? v : v.value;
        const lastSep = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
        const dir = lastSep >= 0 ? p.slice(0, lastSep) : "";
        const rest = lastSep >= 0 ? p.slice(lastSep + 1) : p;
        const dotIdx = rest.lastIndexOf(".");
        const name = dotIdx >= 0 ? rest.slice(0, dotIdx) : rest;
        const ext = dotIdx >= 0 ? rest.slice(dotIdx) : "";
        if (nargout <= 1) return RTV.char(dir);
        if (nargout === 2) return [RTV.char(dir), RTV.char(name)];
        return [RTV.char(dir), RTV.char(name), RTV.char(ext)];
      },
    },
  ],
});

// ── fullfile ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "fullfile",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1) return null;
        for (const t of argTypes) {
          if (t.kind !== "string" && t.kind !== "char" && t.kind !== "unknown")
            return null;
        }
        return [{ kind: "char" } as JitType];
      },
      apply: args => {
        const parts: string[] = [];
        for (const a of args) {
          if (isRuntimeString(a)) parts.push(a);
          else if (isRuntimeChar(a)) parts.push(a.value);
          else throw new RuntimeError("fullfile: arguments must be strings");
        }
        return RTV.char(parts.join("/"));
      },
    },
  ],
});

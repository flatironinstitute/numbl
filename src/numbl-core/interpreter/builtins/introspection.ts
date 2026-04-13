/**
 * Introspection builtins: size, length, numel, ndims, isempty, isscalar,
 * isvector, isrow, iscolumn, ismatrix, islogical, isnumeric, isfloat,
 * isinteger, ischar, isstring, iscell, isstruct, issparse, class.
 */

import {
  FloatXArray,
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
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import type { JitType } from "../jit/jitTypes.js";
import { defineBuiltin, type BuiltinCase, makeTensor } from "./types.js";

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
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "complex_or_number") return "1";
    if (k === "tensor")
      return (types[0] as Extract<JitType, { kind: "tensor" }>).isLogical
        ? "0"
        : "1";
    if (k === "boolean" || k === "string" || k === "char" || k === "struct")
      return "0";
    return null;
  },
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
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "complex_or_number") return "1";
    if (k === "tensor")
      return (types[0] as Extract<JitType, { kind: "tensor" }>).isLogical
        ? "0"
        : "1";
    if (k === "boolean" || k === "string" || k === "char" || k === "struct")
      return "0";
    return null;
  },
});

defineBuiltin({
  name: "isinteger",
  cases: [anyToLogicalCase(() => false)],
  jitEmit: () => "0",
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
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "boolean") return "1";
    if (k === "tensor")
      return (types[0] as Extract<JitType, { kind: "tensor" }>).isLogical
        ? "1"
        : "0";
    if (
      k === "number" ||
      k === "complex_or_number" ||
      k === "string" ||
      k === "char" ||
      k === "struct" ||
      k === "class_instance"
    )
      return "0";
    return null;
  },
});

defineBuiltin({
  name: "ischar",
  cases: [anyToLogicalCase(args => isRuntimeChar(args[0]))],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "char") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

defineBuiltin({
  name: "isstring",
  cases: [anyToLogicalCase(args => isRuntimeString(args[0]))],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "string") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

defineBuiltin({
  name: "iscell",
  cases: [anyToLogicalCase(args => isRuntimeCell(args[0]))],
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

defineBuiltin({
  name: "isstruct",
  cases: [
    anyToLogicalCase(
      args => isRuntimeStruct(args[0]) || isRuntimeStructArray(args[0])
    ),
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "struct") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

defineBuiltin({
  name: "issparse",
  cases: [anyToLogicalCase(args => isRuntimeSparseMatrix(args[0]))],
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

// ── Shape predicates ─────────────────────────────────────────────────────

defineBuiltin({
  name: "isscalar",
  cases: [
    anyToLogicalCase(args => {
      const v = args[0];
      if (typeof v === "number" || typeof v === "boolean") return true;
      if (isRuntimeComplexNumber(v)) return true;
      if (isRuntimeTensor(v)) return v.data.length === 1;
      return false;
    }),
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean" || k === "complex_or_number")
      return "1";
    if (k === "string") return "1";
    return null;
  },
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
      return false;
    }),
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (
      k === "number" ||
      k === "boolean" ||
      k === "complex_or_number" ||
      k === "string"
    )
      return "0";
    return null;
  },
});

defineBuiltin({
  name: "isvector",
  cases: [
    anyToLogicalCase(args => {
      const shape = getShape(args[0]);
      return shape.filter(d => d > 1).length <= 1;
    }),
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean" || k === "complex_or_number")
      return "1";
    return null;
  },
});

defineBuiltin({
  name: "isrow",
  cases: [
    anyToLogicalCase(args => {
      const shape = getShape(args[0]);
      return shape.length === 2 && shape[0] === 1;
    }),
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean" || k === "complex_or_number")
      return "1";
    return null;
  },
});

defineBuiltin({
  name: "iscolumn",
  cases: [
    anyToLogicalCase(args => {
      const shape = getShape(args[0]);
      return shape.length === 2 && shape[1] === 1;
    }),
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "boolean" || k === "complex_or_number")
      return "1";
    return null;
  },
});

defineBuiltin({
  name: "ismatrix",
  cases: [anyToLogicalCase(args => getShape(args[0]).length <= 2)],
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "1";
    return null;
  },
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
        return 1;
      },
    },
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (
      k === "number" ||
      k === "boolean" ||
      k === "complex_or_number" ||
      k === "string"
    )
      return "1";
    return null;
  },
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
        return 1;
      },
    },
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (
      k === "number" ||
      k === "boolean" ||
      k === "complex_or_number" ||
      k === "string"
    )
      return "1";
    return null;
  },
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
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (
      k === "number" ||
      k === "boolean" ||
      k === "complex_or_number" ||
      k === "string" ||
      k === "char"
    )
      return "2";
    return null;
  },
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
    // size(A)
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
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
        return makeTensor(new FloatXArray(shape), undefined, [1, shape.length]);
      },
    },
  ],
  jitEmit: (args, types) => {
    if (args.length === 2) {
      const k = types[0]?.kind;
      if (
        k === "number" ||
        k === "boolean" ||
        k === "complex_or_number" ||
        k === "string"
      )
        return "1";
    }
    return null;
  },
});

// ── class() ──────────────────────────────────────────────────────────────

/** Helper to create a RuntimeChar value. */
function mkChar(value: string): RuntimeChar {
  return { kind: "char", value };
}

defineBuiltin({
  name: "class",
  cases: [
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
        if (isRuntimeChar(v)) return mkChar("char");
        if (isRuntimeStruct(v) || isRuntimeStructArray(v))
          return mkChar("struct");
        if (isRuntimeCell(v)) return mkChar("cell");
        if (isRuntimeDictionary(v)) return mkChar("dictionary");
        if (isRuntimeClassInstance(v)) return mkChar(v.className);
        if (isRuntimeFunction(v)) return mkChar("function_handle");
        if (isRuntimeDummyHandle(v)) return mkChar("dummy_handle");
        if (isRuntimeGraphicsHandle(v))
          return mkChar("matlab.graphics.primitive.Surface");
        return mkChar("unknown");
      },
    },
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    const wrap = (s: string) => `{kind:"char",value:"${s}"}`;
    switch (k) {
      case "number":
      case "complex_or_number":
        return wrap("double");
      case "boolean":
        return wrap("logical");
      case "tensor":
        return (types[0] as Extract<JitType, { kind: "tensor" }>).isLogical
          ? wrap("logical")
          : wrap("double");
      case "string":
        return wrap("string");
      case "char":
        return wrap("char");
      case "struct":
        return wrap("struct");
      case "class_instance":
        return wrap(
          (types[0] as Extract<JitType, { kind: "class_instance" }>).className
        );
      default:
        return null;
    }
  },
});

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

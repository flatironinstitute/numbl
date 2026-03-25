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
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeClassInstance,
  isRuntimeFunction,
  isRuntimeDummyHandle,
} from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
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
      if (argTypes[0].kind === "unknown") return null;
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
        if (argTypes[0].kind === "unknown") return null;
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
        if (argTypes[0].kind === "unknown") return null;
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
        if (argTypes[0].kind === "unknown") return null;
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
        if (argTypes[0].kind === "unknown") return null;
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
        if (argTypes[0].kind === "unknown") return null;
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

defineBuiltin({
  name: "class",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "complex_or_number")
          return [{ kind: "string", value: "double" }];
        return null;
      },
      apply: () => "double",
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "boolean") return null;
        return [{ kind: "string", value: "logical" }];
      },
      apply: () => "logical",
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (a.kind !== "tensor") return null;
        return [{ kind: "string", value: a.isLogical ? "logical" : "double" }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeTensor(v)) return v._isLogical ? "logical" : "double";
        throw new Error("class: unexpected tensor state");
      },
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "string") return null;
        return [{ kind: "string", value: "string" }];
      },
      apply: () => "string",
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "char") return null;
        return [{ kind: "string", value: "char" }];
      },
      apply: () => "char",
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "struct") return null;
        return [{ kind: "string", value: "struct" }];
      },
      apply: () => "struct",
    },
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        const a = argTypes[0];
        if (a.kind !== "class_instance") return null;
        return [{ kind: "string", value: a.className }];
      },
      apply: args => {
        const v = args[0];
        if (isRuntimeClassInstance(v)) return v.className;
        throw new Error("class: unexpected class_instance state");
      },
    },
    // Fallback for runtime types not representable in JitType (cell, function_handle, etc.)
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        // Accept unknown — fall through to runtime dispatch
        return null;
      },
      apply: args => {
        const v = args[0];
        if (typeof v === "number") return "double";
        if (typeof v === "boolean") return "logical";
        if (isRuntimeComplexNumber(v)) return "double";
        if (isRuntimeSparseMatrix(v)) return "double";
        if (isRuntimeCell(v)) return "cell";
        if (isRuntimeStruct(v)) return "struct";
        if (isRuntimeFunction(v)) return "function_handle";
        if (isRuntimeDummyHandle(v)) return "dummy_handle";
        if (isRuntimeStructArray(v)) return "struct";
        return "unknown";
      },
    },
  ],
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    switch (k) {
      case "number":
      case "complex_or_number":
        return '"double"';
      case "boolean":
        return '"logical"';
      case "tensor":
        return (types[0] as Extract<JitType, { kind: "tensor" }>).isLogical
          ? '"logical"'
          : '"double"';
      case "string":
        return '"string"';
      case "char":
        return '"char"';
      case "struct":
        return '"struct"';
      case "class_instance":
        return `"${(types[0] as Extract<JitType, { kind: "class_instance" }>).className}"`;
      default:
        return null;
    }
  },
});

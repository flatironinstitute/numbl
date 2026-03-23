/* eslint-disable @typescript-eslint/no-unused-vars */
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
import { registerIBuiltin, makeTensor } from "./types.js";

// ── Helpers ──────────────────────────────────────────────────────────────

/** Type rule for predicates that accept any type and return logical. */
function anyToLogical(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 1) return null;
  if (argTypes[0].kind === "unknown") return null;
  return [{ kind: "logical" }];
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

registerIBuiltin({
  name: "isnumeric",
  typeRule: anyToLogical,
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return true;
    if (isRuntimeTensor(v) && !v._isLogical) return true;
    if (isRuntimeComplexNumber(v)) return true;
    if (isRuntimeSparseMatrix(v)) return true;
    if (isRuntimeClassInstance(v) && v._builtinData !== undefined) return true;
    return false;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "complex") return "1";
    if (k === "realTensor")
      return (types[0] as { isLogical?: boolean }).isLogical ? "0" : "1";
    if (k === "complexTensor") return "1";
    if (k === "logical" || k === "string" || k === "char") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "isfloat",
  typeRule: anyToLogical,
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return true;
    if (isRuntimeTensor(v) && !v._isLogical) return true;
    if (isRuntimeComplexNumber(v)) return true;
    if (isRuntimeSparseMatrix(v)) return true;
    return false;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "complex") return "1";
    if (k === "realTensor")
      return (types[0] as { isLogical?: boolean }).isLogical ? "0" : "1";
    if (k === "complexTensor") return "1";
    if (k === "logical" || k === "string" || k === "char") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "isinteger",
  typeRule: anyToLogical,
  apply: () => false,
  jitEmit: () => "0",
});

registerIBuiltin({
  name: "islogical",
  typeRule: anyToLogical,
  apply: args => {
    const v = args[0];
    if (typeof v === "boolean") return true;
    if (isRuntimeTensor(v) && v._isLogical === true) return true;
    return false;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "logical") return "1";
    if (k === "realTensor")
      return (types[0] as { isLogical?: boolean }).isLogical ? "1" : "0";
    if (
      k === "number" ||
      k === "complex" ||
      k === "complexTensor" ||
      k === "string" ||
      k === "char"
    )
      return "0";
    return null;
  },
});

registerIBuiltin({
  name: "ischar",
  typeRule: anyToLogical,
  apply: args => isRuntimeChar(args[0]),
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "char") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "isstring",
  typeRule: anyToLogical,
  apply: args => isRuntimeString(args[0]),
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "string") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "iscell",
  typeRule: anyToLogical,
  apply: args => isRuntimeCell(args[0]),
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "isstruct",
  typeRule: anyToLogical,
  apply: args => isRuntimeStruct(args[0]) || isRuntimeStructArray(args[0]),
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "issparse",
  typeRule: anyToLogical,
  apply: args => isRuntimeSparseMatrix(args[0]),
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

// ── Shape predicates ─────────────────────────────────────────────────────

registerIBuiltin({
  name: "isscalar",
  typeRule: anyToLogical,
  apply: args => {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return true;
    if (isRuntimeComplexNumber(v)) return true;
    if (isRuntimeTensor(v)) return v.data.length === 1;
    return false;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    if (k === "string") return "1";
    return null; // tensors/char need runtime check
  },
});

registerIBuiltin({
  name: "isempty",
  typeRule: anyToLogical,
  apply: args => {
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
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex" || k === "string")
      return "0";
    return null; // tensors/char need runtime check
  },
});

registerIBuiltin({
  name: "isvector",
  typeRule: anyToLogical,
  apply: args => {
    const shape = getShape(args[0]);
    return shape.filter(d => d > 1).length <= 1;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    return null;
  },
});

registerIBuiltin({
  name: "isrow",
  typeRule: anyToLogical,
  apply: args => {
    const shape = getShape(args[0]);
    return shape.length === 2 && shape[0] === 1;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    return null;
  },
});

registerIBuiltin({
  name: "iscolumn",
  typeRule: anyToLogical,
  apply: args => {
    const shape = getShape(args[0]);
    return shape.length === 2 && shape[1] === 1;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    return null;
  },
});

registerIBuiltin({
  name: "ismatrix",
  typeRule: anyToLogical,
  apply: args => getShape(args[0]).length <= 2,
  jitEmit: (_args, types) => {
    // JIT only handles ≤2D, so always true for known types
    if (types[0]?.kind !== "unknown") return "1";
    return null;
  },
});

// ── Shape queries ────────────────────────────────────────────────────────

registerIBuiltin({
  name: "numel",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind === "unknown") return null;
    return [{ kind: "number", nonneg: true }];
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
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex" || k === "string")
      return "1";
    return null;
  },
});

registerIBuiltin({
  name: "length",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind === "unknown") return null;
    return [{ kind: "number", nonneg: true }];
  },
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
    if (isRuntimeStructArray(v)) return v.elements.length;
    return 1;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex" || k === "string")
      return "1";
    return null;
  },
});

registerIBuiltin({
  name: "ndims",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind === "unknown") return null;
    return [{ kind: "number", nonneg: true }];
  },
  apply: args => {
    const v = args[0];
    if (isRuntimeTensor(v)) return Math.max(2, v.shape.length);
    return 2;
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (
      k === "number" ||
      k === "logical" ||
      k === "complex" ||
      k === "string" ||
      k === "char"
    )
      return "2";
    return null;
  },
});

registerIBuiltin({
  name: "size",
  typeRule: (argTypes, _nargout) => {
    if (argTypes.length === 1) {
      if (argTypes[0].kind === "unknown") return null;
      const a = argTypes[0];
      const ndims =
        a.kind === "realTensor" || a.kind === "complexTensor"
          ? a.shape.length
          : 2;
      return [{ kind: "realTensor", shape: [1, ndims], nonneg: true }];
    }
    if (argTypes.length === 2) {
      if (argTypes[0].kind === "unknown") return null;
      const dimKind = argTypes[1].kind;
      if (dimKind !== "number" && dimKind !== "logical") return null;
      return [{ kind: "number", nonneg: true }];
    }
    return null;
  },
  apply: (args, nargout) => {
    const v = args[0];
    const shape = getShape(v);

    if (args.length === 2) {
      const dim =
        typeof args[1] === "number" ? Math.round(args[1]) : args[1] ? 1 : 0;
      return dim > 0 && dim <= shape.length ? shape[dim - 1] : 1;
    }

    if (nargout > 1) {
      const result: RuntimeValue[] = [];
      for (let i = 0; i < nargout; i++) {
        result.push(i < shape.length ? shape[i] : 1);
      }
      return result;
    }

    return makeTensor(new FloatXArray(shape), undefined, [1, shape.length]);
  },
  jitEmit: (args, types) => {
    if (args.length === 2) {
      const k = types[0]?.kind;
      if (
        k === "number" ||
        k === "logical" ||
        k === "complex" ||
        k === "string"
      )
        return "1";
    }
    return null;
  },
});

// ── class() ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "class",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    switch (k) {
      case "number":
      case "complex":
      case "complexTensor":
        return [{ kind: "string", value: "double" }];
      case "logical":
        return [{ kind: "string", value: "logical" }];
      case "realTensor":
        return [
          {
            kind: "string",
            value: (argTypes[0] as { isLogical?: boolean }).isLogical
              ? "logical"
              : "double",
          },
        ];
      case "string":
        return [{ kind: "string", value: "string" }];
      case "char":
        return [{ kind: "string", value: "char" }];
      case "unknown":
        return null;
      default:
        return null;
    }
  },
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return "double";
    if (typeof v === "boolean") return "logical";
    if (isRuntimeString(v)) return "string";
    if (isRuntimeChar(v)) return "char";
    if (isRuntimeTensor(v)) return v._isLogical ? "logical" : "double";
    if (isRuntimeCell(v)) return "cell";
    if (isRuntimeStruct(v)) return "struct";
    if (isRuntimeFunction(v)) return "function_handle";
    if (isRuntimeClassInstance(v)) return v.className;
    if (isRuntimeComplexNumber(v)) return "double";
    if (isRuntimeSparseMatrix(v)) return "double";
    if (isRuntimeDummyHandle(v)) return "dummy_handle";
    if (isRuntimeStructArray(v)) return "struct";
    return "unknown";
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    switch (k) {
      case "number":
      case "complex":
      case "complexTensor":
        return '"double"';
      case "logical":
        return '"logical"';
      case "realTensor":
        return (types[0] as { isLogical?: boolean }).isLogical
          ? '"logical"'
          : '"double"';
      case "string":
        return '"string"';
      case "char":
        return '"char"';
      default:
        return null;
    }
  },
});

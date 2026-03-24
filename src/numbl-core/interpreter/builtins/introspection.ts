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
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "number") return true;
        if (isRuntimeTensor(v) && !v._isLogical) return true;
        if (isRuntimeComplexNumber(v)) return true;
        if (isRuntimeSparseMatrix(v)) return true;
        if (isRuntimeClassInstance(v) && v._builtinData !== undefined)
          return true;
        return false;
      },
    };
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
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "number") return true;
        if (isRuntimeTensor(v) && !v._isLogical) return true;
        if (isRuntimeComplexNumber(v)) return true;
        if (isRuntimeSparseMatrix(v)) return true;
        return false;
      },
    };
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
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return { outputTypes, apply: () => false };
  },
  jitEmit: () => "0",
});

registerIBuiltin({
  name: "islogical",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "boolean") return true;
        if (isRuntimeTensor(v) && v._isLogical === true) return true;
        return false;
      },
    };
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
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return { outputTypes, apply: args => isRuntimeChar(args[0]) };
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "char") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "isstring",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return { outputTypes, apply: args => isRuntimeString(args[0]) };
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "string") return "1";
    if (k !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "iscell",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return { outputTypes, apply: args => isRuntimeCell(args[0]) };
  },
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "isstruct",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => isRuntimeStruct(args[0]) || isRuntimeStructArray(args[0]),
    };
  },
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

registerIBuiltin({
  name: "issparse",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return { outputTypes, apply: args => isRuntimeSparseMatrix(args[0]) };
  },
  jitEmit: (_args, types) => {
    if (types[0]?.kind !== "unknown") return "0";
    return null;
  },
});

// ── Shape predicates ─────────────────────────────────────────────────────

registerIBuiltin({
  name: "isscalar",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "number" || typeof v === "boolean") return true;
        if (isRuntimeComplexNumber(v)) return true;
        if (isRuntimeTensor(v)) return v.data.length === 1;
        return false;
      },
    };
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
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
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
    };
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
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const shape = getShape(args[0]);
        return shape.filter(d => d > 1).length <= 1;
      },
    };
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    return null;
  },
});

registerIBuiltin({
  name: "isrow",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const shape = getShape(args[0]);
        return shape.length === 2 && shape[0] === 1;
      },
    };
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    return null;
  },
});

registerIBuiltin({
  name: "iscolumn",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const shape = getShape(args[0]);
        return shape.length === 2 && shape[1] === 1;
      },
    };
  },
  jitEmit: (_args, types) => {
    const k = types[0]?.kind;
    if (k === "number" || k === "logical" || k === "complex") return "1";
    return null;
  },
});

registerIBuiltin({
  name: "ismatrix",
  resolve: argTypes => {
    const outputTypes = anyToLogical(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => getShape(args[0]).length <= 2,
    };
  },
  jitEmit: (_args, types) => {
    // JIT only handles ≤2D, so always true for known types
    if (types[0]?.kind !== "unknown") return "1";
    return null;
  },
});

// ── Shape queries ────────────────────────────────────────────────────────

registerIBuiltin({
  name: "numel",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind === "unknown") return null;
    return {
      outputTypes: [{ kind: "number", nonneg: true }],
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
    };
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
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind === "unknown") return null;
    return {
      outputTypes: [{ kind: "number", nonneg: true }],
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
    };
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
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind === "unknown") return null;
    return {
      outputTypes: [{ kind: "number", nonneg: true }],
      apply: args => {
        const v = args[0];
        if (isRuntimeTensor(v)) return Math.max(2, v.shape.length);
        return 2;
      },
    };
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
  resolve: (argTypes, _nargout) => {
    if (argTypes.length === 1) {
      if (argTypes[0].kind === "unknown") return null;
      const a = argTypes[0];
      const ndims =
        a.kind === "realTensor" || a.kind === "complexTensor"
          ? a.shape.length
          : 2;
      return {
        outputTypes: [{ kind: "realTensor", shape: [1, ndims], nonneg: true }],
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
          return makeTensor(new FloatXArray(shape), undefined, [
            1,
            shape.length,
          ]);
        },
      };
    }
    if (argTypes.length === 2) {
      if (argTypes[0].kind === "unknown") return null;
      const dimKind = argTypes[1].kind;
      if (dimKind !== "number" && dimKind !== "logical") return null;
      return {
        outputTypes: [{ kind: "number", nonneg: true }],
        apply: args => {
          const v = args[0];
          const shape = getShape(v);
          const dim =
            typeof args[1] === "number" ? Math.round(args[1]) : args[1] ? 1 : 0;
          return dim > 0 && dim <= shape.length ? shape[dim - 1] : 1;
        },
      };
    }
    return null;
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
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const k = argTypes[0].kind;
    let outputTypes: JitType[];
    switch (k) {
      case "number":
      case "complex":
      case "complexTensor":
        outputTypes = [{ kind: "string", value: "double" }];
        break;
      case "logical":
        outputTypes = [{ kind: "string", value: "logical" }];
        break;
      case "realTensor":
        outputTypes = [
          {
            kind: "string",
            value: (argTypes[0] as { isLogical?: boolean }).isLogical
              ? "logical"
              : "double",
          },
        ];
        break;
      case "string":
        outputTypes = [{ kind: "string", value: "string" }];
        break;
      case "char":
        outputTypes = [{ kind: "string", value: "char" }];
        break;
      case "unknown":
        return null;
      default:
        return null;
    }
    return {
      outputTypes,
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
    };
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

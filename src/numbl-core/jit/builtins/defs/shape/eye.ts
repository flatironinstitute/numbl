/**
 * `eye(n)` and `eye(m, n)` — 2-D identity matrices.
 */

import { UnsupportedConstruct, TypeError } from "../../../lowering/errors.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  DIM_ONE,
  scalarDouble,
  tensorDouble,
  tensorDoubleFromDims,
  isNumeric,
  isScalar,
} from "../../../lowering/types.js";
import type { DimInfo, NumericType, Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { mtoc2_eye_rect as jsEyeRect } from "../../runtime/snippets.gen.js";
import { CHECK_DIM_SNIPPET, checkDimRuntime } from "./_construct.js";

interface ResolvedAxis {
  exact: number | undefined;
  argIndex: number;
}

function resolveArgs(argTypes: Type[]): ResolvedAxis[] {
  if (argTypes.length < 1 || argTypes.length > 2) {
    throw new UnsupportedConstruct(
      `'eye' supports 1 or 2 arguments (got ${argTypes.length})`
    );
  }
  if (
    argTypes.length === 1 &&
    isNumeric(argTypes[0]) &&
    !isScalar(argTypes[0])
  ) {
    throw new UnsupportedConstruct(
      `'eye' with a size-vector argument is not yet supported; ` +
        `use 'eye(m, n)' instead`
    );
  }
  const out: ResolvedAxis[] = [];
  for (let i = 0; i < argTypes.length; i++) {
    const a = argTypes[i];
    if (!isNumeric(a) || a.elem !== "double" || a.isComplex) {
      throw new TypeError(
        `'eye' arg ${i + 1} must be a scalar real double (got ${a.kind})`
      );
    }
    if (!isScalar(a)) {
      throw new TypeError(
        `'eye' arg ${i + 1} must be a scalar real double (got tensor)`
      );
    }
    const v = exactDouble(a);
    if (v !== undefined) {
      if (!Number.isInteger(v) || v < 0) {
        throw new TypeError(
          `'eye' arg ${i + 1} must be a finite non-negative integer (got ${v})`
        );
      }
      out.push({ exact: v, argIndex: i });
    } else {
      out.push({ exact: undefined, argIndex: i });
    }
  }
  if (out.length === 1) {
    return [out[0], { exact: out[0].exact, argIndex: out[0].argIndex }];
  }
  return out;
}

export const eye: Builtin = {
  name: "eye",
  transfer(argTypes, nargout) {
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'eye' does not support multi-output (nargout=${nargout})`
      );
    }
    const axes = resolveArgs(argTypes);
    const rows = axes[0].exact;
    const cols = axes[1].exact;
    if (rows !== undefined && cols !== undefined) {
      const shape = [rows, cols];
      const total = rows * cols;
      if (total === 0) return [tensorDouble(shape)];
      if (rows === 1 && cols === 1) return [scalarDouble("positive", 1)];
      if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
        const data = new Float64Array(total);
        const min = rows < cols ? rows : cols;
        for (let i = 0; i < min; i++) data[i + i * rows] = 1;
        return [tensorDouble(shape, data)];
      }
      const t = tensorDouble(shape);
      t.sign = "nonneg";
      return [t];
    }
    const dims: DimInfo[] = axes.map(a =>
      a.exact === undefined
        ? { kind: "unknown" }
        : a.exact === 1
          ? DIM_ONE
          : { kind: "exact", value: a.exact }
    );
    const t: NumericType = tensorDoubleFromDims(dims);
    t.sign = "nonneg";
    return [t];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_eye");
    const axes = resolveArgs(argTypes);
    const rows = axes[0].exact;
    const cols = axes[1].exact;
    if (rows === 1 && cols === 1) return "1.0";
    // A non-integer dynamic dim is an error (interpreter validateDim +
    // MATLAB), not a silent truncation — validate via mtoc2_check_dim.
    if (rows === undefined || cols === undefined) useRuntime(CHECK_DIM_SNIPPET);
    const rowsC =
      rows !== undefined
        ? `${rows}L`
        : `mtoc2_check_dim(${argsC[axes[0].argIndex]})`;
    const colsC =
      cols !== undefined
        ? `${cols}L`
        : `mtoc2_check_dim(${argsC[axes[1].argIndex]})`;
    if (
      argTypes.length === 1 &&
      rows === undefined &&
      cols === undefined &&
      axes[0].argIndex === axes[1].argIndex
    ) {
      return `mtoc2_eye_square(mtoc2_check_dim(${argsC[axes[0].argIndex]}))`;
    }
    return `mtoc2_eye_rect(${rowsC}, ${colsC})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_eye");
    const axes = resolveArgs(argTypes);
    const rows = axes[0].exact;
    const cols = axes[1].exact;
    if (rows === 1 && cols === 1) return "1";
    if (rows === undefined || cols === undefined) useRuntime(CHECK_DIM_SNIPPET);
    const rowsJs =
      rows !== undefined
        ? String(rows)
        : `mtoc2_check_dim(${argsJs[axes[0].argIndex]})`;
    const colsJs =
      cols !== undefined
        ? String(cols)
        : `mtoc2_check_dim(${argsJs[axes[1].argIndex]})`;
    if (
      argTypes.length === 1 &&
      rows === undefined &&
      cols === undefined &&
      axes[0].argIndex === axes[1].argIndex
    ) {
      return `mtoc2_eye_square(mtoc2_check_dim(${argsJs[axes[0].argIndex]}))`;
    }
    return `mtoc2_eye_rect(${rowsJs}, ${colsJs})`;
  },
  call({ args, argTypes }) {
    const axes = resolveArgs(argTypes);
    const rows =
      axes[0].exact !== undefined
        ? axes[0].exact
        : checkDimRuntime(
            typeof args[axes[0].argIndex] === "number"
              ? (args[axes[0].argIndex] as number)
              : Number(args[axes[0].argIndex] as object)
          );
    const cols =
      axes[1].exact !== undefined
        ? axes[1].exact
        : checkDimRuntime(
            typeof args[axes[1].argIndex] === "number"
              ? (args[axes[1].argIndex] as number)
              : Number(args[axes[1].argIndex] as object)
          );
    if (rows === 1 && cols === 1) return [1];
    return [jsEyeRect(rows, cols) as unknown as RuntimeTensor];
  },
};

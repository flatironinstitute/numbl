/**
 * Map a runtime value back into a Type. Used by the interpreter to
 * feed `argTypes` into a builtin's `transfer` (for validation) and
 * `call` (for type-dispatched implementations) — so that the three
 * hooks (`emitC`, `emitJs`, `call`) all branch on the same `argTypes`
 * shape rather than reinventing dispatch per-backend.
 */

import {
  cellTuple,
  cellUniform,
  EXACT_ARRAY_MAX_ELEMENTS,
  scalarDouble,
  scalarLogical,
  scalarComplex,
  signFromNumber,
  tensorComplex,
  tensorComplexFromDims,
  tensorDouble,
  UNKNOWN,
  type DimInfo,
  type Type,
} from "../lowering/types.js";
import {
  isChar,
  isComplexValue,
  isTensor,
  type RuntimeValue,
} from "./value.js";

export function inferTypeFromValue(v: RuntimeValue): Type {
  if (typeof v === "number") return scalarDouble(signFromNumber(v), v);
  if (typeof v === "boolean") return scalarLogical(v);
  if (typeof v === "string") return { kind: "String", exact: v };
  if (isChar(v)) return { kind: "Char", exact: v.value };
  if (isTensor(v)) {
    // Complex tensor → carry both lanes as exact when they fit the
    // cap; otherwise produce a shape-only complex type.
    if (v.imag !== undefined) {
      if (v.data.length <= EXACT_ARRAY_MAX_ELEMENTS) {
        return tensorComplex(v.shape.slice(), {
          re: new Float64Array(v.data),
          im: new Float64Array(v.imag),
        });
      }
      const dims: DimInfo[] = v.shape.map(n => ({
        kind: "exact" as const,
        value: n,
      }));
      return tensorComplexFromDims(dims);
    }
    // Real tensor — carry the data as `exact` when it fits the
    // lattice's cap so dim-vector-consuming builtins
    // (`zeros(size(xs))`, `reshape(A, [r c])`, `sum(v, ...)` with a
    // vector dim, …) can read the runtime values via the same
    // `exactRealArray` path c-aot uses. Above the cap we fall back
    // to shape-only.
    if (v.data.length <= EXACT_ARRAY_MAX_ELEMENTS) {
      return tensorDouble(v.shape.slice(), new Float64Array(v.data));
    }
    return tensorDouble(v.shape.slice());
  }
  if (isComplexValue(v)) {
    return scalarComplex({ re: v.re, im: v.im });
  }
  if (typeof v === "object" && v !== null) {
    // Cell-array runtime value: tagged with `mtoc2Tag: "cell"`. Build
    // a `CellType` so builtins that branch on the type (iscell, disp)
    // see the same shape they would in the AOT backends. Use tuple
    // mode when the total slot count fits the exact cap so per-slot
    // types are preserved; uniform mode (with empty-double elem) for
    // larger or non-bounded cells.
    if ((v as { mtoc2Tag?: string }).mtoc2Tag === "cell") {
      const c = v as unknown as {
        mtoc2Tag: "cell";
        shape: number[];
        data: RuntimeValue[];
      };
      if (c.data.length <= EXACT_ARRAY_MAX_ELEMENTS) {
        return cellTuple(
          c.shape.slice(),
          c.data.map(slot => inferTypeFromValue(slot))
        );
      }
      const dims: DimInfo[] = c.shape.map(n => ({
        kind: "exact" as const,
        value: n,
      }));
      return cellUniform(dims, {
        kind: "Numeric",
        elem: "double",
        isComplex: false,
        dims: [
          { kind: "exact", value: 0 },
          { kind: "exact", value: 0 },
        ],
        shape: [0, 0],
        sign: "unknown",
      });
    }
    // Class instance: tagged with `mtoc2Class` by `constructClassInstance`
    // so workspace dispatch can detect the receiver type for
    // `method(obj, args)` resolution.
    const className = (v as { mtoc2Class?: string }).mtoc2Class;
    if (className !== undefined) {
      const properties: { name: string; ty: Type }[] = [];
      for (const k of Object.keys(v as Record<string, RuntimeValue>)) {
        properties.push({
          name: k,
          ty: inferTypeFromValue((v as Record<string, RuntimeValue>)[k]),
        });
      }
      return { kind: "Class", className, properties };
    }
    // Struct: infer per-field types recursively. The interpreter
    // dispatches builtins by argType, so a struct value needs a
    // matching `Struct`-kind type for `disp(s)` / member loads to
    // route through the right paths.
    const fields: { name: string; ty: Type }[] = [];
    for (const k of Object.keys(v as Record<string, RuntimeValue>)) {
      fields.push({
        name: k,
        ty: inferTypeFromValue((v as Record<string, RuntimeValue>)[k]),
      });
    }
    return { kind: "Struct", fields };
  }
  return UNKNOWN;
}

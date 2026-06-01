/**
 * `var` / `std` reductions. Two-pass (mean, then mean-square-deviation),
 * matching numbl's `varianceOf` (interpreter/builtins/reductions.ts):
 *
 *   var(A)          weight 0 (÷ n-1), reduce along the first non-singleton axis
 *   var(A, w)       w ∈ {0, 1}: 0 ÷ (n-1), 1 ÷ n
 *   var(A, w, dim)  reduce along axis `dim`
 *   var(A, w, 'all')reduce every element
 *   std(...)        = sqrt(var(...))
 *
 * Real only. Complex input, vector weights, `omitnan`/`includenan`,
 * the empty-weight `var(A, [], dim)` form, and runtime (non-exact) w/dim
 * all decline to the interpreter. A scalar input yields 0 (numbl returns
 * 0 for any scalar, regardless of value or weight).
 */

import { UnsupportedConstruct, TypeError } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  scalarDouble,
  tensorDouble,
  tensorDoubleFromDims,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import {
  chooseDefaultAxis,
  reduceConcreteShape,
  reduceLatticeDims,
  type AxisAll,
  type AxisFixed,
} from "./_shape.js";
import type { RuntimeTensor, RuntimeValue } from "../../../runtime/value.js";
import {
  mtoc2_var_all,
  mtoc2_var_dim,
  mtoc2_std_all,
  mtoc2_std_dim,
} from "../../runtime/tensor_ops/tensor_var.js";

type AxisChoice = AxisAll | AxisFixed | { kind: "default" };

/** Parse the weight slot (arg 1). Returns 0 or 1; declines anything
 *  else (vector weights, runtime values, the empty `[]` form). */
function parseWeight(name: string, wType: Type | undefined): 0 | 1 {
  if (wType === undefined) return 0;
  if (!isNumeric(wType) || wType.isComplex || !isScalar(wType)) {
    throw new UnsupportedConstruct(
      `'${name}' weight must be a scalar 0 or 1 (vector weights / '[]' ` +
        `fall back to the interpreter)`
    );
  }
  const v = exactDouble(wType);
  if (v === undefined) {
    throw new UnsupportedConstruct(
      `'${name}' weight must be a statically-known 0 or 1`
    );
  }
  if (v !== 0 && v !== 1) {
    throw new TypeError(`'${name}' weight must be 0 or 1 (got ${v})`);
  }
  return v === 1 ? 1 : 0;
}

/** Parse the dim slot (arg 2): a statically-known positive integer or
 *  the literal 'all'. */
function parseDimSlot(name: string, dimType: Type | undefined): AxisChoice {
  if (dimType === undefined) return { kind: "default" };
  if (dimType.kind === "String" || dimType.kind === "Char") {
    if (dimType.exact === "all") return { kind: "all" };
    throw new UnsupportedConstruct(
      `'${name}' dim arg: text literal must be 'all'`
    );
  }
  if (!isNumeric(dimType) || dimType.isComplex || !isScalar(dimType)) {
    throw new TypeError(
      `'${name}' dim arg must be a scalar real integer or the string 'all'`
    );
  }
  const v = exactDouble(dimType);
  if (v === undefined) {
    throw new UnsupportedConstruct(
      `'${name}' dim arg must be a statically-known integer`
    );
  }
  if (!Number.isFinite(v) || !Number.isInteger(v) || v < 1) {
    throw new TypeError(
      `'${name}' dim arg must be a finite positive integer (got ${v})`
    );
  }
  return { kind: "fixed", dim: v };
}

/** Validate input + parse w/dim. Throws/declines on unsupported forms. */
function planVar(
  name: string,
  argTypes: Type[]
): { input: NumericType; w: 0 | 1; axis: AxisChoice } {
  if (argTypes.length < 1 || argTypes.length > 3) {
    throw new TypeError(
      `'${name}' expects 1..3 arg(s), got ${argTypes.length}`
    );
  }
  const input = argTypes[0];
  if (!isNumeric(input)) {
    throw new TypeError(`'${name}' arg must be numeric (got ${input.kind})`);
  }
  if (input.isComplex) {
    throw new UnsupportedConstruct(
      `'${name}' on complex input is not JIT-compiled`
    );
  }
  if (input.elem !== "double" && input.elem !== "logical") {
    throw new TypeError(
      `'${name}' arg must be double or logical (got ${input.elem})`
    );
  }
  const w = parseWeight(name, argTypes[1]);
  const axis = parseDimSlot(name, argTypes[2]);
  return { input, w, axis };
}

function varResultType(input: NumericType, axis: AxisAll | AxisFixed): Type {
  if (axis.kind === "all") return scalarDouble("nonneg");
  if (input.shape !== undefined) {
    const r = reduceConcreteShape(input.shape, axis.dim);
    if (r.scalar) return scalarDouble("nonneg");
    const out = tensorDouble(r.shape);
    out.sign = "nonneg";
    return out;
  }
  const r = reduceLatticeDims(input.dims, axis.dim);
  if (r.scalar) return scalarDouble("nonneg");
  const out = tensorDoubleFromDims(r.dims);
  out.sign = "nonneg";
  return out;
}

export function defineVariance(name: "var" | "std"): Builtin {
  const allKernelJs = name === "std" ? mtoc2_std_all : mtoc2_var_all;
  const dimKernelJs = name === "std" ? mtoc2_std_dim : mtoc2_var_dim;
  return {
    name,
    transfer(argTypes, nargout) {
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      const { input, axis } = planVar(name, argTypes);
      // Scalar input → 0 (numbl returns 0 for any scalar).
      if (isScalar(input)) return [scalarDouble("zero", 0)];
      const resolved =
        axis.kind === "default" ? chooseDefaultAxis(name, input) : axis;
      return [varResultType(input, resolved)];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      useRuntime("mtoc2_tensor_var");
      const { input, w, axis } = planVar(name, argTypes);
      if (isScalar(input)) return `((void)(${argsC[0]}), 0.0)`;
      const resolved =
        axis.kind === "default" ? chooseDefaultAxis(name, input) : axis;
      if (collapsesToScalar(input, resolved)) {
        return `mtoc2_${name}_all(${argsC[0]}, ${w})`;
      }
      return `mtoc2_${name}_dim(${argsC[0]}, ${(resolved as AxisFixed).dim}, ${w})`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      useRuntime("mtoc2_tensor_var");
      const { input, w, axis } = planVar(name, argTypes);
      if (isScalar(input)) return `((${argsJs[0]}), 0)`;
      const resolved =
        axis.kind === "default" ? chooseDefaultAxis(name, input) : axis;
      if (collapsesToScalar(input, resolved)) {
        return `mtoc2_${name}_all(${argsJs[0]}, ${w})`;
      }
      return `mtoc2_${name}_dim(${argsJs[0]}, ${(resolved as AxisFixed).dim}, ${w})`;
    },
    call({ args, argTypes }) {
      const { input, w, axis } = planVar(name, argTypes);
      if (isScalar(input)) return [0];
      const resolved =
        axis.kind === "default" ? chooseDefaultAxis(name, input) : axis;
      const t = args[0] as RuntimeTensor;
      if (collapsesToScalar(input, resolved)) {
        return [allKernelJs(t, w) as RuntimeValue];
      }
      return [dimKernelJs(t, (resolved as AxisFixed).dim, w) as RuntimeValue];
    },
  };
}

/** True when reducing `input` along `axis` collapses to a scalar (so
 *  the `_all` kernel applies instead of `_dim`). */
function collapsesToScalar(
  input: NumericType,
  axis: AxisAll | AxisFixed
): boolean {
  if (axis.kind === "all") return true;
  if (input.shape !== undefined) {
    return reduceConcreteShape(input.shape, axis.dim).scalar;
  }
  return reduceLatticeDims(input.dims, axis.dim).scalar;
}

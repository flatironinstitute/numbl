import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  isMultiElement,
  signFromExactArray,
  tensorDouble,
  tensorComplexFromDims,
  type NumericType,
  type Sign,
} from "../../../lowering/types.js";
import {
  type Builtin,
  requireEmitC,
  requireEmitJs,
  requireCall,
} from "../../registry.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import {
  mtoc2_tensor_mtimes_real,
  mtoc2_tensor_mtimes_real_scalar,
  mtoc2_tensor_mtimes_complex,
  mtoc2_tensor_mtimes_complex_scalar,
} from "../../runtime/snippets.gen.js";
import { times } from "./times.js";
import { exactRealArray, requireRealOrComplex } from "../_shared.js";

/** Result-sign rule for matrix multiplication. Each output element is
 *  a sum of products; we only assert sign when both operands are
 *  fully non-negative (or both non-positive), since any sign mixing
 *  in either operand can produce zeros from cancellation. Conservative
 *  by design — mtimes results are rarely sign-pinnable. */
function mtimesSign(a: NumericType, b: NumericType): Sign {
  const aNonneg =
    a.sign === "positive" || a.sign === "nonneg" || a.sign === "zero";
  const bNonneg =
    b.sign === "positive" || b.sign === "nonneg" || b.sign === "zero";
  if (aNonneg && bNonneg) return "nonneg";
  const aNonpos =
    a.sign === "negative" || a.sign === "nonpositive" || a.sign === "zero";
  const bNonpos =
    b.sign === "negative" || b.sign === "nonpositive" || b.sign === "zero";
  if (aNonpos && bNonpos) return "nonneg";
  if (aNonneg && bNonpos) return "nonpositive";
  if (aNonpos && bNonneg) return "nonpositive";
  return "unknown";
}

/** Compute `A * B` exactly given column-major data and statically known
 *  shapes. */
function mtimesExact(
  ad: Float64Array,
  m: number,
  k: number,
  bd: Float64Array,
  n: number
): Float64Array {
  const out = new Float64Array(m * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let acc = 0;
      for (let p = 0; p < k; p++) {
        acc += ad[i + p * m] * bd[p + j * k];
      }
      out[i + j * m] = acc;
    }
  }
  return out;
}

// `mtimes` (matrix *): mirrors elementwise `times` when at least one
// arg is scalar; routes to a real 2-D matrix-multiply runtime helper
// when both args are tensors.
export const mtimes: Builtin = {
  name: "mtimes",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'mtimes' expects 2 arg(s), got ${argTypes.length}`);
    }
    requireRealOrComplex(argTypes[0], `'mtimes' arg 1`);
    requireRealOrComplex(argTypes[1], `'mtimes' arg 2`);
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    if (!isMultiElement(a) || !isMultiElement(b)) {
      // Scalar mtimes delegates to elementwise times — identical fold
      // and codegen. The `times` transfer handles complex contamination.
      return times.transfer(argTypes, nargout);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'mtimes' does not support multi-output (nargout=${nargout})`
      );
    }
    // Tensor mtimes: real and complex both supported. Complex result
    // when either operand is complex.
    if (a.elem !== "double" || b.elem !== "double") {
      throw new TypeError(
        `'mtimes' tensor operands must be double (got ${a.elem}, ${b.elem})`
      );
    }
    const resultIsComplex = a.isComplex || b.isComplex;
    // Both tensors. v1: 2-D, statically inner-dim-matching when shapes
    // are known.
    if (a.shape !== undefined && a.shape.length !== 2) {
      throw new UnsupportedConstruct(
        `'mtimes' on a ${a.shape.length}-D tensor is not supported ` +
          `(MATLAB defines matrix multiplication only on 2-D operands)`
      );
    }
    if (b.shape !== undefined && b.shape.length !== 2) {
      throw new UnsupportedConstruct(
        `'mtimes' on a ${b.shape.length}-D tensor is not supported ` +
          `(MATLAB defines matrix multiplication only on 2-D operands)`
      );
    }
    if (a.dims.length !== 2 || b.dims.length !== 2) {
      throw new UnsupportedConstruct(
        `'mtimes' requires 2-D tensor operands (got ${a.dims.length}-D, ${b.dims.length}-D)`
      );
    }
    // Read dims directly (not `shape`) so partially-known cases —
    // e.g. `[?, 1] * [1, ?]` — preserve their known axes in the
    // result type. `shape` is only set when every axis is exact.
    const aRows = a.dims[0].kind === "exact" ? a.dims[0].value : undefined;
    const aCols = a.dims[1].kind === "exact" ? a.dims[1].value : undefined;
    const bRows = b.dims[0].kind === "exact" ? b.dims[0].value : undefined;
    const bCols = b.dims[1].kind === "exact" ? b.dims[1].value : undefined;
    if (aCols !== undefined && bRows !== undefined && aCols !== bRows) {
      throw new TypeError(
        `'mtimes' inner-dim mismatch: ${aRows ?? "?"}×${aCols} * ${bRows}×${bCols ?? "?"}`
      );
    }
    // Result shape: outer dims of A and B.
    const m = aRows;
    const n = bCols;
    const sign = mtimesSign(a, b);
    if (m !== undefined && n !== undefined) {
      const k = aCols ?? bRows!;
      const total = m * n;
      // Exact-fold for the real path when both inputs are exact and
      // the result fits the cap. Complex tensors don't have an exact
      // carrier of the same shape (the {re, im} carrier is used only
      // for fully-exact tensors); skip the fold there.
      if (!resultIsComplex) {
        const ad = exactRealArray(a);
        const bd = exactRealArray(b);
        if (
          ad !== undefined &&
          bd !== undefined &&
          total <= EXACT_ARRAY_MAX_ELEMENTS
        ) {
          const out = mtimesExact(ad, m, k, bd, n);
          const t = tensorDouble([m, n], out);
          return [{ ...t, sign: signFromExactArray(out) }];
        }
        const t = tensorDouble([m, n]);
        return [{ ...t, sign }];
      }
      // Complex result — no fold, no sign refinement.
      return [
        tensorComplexFromDims([
          { kind: "exact", value: m },
          { kind: "exact", value: n },
        ]),
      ];
    }
    // Partially-unknown shape (e.g. one axis is `unknown` due to a
    // runtime-only length): emit the helper call; shape stays as
    // unknown on the relevant axis.
    if (resultIsComplex) {
      return [
        tensorComplexFromDims([
          m !== undefined ? { kind: "exact", value: m } : { kind: "unknown" },
          n !== undefined ? { kind: "exact", value: n } : { kind: "unknown" },
        ]),
      ];
    }
    return [
      {
        kind: "Numeric",
        elem: "double",
        isComplex: false,
        dims: [
          m !== undefined ? { kind: "exact", value: m } : { kind: "unknown" },
          n !== undefined ? { kind: "exact", value: n } : { kind: "unknown" },
        ],
        sign,
      },
    ];
  },
  emitC(args) {
    const { argsC, argTypes, useRuntime } = args;
    if (!isMultiElement(argTypes[0]) || !isMultiElement(argTypes[1])) {
      return requireEmitC(times)(args);
    }
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    const isComplex = a.isComplex || b.isComplex;
    if (isComplex) {
      useRuntime("mtoc2_tensor_mtimes_complex");
      useRuntime("mtoc2_cscalar");
    } else {
      useRuntime("mtoc2_tensor_mtimes_real");
    }
    // 1×k * k×1 → 1×1 scalar. Transfer returns a scalar type for this
    // shape; consumers expect a `double` (or `double _Complex`) C
    // expression. The scalar helper does the inner product without
    // allocating a tensor.
    if (a.shape?.[0] === 1 && b.shape?.[1] === 1) {
      return isComplex
        ? `mtoc2_tensor_mtimes_complex_scalar(${argsC[0]}, ${argsC[1]})`
        : `mtoc2_tensor_mtimes_real_scalar(${argsC[0]}, ${argsC[1]})`;
    }
    return isComplex
      ? `mtoc2_tensor_mtimes_complex(${argsC[0]}, ${argsC[1]})`
      : `mtoc2_tensor_mtimes_real(${argsC[0]}, ${argsC[1]})`;
  },
  emitJs(args) {
    const { argsJs, argTypes, useRuntime } = args;
    if (!isMultiElement(argTypes[0]) || !isMultiElement(argTypes[1])) {
      return requireEmitJs(times)(args);
    }
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    const isComplex = a.isComplex || b.isComplex;
    if (isComplex) {
      useRuntime("mtoc2_tensor_mtimes_complex");
      useRuntime("mtoc2_cscalar");
    } else {
      useRuntime("mtoc2_tensor_mtimes_real");
    }
    if (a.shape?.[0] === 1 && b.shape?.[1] === 1) {
      return isComplex
        ? `mtoc2_tensor_mtimes_complex_scalar(${argsJs[0]}, ${argsJs[1]})`
        : `mtoc2_tensor_mtimes_real_scalar(${argsJs[0]}, ${argsJs[1]})`;
    }
    return isComplex
      ? `mtoc2_tensor_mtimes_complex(${argsJs[0]}, ${argsJs[1]})`
      : `mtoc2_tensor_mtimes_real(${argsJs[0]}, ${argsJs[1]})`;
  },
  call(args) {
    const { argTypes } = args;
    if (!isMultiElement(argTypes[0]) || !isMultiElement(argTypes[1])) {
      return requireCall(times)(args);
    }
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;
    const isComplex = a.isComplex || b.isComplex;
    const at = args.args[0] as RuntimeTensor;
    const bt = args.args[1] as RuntimeTensor;
    if (a.shape?.[0] === 1 && b.shape?.[1] === 1) {
      return isComplex
        ? [mtoc2_tensor_mtimes_complex_scalar(at, bt)]
        : [mtoc2_tensor_mtimes_real_scalar(at, bt)];
    }
    return isComplex
      ? [mtoc2_tensor_mtimes_complex(at, bt) as unknown as RuntimeTensor]
      : [mtoc2_tensor_mtimes_real(at, bt) as unknown as RuntimeTensor];
  },
  // Elementwise per-slot template — only valid when at least one
  // operand is scalar (tensor * tensor is matrix product). The
  // `isPureElementwiseExpr` predicate in
  // `src/codegen/emitTensorFused.ts` rejects the both-tensor case
  // so this branch only fires on the elementwise call.
  elementwise: true,
};

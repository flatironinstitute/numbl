/**
 * `repmat` builtin. Surface forms (mirrors numbl's
 * array-manipulation `repmat`):
 *
 *   repmat(A, n)            — `n×n` tile (single scalar rep → square)
 *   repmat(A, m, n, ...)    — Form A: variadic scalar reps per axis
 *   repmat(A, [m n p ...])  — Form B: dim-vector reps (must be static)
 *
 * Discipline:
 *   - A may be a scalar or a tensor of double / logical / complex
 *     elements. Complex inputs route through `mtoc2_tensor_repmat_complex`
 *     (or `_fill_nd_complex` for scalar A).
 *   - Each rep arg is a scalar real double; statically-known finite
 *     integers pin the corresponding output axis, dynamic scalars leave
 *     that axis as `unknown` in the result lattice.
 *   - Form B requires the dim vector to be a statically-known constant
 *     (no runtime-vector reps in v1).
 *   - Negative reps clamp to 0 (matching numbl/MATLAB), which yields an
 *     empty axis at runtime. Statically-known negatives propagate
 *     through the type system as 0 too.
 *   - Output shape is `padShape[i] * padReps[i]` where the input's
 *     shape and the reps vector are both right-padded with 1s to a
 *     common rank. There's one numbl quirk we replicate: scalar input
 *     with a single-element Form B vector (`repmat(5, [3])`) produces
 *     an `n×n` square instead of a 1-D vector.
 *
 * Codegen:
 *   - Scalar input: emit `mtoc2_tensor_fill_nd(<val>, ndim, dims)`
 *     directly — no need to build a 1×1 tensor first.
 *   - Tensor input: emit `mtoc2_tensor_repmat(<in>, ndim, dims)`.
 *
 * Exact-data folding: when the input is exact and the reps are exact,
 * the transfer computes the tiled data and pins it on the result type
 * (subject to the EXACT_ARRAY_MAX_ELEMENTS cap).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  EXACT_ARRAY_MAX_ELEMENTS,
  MTOC2_MAX_NDIM,
  isNumeric,
  isScalar,
  scalarComplex,
  scalarDouble,
  shapeNumel,
  signFromExactArray,
  signFromNumber,
  tensorComplex,
  tensorComplexFromDims,
  tensorDouble,
  tensorDoubleFromDims,
  typeToString,
} from "../../../lowering/types.js";
import type { DimInfo, NumericType, Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  exactComplex,
  exactComplexArray,
  exactDouble,
  exactRealArray,
} from "../_shared.js";
import type { RuntimeTensor, RuntimeValue } from "../../../runtime/value.js";
import { isComplexValue } from "../../../runtime/value.js";
import {
  mtoc2_tensor_fill_nd as jsFillNd,
  mtoc2_tensor_fill_nd_complex as jsFillNdComplex,
  mtoc2_tensor_repmat as jsRepmat,
  mtoc2_tensor_repmat_complex as jsRepmatComplex,
} from "../../runtime/snippets.gen.js";

/** Per-axis rep resolution. `argIndex` is the position in the original
 *  argTypes array (i.e. `argsC[axis.argIndex]`) that supplies the rep
 *  value at runtime. Synthetic axes inserted by the right-pad-with-1s
 *  normalizer are exact and carry argIndex of the leading rep arg as
 *  a placeholder (the emitter never reads argIndex for exact axes). */
type RepAxis =
  | { kind: "exact"; value: number; argIndex: number }
  | { kind: "dynamic"; argIndex: number };

interface ResolvedReps {
  /** Per-axis reps. Length is the natural rep length (before padding
   *  with 1s against input shape) — i.e. 2 for the `repmat(A, n)` and
   *  `repmat(A, m, n)` forms, N for `repmat(A, [m n p ...])`. */
  reps: RepAxis[];
  /** True when surfaced as Form A's single-arg `repmat(A, n)` (square
   *  expansion already applied). */
  squareFromScalar: boolean;
  /** True when surfaced as Form B (vector reps). */
  formB: boolean;
}

function resolveReps(argTypes: Type[]): ResolvedReps {
  if (argTypes.length < 2) {
    throw new UnsupportedConstruct(
      `'repmat' requires at least 2 arguments (got ${argTypes.length})`
    );
  }
  // Form B: single non-scalar numeric vector as the rep arg.
  if (
    argTypes.length === 2 &&
    isNumeric(argTypes[1]) &&
    !isScalar(argTypes[1])
  ) {
    const v = argTypes[1];
    if (v.isComplex || (v.elem !== "double" && v.elem !== "logical")) {
      throw new TypeError(
        `'repmat' rep vector must be a real double / logical tensor (got ${typeToString(v)})`
      );
    }
    const arr = exactRealArray(v);
    if (arr === undefined) {
      throw new UnsupportedConstruct(
        `'repmat' rep vector must be a statically-known constant in v1`
      );
    }
    if (arr.length < 1) {
      throw new TypeError(`'repmat' rep vector must be non-empty`);
    }
    if (arr.length > MTOC2_MAX_NDIM) {
      throw new UnsupportedConstruct(
        `'repmat' supports up to ${MTOC2_MAX_NDIM} rep dims (got ${arr.length})`
      );
    }
    const reps: RepAxis[] = [];
    for (let i = 0; i < arr.length; i++) {
      const rv = arr[i];
      if (!Number.isFinite(rv)) {
        throw new TypeError(
          `'repmat' rep ${i + 1} must be a finite integer (got ${rv})`
        );
      }
      const rounded = Math.round(rv);
      reps.push({
        kind: "exact",
        value: rounded < 0 ? 0 : rounded,
        argIndex: 1,
      });
    }
    return { reps, squareFromScalar: false, formB: true };
  }

  // Form A: variadic scalar reps.
  if (argTypes.length - 1 > MTOC2_MAX_NDIM) {
    throw new UnsupportedConstruct(
      `'repmat' supports up to ${MTOC2_MAX_NDIM} rep dims (got ${argTypes.length - 1})`
    );
  }
  const reps: RepAxis[] = [];
  for (let i = 1; i < argTypes.length; i++) {
    const r = argTypes[i];
    if (!isNumeric(r) || !isScalar(r)) {
      throw new TypeError(
        `'repmat' rep arg ${i} must be a scalar real numeric (got ${typeToString(r)})`
      );
    }
    if (r.isComplex || (r.elem !== "double" && r.elem !== "logical")) {
      throw new TypeError(
        `'repmat' rep arg ${i} must be a scalar real double / logical (got ${typeToString(r)})`
      );
    }
    const v = exactDouble(r);
    if (v === undefined) {
      reps.push({ kind: "dynamic", argIndex: i });
      continue;
    }
    if (!Number.isFinite(v)) {
      throw new TypeError(
        `'repmat' rep arg ${i} must be a finite integer (got ${v})`
      );
    }
    const rounded = Math.round(v);
    reps.push({
      kind: "exact",
      value: rounded < 0 ? 0 : rounded,
      argIndex: i,
    });
  }
  // Single-scalar form `repmat(A, n)` → square tile `[n, n]`. The two
  // axes share `argIndex = 1` so codegen evaluates the dim expression
  // once when n is dynamic.
  if (reps.length === 1) {
    return {
      reps: [reps[0], reps[0]],
      squareFromScalar: true,
      formB: false,
    };
  }
  return { reps, squareFromScalar: false, formB: false };
}

/** Compute the input's shape (padded right with 1s to at least 2 axes).
 *  Returns `undefined` if any input dim is non-exact. Matches numbl's
 *  `srcShape = v.shape.length >= 2 ? v.shape : [1, v.shape[0] || 1]`. */
function inputShape(a: NumericType): number[] | undefined {
  if (a.shape === undefined) return undefined;
  return a.shape.slice();
}

/** Apply numbl's scalar-input quirk: a Form B with a single-element
 *  rep vector produces an n×n square. Returns the effective reps. */
function effectiveRepsForScalarInput(r: ResolvedReps): RepAxis[] {
  if (r.formB && r.reps.length === 1) {
    return [r.reps[0], r.reps[0]];
  }
  return r.reps;
}

/** Tile an exact Float64Array into a new buffer with the given
 *  per-axis reps. Both padShape and padReps are pre-padded to the
 *  same length (out_ndim). */
function tileExact(
  src: Float64Array,
  padShape: number[],
  padReps: number[]
): Float64Array {
  const outNdim = padShape.length;
  const outDims = padShape.map((s, i) => s * padReps[i]);
  let outTotal = 1;
  for (const d of outDims) outTotal *= d;
  const out = new Float64Array(outTotal);
  if (outTotal === 0) return out;
  let inTotal = 1;
  for (const d of padShape) inTotal *= d;
  if (inTotal === 0) return out;
  out.set(src.subarray(0, inTotal), 0);
  const curShape = padShape.slice();
  let curTotal = inTotal;
  for (let d = 0; d < outNdim; d++) {
    const rep = padReps[d];
    if (rep === 1) continue;
    let blockSize = 1;
    for (let i = 0; i <= d; i++) blockSize *= curShape[i];
    if (rep === 0 || blockSize === 0) return new Float64Array(outTotal);
    const numBlocks = curTotal / blockSize;
    for (let b = numBlocks - 1; b >= 0; b--) {
      const srcOff = b * blockSize;
      const dstBase = b * blockSize * rep;
      if (dstBase !== srcOff) {
        out.copyWithin(dstBase, srcOff, srcOff + blockSize);
      }
      for (let r = 1; r < rep; r++) {
        out.copyWithin(dstBase + r * blockSize, dstBase, dstBase + blockSize);
      }
    }
    curShape[d] *= rep;
    curTotal *= rep;
  }
  return out;
}

/** Right-pad two arrays with `1`s to a common length. */
function padTo<T extends number | RepAxis>(arr: T[], pad: T, len: number): T[] {
  const out = arr.slice();
  while (out.length < len) out.push(pad);
  return out;
}

/** Trim trailing entries equal to `1` down to a 2-axis floor. Matches
 *  numbl's `tensorDouble` / shape-constructor `normalizeAxes` rule, so
 *  `repmat([1 2 3], 1, 1, 1)` lands as a 2-D `[1 3]` (not 3-D `[1 3 1]`). */
function trimTrailingOnes(shape: number[]): number[] {
  const out = shape.slice();
  while (out.length > 2 && out[out.length - 1] === 1) out.pop();
  while (out.length < 2) out.push(1);
  return out;
}

/** Same trim on a per-axis `DimInfo` lattice: pop trailing axes that
 *  are statically exact-1 down to a 2-axis floor. Dynamic-extent axes
 *  are kept (we can't prove they'll be 1 at runtime). */
function trimTrailingOneDims(dims: DimInfo[]): DimInfo[] {
  const out = dims.slice();
  const isOne = (d: DimInfo): boolean => d.kind === "exact" && d.value === 1;
  while (out.length > 2 && isOne(out[out.length - 1])) out.pop();
  while (out.length < 2) out.push(DIM_ONE);
  return out;
}

/** Compute the reps list and ndim the codegen should pass to the
 *  runtime helper so the produced tensor's shape matches what
 *  `transfer` reported. Mirrors the trailing-1 trim: pop trailing
 *  axes while both the input dim AND the rep are statically exact-1,
 *  down to a 2-axis floor. Right-pads with synthetic exact-1 axes
 *  when the rep list is shorter than the resulting ndim.
 *
 *  When `scalarInput` is true the input shape is `[1, 1, …]` —
 *  any trailing rep of 1 can be trimmed (the input never contributes
 *  a non-1 trailing axis). */
function effectiveCodegenReps(
  inDims: ReadonlyArray<DimInfo>,
  reps: RepAxis[],
  scalarInput: boolean
): { reps: RepAxis[]; ndim: number } {
  const full = Math.max(scalarInput ? 0 : inDims.length, reps.length);
  let n = full < 2 ? 2 : full;
  while (n > 2) {
    const i = n - 1;
    const inIsOne =
      scalarInput ||
      i >= inDims.length ||
      (inDims[i].kind === "exact" && inDims[i].value === 1);
    const rep =
      i < reps.length
        ? reps[i]
        : ({ kind: "exact", value: 1, argIndex: 0 } as RepAxis);
    const repIsOne = rep.kind === "exact" && rep.value === 1;
    if (!(inIsOne && repIsOne)) break;
    n--;
  }
  const out = reps.slice(0, n);
  while (out.length < n) {
    out.push({ kind: "exact", value: 1, argIndex: 0 });
  }
  return { reps: out, ndim: n };
}

/** Per-axis C expression for one rep axis. */
function repC(axis: RepAxis, argsC: string[]): string {
  if (axis.kind === "exact") return `${axis.value}L`;
  return `(long)(${argsC[axis.argIndex]})`;
}

/** Per-axis JS expression for one rep axis. */
function repJs(axis: RepAxis, argsJs: string[]): string {
  if (axis.kind === "exact") return String(axis.value);
  return `Math.round(${argsJs[axis.argIndex]})`;
}

export const repmat: Builtin = {
  name: "repmat",
  transfer(argTypes, nargout) {
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'repmat' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `'repmat' first arg must be numeric (got ${typeToString(a)})`
      );
    }
    if (a.elem !== "double" && a.elem !== "logical") {
      throw new TypeError(
        `'repmat' first arg must be a real double / logical (got ${a.elem})`
      );
    }

    const resolved = resolveReps(argTypes);
    const scalarInput = isScalar(a);
    const reps = scalarInput
      ? effectiveRepsForScalarInput(resolved)
      : resolved.reps;

    // Try statically-known shape derivation.
    const allExact = reps.every(r => r.kind === "exact");
    if (allExact) {
      const repVals = reps.map(r => (r as { value: number }).value);
      if (scalarInput) {
        // Trim trailing exact-1 axes down to a 2-axis floor so a call
        // like `repmat(5, 1, 1, 1)` lands as a 1×1 scalar / 2-D tensor
        // rather than 3-D — matching numbl's tensor normalization.
        const outShape = trimTrailingOnes(repVals.slice());
        const total = shapeNumel(outShape);
        // Scalar input: every cell = the scalar's value.
        if (a.isComplex) {
          const cx = exactComplex(a);
          if (outShape.every(s => s === 1)) {
            if (cx !== undefined) return [scalarComplex(cx)];
            return [scalarComplex()];
          }
          if (cx !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
            const re = new Float64Array(total);
            const im = new Float64Array(total);
            re.fill(cx.re);
            im.fill(cx.im);
            return [tensorComplex(outShape, { re, im })];
          }
          return [tensorComplex(outShape)];
        }
        const scalarExact = exactDouble(a);
        if (outShape.every(s => s === 1)) {
          if (scalarExact !== undefined) {
            return [scalarDouble(signFromNumber(scalarExact), scalarExact)];
          }
          return [scalarDouble(a.sign)];
        }
        if (scalarExact !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const data = new Float64Array(total);
          if (scalarExact !== 0) data.fill(scalarExact);
          return [tensorDouble(outShape, data)];
        }
        const t = tensorDouble(outShape);
        t.sign =
          scalarExact !== undefined ? signFromNumber(scalarExact) : a.sign;
        return [t];
      }
      // Tensor input.
      const inShape = inputShape(a);
      if (inShape !== undefined) {
        const outNdim = Math.max(inShape.length, repVals.length);
        const padShape = padTo(inShape, 1, outNdim);
        const padReps = padTo(repVals, 1, outNdim);
        // Trim trailing exact-1 axes; the same trim applies to padShape
        // and padReps so they keep aligned for the per-axis tile in
        // `tileExact` below.
        const outShape = trimTrailingOnes(
          padShape.map((s, i) => s * padReps[i])
        );
        const newNdim = outShape.length;
        while (padShape.length > newNdim) padShape.pop();
        while (padReps.length > newNdim) padReps.pop();
        while (padShape.length < newNdim) padShape.push(1);
        while (padReps.length < newNdim) padReps.push(1);
        if (a.isComplex) {
          if (outShape.every(s => s === 1)) {
            const cx = exactComplexArray(a);
            if (cx !== undefined && cx.re.length === 1) {
              return [scalarComplex({ re: cx.re[0], im: cx.im[0] })];
            }
            return [scalarComplex()];
          }
          const total = shapeNumel(outShape);
          const cx = exactComplexArray(a);
          if (cx !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
            const tiledRe = tileExact(cx.re, padShape, padReps);
            const tiledIm = tileExact(cx.im, padShape, padReps);
            return [tensorComplex(outShape, { re: tiledRe, im: tiledIm })];
          }
          return [tensorComplex(outShape)];
        }
        // Normalize: mtoc2 tensors are always ≥ 2-D. inShape is
        // already ≥ 2-D and outNdim ≥ inShape.length, so outShape is too.
        if (outShape.every(s => s === 1)) {
          const ex = exactRealArray(a);
          if (ex !== undefined && ex.length === 1) {
            return [scalarDouble(signFromNumber(ex[0]), ex[0])];
          }
          const exD = exactDouble(a);
          if (exD !== undefined) {
            return [scalarDouble(signFromNumber(exD), exD)];
          }
          return [scalarDouble(a.sign)];
        }
        const total = shapeNumel(outShape);
        const srcArr = exactRealArray(a);
        if (srcArr !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const tiled = tileExact(srcArr, padShape, padReps);
          const t = tensorDouble(outShape, tiled);
          t.sign = signFromExactArray(tiled);
          return [t];
        }
        const t = tensorDouble(outShape);
        t.sign = a.sign;
        return [t];
      }
      // Input shape unknown (dynamic). Fall through to per-axis dims.
    }

    // Dynamic-axis path: build DimInfo per axis.
    const inDims = a.dims;
    const outNdim = Math.max(inDims.length, reps.length);
    const dims: DimInfo[] = [];
    for (let i = 0; i < outNdim; i++) {
      const inDim: DimInfo = i < inDims.length ? inDims[i] : DIM_ONE;
      const rep =
        i < reps.length
          ? reps[i]
          : ({ kind: "exact", value: 1, argIndex: 0 } as RepAxis);
      if (inDim.kind === "exact" && rep.kind === "exact") {
        const v = inDim.value * rep.value;
        dims.push(v === 1 ? DIM_ONE : { kind: "exact", value: v });
      } else {
        dims.push({ kind: "unknown" });
      }
    }
    // Same trailing-1 trim on the lattice form — only pops axes we
    // can statically prove are 1.
    const trimmed = trimTrailingOneDims(dims);
    if (a.isComplex) return [tensorComplexFromDims(trimmed)];
    const t = tensorDoubleFromDims(trimmed);
    t.sign = a.sign;
    return [t];
  },

  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    const resolved = resolveReps(argTypes);
    const scalarInput = isScalar(a);
    const reps = scalarInput
      ? effectiveRepsForScalarInput(resolved)
      : resolved.reps;
    const { reps: cgReps, ndim } = effectiveCodegenReps(
      a.dims,
      reps,
      scalarInput
    );

    // After the trailing-1 trim, every effective rep may still be 1.
    // For scalar input that means the output is just the scalar (no
    // tile happens) — match the transfer's scalarDouble result by
    // returning the arg directly instead of invoking `fill_nd`.
    const allRepsOne = cgReps.every(r => r.kind === "exact" && r.value === 1);
    if (scalarInput) {
      if (allRepsOne) return argsC[0];
      useRuntime("mtoc2_tensor_fill_nd");
      const dimList = cgReps.map(r => repC(r, argsC)).join(", ");
      if (a.isComplex) {
        useRuntime("mtoc2_cscalar");
        return `mtoc2_tensor_fill_nd_complex(creal(${argsC[0]}), cimag(${argsC[0]}), ${ndim}, (long[]){${dimList}})`;
      }
      return `mtoc2_tensor_fill_nd((double)(${argsC[0]}), ${ndim}, (long[]){${dimList}})`;
    }

    useRuntime("mtoc2_tensor_repmat");
    const dimList = cgReps.map(r => repC(r, argsC)).join(", ");
    if (a.isComplex) {
      return `mtoc2_tensor_repmat_complex(${argsC[0]}, ${ndim}, (long[]){${dimList}})`;
    }
    return `mtoc2_tensor_repmat(${argsC[0]}, ${ndim}, (long[]){${dimList}})`;
  },

  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    const resolved = resolveReps(argTypes);
    const scalarInput = isScalar(a);
    const reps = scalarInput
      ? effectiveRepsForScalarInput(resolved)
      : resolved.reps;
    const { reps: cgReps, ndim } = effectiveCodegenReps(
      a.dims,
      reps,
      scalarInput
    );

    const allRepsOne = cgReps.every(r => r.kind === "exact" && r.value === 1);
    if (scalarInput) {
      if (allRepsOne) return argsJs[0];
      useRuntime("mtoc2_tensor_fill_nd");
      const dimList = cgReps.map(r => repJs(r, argsJs)).join(", ");
      if (a.isComplex) {
        return `mtoc2_tensor_fill_nd_complex(${argsJs[0]}.re, ${argsJs[0]}.im, ${ndim}, [${dimList}])`;
      }
      return `mtoc2_tensor_fill_nd(${argsJs[0]}, ${ndim}, [${dimList}])`;
    }

    useRuntime("mtoc2_tensor_repmat");
    const dimList = cgReps.map(r => repJs(r, argsJs)).join(", ");
    if (a.isComplex) {
      return `mtoc2_tensor_repmat_complex(${argsJs[0]}, ${ndim}, [${dimList}])`;
    }
    return `mtoc2_tensor_repmat(${argsJs[0]}, ${ndim}, [${dimList}])`;
  },

  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    const resolved = resolveReps(argTypes);
    const scalarInput = isScalar(a);
    const reps = scalarInput
      ? effectiveRepsForScalarInput(resolved)
      : resolved.reps;
    const { reps: cgReps } = effectiveCodegenReps(a.dims, reps, scalarInput);

    const repVals: number[] = cgReps.map(r => {
      if (r.kind === "exact") return r.value;
      const v = args[r.argIndex] as RuntimeValue;
      const n = typeof v === "number" ? v : Number(v as unknown);
      const rounded = Math.round(n);
      return rounded < 0 ? 0 : rounded;
    });

    if (scalarInput) {
      const padReps = repVals.slice();
      while (padReps.length < 2) padReps.push(1);
      if (a.isComplex) {
        const v0 = args[0];
        const cx = isComplexValue(v0)
          ? v0
          : { re: typeof v0 === "number" ? v0 : Number(v0 as unknown), im: 0 };
        return [
          jsFillNdComplex(
            cx.re,
            cx.im,
            padReps.length,
            padReps
          ) as unknown as RuntimeTensor,
        ];
      }
      const scalarVal =
        typeof args[0] === "number"
          ? (args[0] as number)
          : Number(args[0] as unknown);
      return [
        jsFillNd(
          scalarVal,
          padReps.length,
          padReps
        ) as unknown as RuntimeTensor,
      ];
    }

    if (a.isComplex) {
      return [
        jsRepmatComplex(
          args[0] as RuntimeTensor,
          repVals.length,
          repVals
        ) as unknown as RuntimeTensor,
      ];
    }
    return [
      jsRepmat(
        args[0] as RuntimeTensor,
        repVals.length,
        repVals
      ) as unknown as RuntimeTensor,
    ];
  },
};

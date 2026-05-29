/**
 * Shared reducer infrastructure for the `sum / prod / mean / min /
 * max / any / all` family. Each builtin is a thin wrapper that
 * supplies its kernel spec (init, op, finalize, sign rule, empty
 * result) and lets this module drive transfer / codegen / arity.
 *
 * Numbl is the dialect oracle (see `../numbl/src/numbl-core/interpreter/
 * builtins/reductions.ts`). The shape rules here mirror numbl's
 * `shapeAfterReduction` against mtoc2's `DimInfo` lattice
 * (`{exact, value}` / `unknown`) plus concrete `shape` when set.
 *
 * The lattice analysis is sharper than numbl's static type pass in one
 * place worth flagging:
 *   - A `(unknown, 1, ..., 1)` dims signature (e.g. a slice
 *     `M(:, 2:3)` of a known-row-vector `M`) proves the output is
 *     scalar even though the leading dim's runtime length is unknown,
 *     because every later axis is statically `1`.
 *
 * Folding: when every input element is exact (scalar `t.exact: number`
 * or tensor `t.exact: Float64Array`), the transfer computes the
 * compile-time result. We cap the materialized result size at
 * `EXACT_ARRAY_MAX_ELEMENTS`; beyond the cap the runtime path takes
 * over.
 *
 * Out of scope: complex inputs, the `omitnan` / `includenan` flag,
 * runtime (non-exact) integer `dim`, and multi-output
 * `[v, i] = min(x)` — all rejected with span.
 */

import { UnsupportedConstruct, TypeError } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  EXACT_ARRAY_MAX_ELEMENTS,
  isDimOne,
  isNumeric,
  isScalar,
  provablyNonEmpty,
  scalarComplex,
  scalarDouble,
  scalarLogical,
  signFromNumber,
  tensorComplex,
  tensorComplexFromDims,
  tensorDouble,
  tensorDoubleFromDims,
  unifySign,
  type DimInfo,
  type NumericType,
  type Sign,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble, exactComplex, exactRealArray } from "../_shared.js";

/** What kind of reducer we're computing. Drives the kernel pieces
 *  (init / accumulator / finalizer / empty fallback / sign rule /
 *  result element type) without each builtin file rewriting the
 *  reduce loop. */
export interface KernelSpec {
  /** Public source-level name (e.g. `"sum"`). */
  name: string;
  /** Position of the dim slot in the source-level arg list. For
   *  `sum/prod/mean/any/all`: slot 2 (`name(A, dim)`).
   *  For `min/max`: slot 3 (`name(A, [], dim)`). */
  dimArgIndex: 1 | 2;
  /** Identity / empty result. For scalar reducers (`sum`, `prod`,
   *  `mean`), the empty-data scalar (`0`, `1`, `NaN`). For
   *  logical reducers (`any`, `all`), 0 or 1. For NaN-seed
   *  reducers (`min`, `max`), `NaN`. */
  emptyValue: number;
  /** Accumulator step. NaN-aware (the impl skips NaN where
   *  appropriate). Receives the running accumulator and the next
   *  element; returns the new accumulator. */
  step(acc: number, x: number, count: number): number;
  /** Optional finalizer (e.g. mean divides by count). */
  finalize?(acc: number, count: number): number;
  /** Output element kind. `"double"` for sum/prod/mean/min/max,
   *  `"logical"` for any/all. */
  outputElem: "double" | "logical";
  /** Compute the sign of the result given the *input* type plus a
   *  flag indicating whether the reduced fiber is provably
   *  non-empty. */
  resultSign(t: NumericType, nonEmpty: boolean): Sign;
}

/** True iff `t` is the static `[]` empty-bracket literal (shape
 *  `[0, 0]`). Used by `min` / `max`'s 3-arg form to validate that the
 *  middle slot is the bracketed empty placeholder rather than some
 *  other zero-numel tensor. */
function isEmptyBracketLiteral(t: Type): boolean {
  return (
    isNumeric(t) &&
    t.elem === "double" &&
    !t.isComplex &&
    t.shape !== undefined &&
    t.shape.length === 2 &&
    t.shape[0] === 0 &&
    t.shape[1] === 0
  );
}

// ── Axis classification ────────────────────────────────────────────────

interface AxisAll {
  kind: "all";
}
interface AxisFixed {
  kind: "fixed";
  /** 1-based axis number, statically known. */
  dim: number;
}
type AxisChoice = AxisAll | AxisFixed | { kind: "default" };

/** Inspect the parsed dim slot. Throws on rejection cases (non-
 *  exact, non-positive, non-integer, wrong type). Returns `default`
 *  if no dim slot was supplied. */
function classifyDimArg(name: string, dimType: Type | undefined): AxisChoice {
  if (dimType === undefined) return { kind: "default" };
  if (dimType.kind === "String" || dimType.kind === "Char") {
    const v = dimType.exact;
    if (v === "all") return { kind: "all" };
    if (v === undefined) {
      throw new UnsupportedConstruct(
        `'${name}' dim arg: opaque text (only the literal 'all' is supported)`
      );
    }
    throw new UnsupportedConstruct(
      `'${name}' dim arg: text literal must be 'all' (got '${v}')`
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
      `'${name}' dim arg must be a statically-known integer in v1 ` +
        `(runtime dim values can't be deduced into a result shape)`
    );
  }
  if (!Number.isFinite(v) || !Number.isInteger(v) || v < 1) {
    throw new TypeError(
      `'${name}' dim arg must be a finite positive integer (got ${v})`
    );
  }
  return { kind: "fixed", dim: v };
}

/** When no explicit dim was supplied, pick the reduction axis from
 *  the input's lattice. This is the sharp pass: a leading `unknown`
 *  followed only by `one` axes still proves the output is scalar
 *  (the leading axis MUST be the non-singleton one, or the whole
 *  tensor is `(1×1×…)`-shaped → scalar collapse anyway).
 *
 *  Returns `all` for the degenerate `(1, 1, ...)` case (matches
 *  scalar input), `fixed(k)` for a clearly chosen axis, or throws
 *  for the genuinely ambiguous case (`unknown` leading dim followed
 *  by at least one known-non-1 or another `unknown`). */
function chooseDefaultAxis(name: string, t: NumericType): AxisAll | AxisFixed {
  // Concrete shape: numbl's `firstReduceDim` rule
  // (helpers/reduction-helpers.ts) — count dims > 1; collapse to
  // scalar when at most one dim is > 1 (matches `sum([1 2 3]) = 6`
  // AND `sum([]) = 0` / `sum(zeros(0, 5)) = 0`); else reduce along
  // the first dim > 1. Using `!== 1` here (the prior rule) would
  // route empty shapes like `[0, 0]` and `[0, 5]` to a per-axis
  // reduce and produce a non-scalar empty tensor — numbl produces
  // the scalar identity instead.
  if (t.shape !== undefined) {
    const numGt1 = t.shape.filter(s => s > 1).length;
    if (numGt1 <= 1) return { kind: "all" };
    return { kind: "fixed", dim: t.shape.findIndex(s => s > 1) + 1 };
  }
  // Lattice walk. Same rule as the concrete-shape path but
  // articulated dim-by-dim:
  //   - exact == 0 or 1: doesn't count as a non-singleton.
  //   - exact > 1: counts. First one is the axis-fixed candidate.
  //   - unknown: ambiguous. If every later dim is `1` we still know
  //     the result is scalar (any runtime length of one axis reduces
  //     to a scalar); otherwise we can't decide statically.
  let knownGt1 = 0;
  let firstGt1Idx = -1;
  for (let i = 0; i < t.dims.length; i++) {
    const d = t.dims[i];
    if (d.kind === "exact") {
      if (d.value > 1) {
        knownGt1++;
        if (firstGt1Idx === -1) firstGt1Idx = i;
      }
      continue;
    }
    // unknown
    if (t.dims.slice(i + 1).every(isDimOne) && knownGt1 === 0) {
      return { kind: "all" };
    }
    throw new UnsupportedConstruct(
      `'${name}' on a tensor with ambiguous lattice ` +
        `(${t.dims.map(d => (d.kind === "exact" ? String(d.value) : "?")).join("×")}): can't deduce the ` +
        `reduction axis — pass an explicit dim (e.g. ` +
        `${name}(A, 1)) or 'all'`
    );
  }
  if (knownGt1 <= 1) return { kind: "all" };
  return { kind: "fixed", dim: firstGt1Idx + 1 };
}

/** Squeeze trailing singleton dims subject to a 2-axis floor (matches
 *  numbl's `shapeAfterReduction` post-pass and the tensor-shape
 *  normalization rule). */
function squeezeTrailing<T>(arr: T[], isOne: (x: T) => boolean): T[] {
  const out = arr.slice();
  while (out.length > 2 && isOne(out[out.length - 1])) out.pop();
  return out;
}

/** Mirror of numbl's `shapeAfterReduction(shape, dim)` on a concrete
 *  shape. Returns either `{ scalar: true }` (every axis is 1 after
 *  the collapse) or `{ scalar: false, shape: [...] }`. */
function reduceConcreteShape(
  shape: number[],
  dim: number
): { scalar: true } | { scalar: false; shape: number[] } {
  if (dim > shape.length) {
    if (shape.every(d => d === 1)) return { scalar: true };
    return { scalar: false, shape: shape.slice() };
  }
  const out = shape.slice();
  out[dim - 1] = 1;
  const squeezed = squeezeTrailing(out, s => s === 1);
  if (squeezed.every(d => d === 1)) return { scalar: true };
  return { scalar: false, shape: squeezed };
}

/** Same as `reduceConcreteShape` but for the lattice form. Result
 *  is either `scalar` (every dim collapses to `one`) or a new dims
 *  array, post-squeeze. */
function reduceLatticeDims(
  dims: DimInfo[],
  dim: number
): { scalar: true } | { scalar: false; dims: DimInfo[] } {
  if (dim > dims.length) {
    if (dims.every(isDimOne)) return { scalar: true };
    return { scalar: false, dims: dims.slice() };
  }
  const out = dims.slice();
  out[dim - 1] = DIM_ONE;
  const squeezed = squeezeTrailing(out, isDimOne);
  if (squeezed.every(isDimOne)) return { scalar: true };
  return { scalar: false, dims: squeezed };
}

// ── Folding ───────────────────────────────────────────────────────────

/** Compile-time evaluation of the reducer when every input element
 *  is exact. Returns `undefined` to fall back to runtime emission
 *  (e.g. unsupported axis layouts, oversized results, no input
 *  exact). */
function foldExact(
  spec: KernelSpec,
  data: Float64Array,
  shape: number[],
  axis: AxisAll | AxisFixed
):
  | { kind: "scalar"; value: number }
  | { kind: "tensor"; shape: number[]; data: Float64Array }
  | undefined {
  if (axis.kind === "all") {
    let acc = spec.emptyValue;
    const n = data.length;
    if (spec.outputElem === "logical") {
      // Short-circuit logical reductions; matches the runtime.
      if (spec.name === "any") {
        for (let i = 0; i < n; i++) {
          if (data[i] !== 0) {
            acc = 1;
            break;
          }
        }
        return { kind: "scalar", value: acc };
      }
      if (spec.name === "all") {
        acc = 1;
        for (let i = 0; i < n; i++) {
          if (data[i] === 0) {
            acc = 0;
            break;
          }
        }
        return { kind: "scalar", value: acc };
      }
    }
    if (spec.name === "min" || spec.name === "max") {
      // NaN-seed with NaN-skip.
      acc = NaN;
      for (let i = 0; i < n; i++) {
        const x = data[i];
        if (Number.isNaN(x)) continue;
        if (Number.isNaN(acc) || (spec.name === "min" ? x < acc : x > acc)) {
          acc = x;
        }
      }
      return { kind: "scalar", value: acc };
    }
    // Accumulator-based: sum / prod / mean.
    for (let i = 0; i < n; i++) {
      acc = spec.step(acc, data[i], i + 1);
    }
    const finalized = spec.finalize ? spec.finalize(acc, n) : acc;
    return { kind: "scalar", value: finalized };
  }

  // AxisFixed: reduce along one axis. Output shape comes from
  // `reduceConcreteShape`; the layout below mirrors numbl's
  // forEachSlice with column-major fiber stride.
  const dim = axis.dim;
  const result = reduceConcreteShape(shape, dim);
  if (result.scalar) {
    // Fall back to the AxisAll path (every axis collapses to 1).
    return foldExact(spec, data, shape, { kind: "all" });
  }
  // Compute the per-fiber reduction.
  const dimIdx = dim - 1;
  // dim > ndim ⇒ output shape == input shape (copy). We forward the
  // data verbatim; matches the runtime helper's no-op axis branch.
  if (dim > shape.length) {
    if (spec.outputElem === "logical") {
      // any/all dim > ndim is elementwise cast to logical.
      const cast = new Float64Array(data.length);
      for (let i = 0; i < data.length; i++) cast[i] = data[i] !== 0 ? 1 : 0;
      return { kind: "tensor", shape: result.shape, data: cast };
    }
    return {
      kind: "tensor",
      shape: result.shape,
      data: new Float64Array(data),
    };
  }
  const axisLen = shape[dimIdx];
  let before = 1;
  for (let i = 0; i < dimIdx; i++) before *= shape[i];
  let after = 1;
  for (let i = dimIdx + 1; i < shape.length; i++) after *= shape[i];
  const outLen = before * after;
  if (outLen > EXACT_ARRAY_MAX_ELEMENTS) return undefined;
  const out = new Float64Array(outLen);
  const slab = before * axisLen;
  for (let outer = 0; outer < after; outer++) {
    const slabBase = outer * slab;
    for (let inner = 0; inner < before; inner++) {
      let acc =
        spec.name === "min" || spec.name === "max" ? NaN : spec.emptyValue;
      let shortCircuited = false;
      for (let k = 0; k < axisLen; k++) {
        const x = data[slabBase + inner + k * before];
        if (spec.name === "min" || spec.name === "max") {
          if (Number.isNaN(x)) continue;
          if (Number.isNaN(acc) || (spec.name === "min" ? x < acc : x > acc)) {
            acc = x;
          }
        } else if (spec.outputElem === "logical") {
          if (spec.name === "any") {
            if (x !== 0) {
              acc = 1;
              shortCircuited = true;
              break;
            }
          } else {
            // all
            if (x === 0) {
              acc = 0;
              shortCircuited = true;
              break;
            }
          }
        } else {
          acc = spec.step(acc, x, k + 1);
        }
      }
      if (
        !shortCircuited &&
        spec.outputElem === "double" &&
        spec.name !== "min" &&
        spec.name !== "max" &&
        spec.finalize
      ) {
        acc = spec.finalize(acc, axisLen);
      }
      out[outer * before + inner] = acc;
    }
  }
  return { kind: "tensor", shape: result.shape, data: out };
}

// ── Result type construction ──────────────────────────────────────────

/** Stamp a tensor result with the spec's output-elem kind. Logical
 *  reducers (`any`/`all`) flip `elem` and pin the sign to `nonneg`
 *  (range is `{0, 1}`); double reducers keep the caller's sign rule. */
function stampOutputElem(out: NumericType, spec: KernelSpec, sign: Sign): void {
  if (spec.outputElem === "logical") {
    out.elem = "logical";
    out.sign = "nonneg";
  } else {
    out.sign = sign;
  }
}

/** Build a scalar result type. `exactScalar` is the folded value when
 *  every input was exact; pass `undefined` for the runtime path. */
function scalarResult(
  spec: KernelSpec,
  exactScalar: number | undefined,
  sign: Sign
): Type {
  if (spec.outputElem === "logical") {
    return exactScalar !== undefined
      ? scalarLogical(exactScalar !== 0)
      : scalarLogical();
  }
  if (exactScalar !== undefined) {
    return scalarDouble(
      unifySign(sign, signFromNumber(exactScalar)),
      exactScalar
    );
  }
  return scalarDouble(sign);
}

/** Build a tensor result with a concrete shape. `exactArray` is the
 *  folded data when every input was exact AND the result fit the cap;
 *  pass `undefined` for the runtime path. */
function shapedTensorResult(
  spec: KernelSpec,
  shape: number[],
  exactArray: Float64Array | undefined,
  sign: Sign
): Type {
  const out: NumericType = exactArray
    ? tensorDouble(shape, exactArray)
    : tensorDouble(shape);
  stampOutputElem(out, spec, sign);
  return out;
}

/** Build a tensor result with lattice-only dims (no concrete shape). */
function latticeTensorResult(
  spec: KernelSpec,
  dims: DimInfo[],
  sign: Sign
): Type {
  const out = tensorDoubleFromDims(dims);
  stampOutputElem(out, spec, sign);
  return out;
}

// ── Public entry points ────────────────────────────────────────────────

/** The transfer function. Returns the result type; throws with no
 *  span (framework's `withSpan` backfills at the call site). */
export function reductionTransfer(argTypes: Type[], spec: KernelSpec): Type {
  const inputType = argTypes[0];
  if (!isNumeric(inputType)) {
    throw new TypeError(
      `'${spec.name}' arg must be numeric (got ${inputType.kind})`
    );
  }
  if (inputType.elem !== "double" && inputType.elem !== "logical") {
    throw new TypeError(
      `'${spec.name}' arg must be double or logical (got ${inputType.elem})`
    );
  }
  if (inputType.isComplex) {
    return complexReductionTransfer(argTypes, spec);
  }
  if (spec.dimArgIndex === 2 && argTypes.length === 2) {
    return elementwiseMinMaxTransfer(argTypes, spec.name as "min" | "max");
  }
  if (spec.dimArgIndex === 2 && argTypes.length === 3) {
    if (!isEmptyBracketLiteral(argTypes[1])) {
      throw new UnsupportedConstruct(
        `'${spec.name}(A, [], dim)' requires the second arg to be the ` +
          `empty literal '[]' (got something else)`
      );
    }
  }
  const dimType = argTypes[spec.dimArgIndex];
  const axis = classifyDimArg(spec.name, dimType);

  // Scalar input: identity (or logical cast for any/all).
  if (isScalar(inputType)) {
    if (spec.outputElem === "logical") {
      const xv = exactDouble(inputType);
      if (xv !== undefined) {
        return scalarLogical(xv !== 0);
      }
      return scalarLogical();
    }
    const xv = exactDouble(inputType);
    if (xv !== undefined) {
      return scalarDouble(signFromNumber(xv), xv);
    }
    return scalarDouble(inputType.sign);
  }

  // Pick the reduction axis.
  let resolved: AxisAll | AxisFixed;
  if (axis.kind === "default") {
    resolved = chooseDefaultAxis(spec.name, inputType);
  } else {
    resolved = axis;
  }

  // Default-axis min/max on statically empty input: numbl returns
  // `[]` (a 0×0 empty), but the c-aot runtime helper
  // `mtoc2_<min|max>_all` returns a `double` — emitting it through
  // the normal tensor-Call path would type-mismatch at `cc`. Reject
  // until a tensor-returning variant of the helper lands; users can
  // pass an explicit dim arg as a workaround. sum/prod/mean/any/all
  // collapse to the scalar identity via the regular fold path so
  // they aren't affected.
  if (
    axis.kind === "default" &&
    inputType.shape !== undefined &&
    inputType.shape.some(d => d === 0) &&
    (spec.name === "min" || spec.name === "max")
  ) {
    throw new UnsupportedConstruct(
      `'${spec.name}' on a statically empty input with default axis is not ` +
        `yet supported (numbl returns []); pass an explicit dim arg or ` +
        `non-empty input`
    );
  }

  const nonEmpty = provablyNonEmpty(inputType);
  const sign = spec.resultSign(inputType, nonEmpty);

  // Fully-exact fold path.
  const exactArr = exactRealArray(inputType);
  if (exactArr !== undefined && inputType.shape !== undefined) {
    const folded = foldExact(spec, exactArr, inputType.shape, resolved);
    if (folded !== undefined) {
      if (folded.kind === "scalar") {
        return scalarResult(spec, folded.value, sign);
      }
      const exactData =
        folded.data.length <= EXACT_ARRAY_MAX_ELEMENTS
          ? folded.data
          : undefined;
      return shapedTensorResult(spec, folded.shape, exactData, sign);
    }
  }

  // Non-exact path: compute the result shape on lattice + concrete-shape
  // form so downstream codegen has the precision to dispatch tensor
  // vs. scalar.
  if (resolved.kind === "all") {
    return scalarResult(spec, undefined, sign);
  }
  // AxisFixed
  if (inputType.shape !== undefined) {
    const r = reduceConcreteShape(inputType.shape, resolved.dim);
    if (r.scalar) return scalarResult(spec, undefined, sign);
    return shapedTensorResult(spec, r.shape, undefined, sign);
  }
  // Lattice-only.
  const r = reduceLatticeDims(inputType.dims, resolved.dim);
  if (r.scalar) return scalarResult(spec, undefined, sign);
  return latticeTensorResult(spec, r.dims, sign);
}

// ── Codegen ────────────────────────────────────────────────────────────

/** Build the codegenC closure for a reducer. The kernel name powers
 *  the emitted helper call (`mtoc2_<kernel>_all` / `_dim`). */
// ── Elementwise 2-arg min/max (not a reduction) ────────────────────────

/** `max(a, b)` / `min(a, b)` scalar-scalar transfer. Result is a scalar
 *  real double; folds when both inputs are exact (NaN-aware, matching
 *  C99 `fmax`/`fmin` — NaN is treated as missing data, so the non-NaN
 *  operand wins). Tensor and broadcast forms are deferred. */
function elementwiseMinMaxTransfer(
  argTypes: Type[],
  name: "min" | "max"
): Type {
  const [a, b] = argTypes;
  if (!isNumeric(a) || a.isComplex || !isNumeric(b) || b.isComplex) {
    throw new TypeError(`'${name}(a, b)' args must be real numeric`);
  }
  if (a.elem !== "double" && a.elem !== "logical") {
    throw new TypeError(
      `'${name}(a, b)' arg 1 must be double or logical (got ${a.elem})`
    );
  }
  if (b.elem !== "double" && b.elem !== "logical") {
    throw new TypeError(
      `'${name}(a, b)' arg 2 must be double or logical (got ${b.elem})`
    );
  }
  if (!isScalar(a) || !isScalar(b)) {
    throw new UnsupportedConstruct(
      `'${name}(a, b)' tensor / broadcast form is not yet supported — ` +
        `both args must be scalar`
    );
  }
  const av = exactDouble(a);
  const bv = exactDouble(b);
  if (av !== undefined && bv !== undefined) {
    // NaN-aware: if exactly one is NaN, the non-NaN one wins.
    let v: number;
    if (Number.isNaN(av)) {
      v = bv;
    } else if (Number.isNaN(bv)) {
      v = av;
    } else {
      v = name === "max" ? Math.max(av, bv) : Math.min(av, bv);
    }
    return scalarDouble(signFromNumber(v), v);
  }
  return scalarDouble(elementwiseMinMaxSign(name, a.sign, b.sign));
}

/** Reduction transfer for complex input. Pairs each `KernelSpec` with
 *  the right result-shape rule and the right output element type:
 *  sum/prod/mean/min/max → complex; any/all → real logical.
 *
 *  Folding is intentionally skipped on the complex path; the runtime
 *  helper does the work. The plan's `EXACT_ARRAY_MAX_ELEMENTS` cap
 *  on the `{re, im}` carrier still applies to other surfaces (literal
 *  build, elementwise arith), but reducing a small complex array to
 *  a scalar at translate time isn't load-bearing for the cross-runner. */
function complexReductionTransfer(argTypes: Type[], spec: KernelSpec): Type {
  const inputType = argTypes[0] as NumericType;
  if (spec.dimArgIndex === 2 && argTypes.length === 3) {
    if (!isEmptyBracketLiteral(argTypes[1])) {
      throw new UnsupportedConstruct(
        `'${spec.name}(A, [], dim)' requires the second arg to be the ` +
          `empty literal '[]' (got something else)`
      );
    }
  }
  const dimType = argTypes[spec.dimArgIndex];
  const axis = classifyDimArg(spec.name, dimType);

  const isLogical = spec.outputElem === "logical";

  // Scalar input: identity (or logical cast via complex toBool).
  if (isScalar(inputType)) {
    if (isLogical) {
      const cx = exactComplex(inputType);
      if (cx !== undefined) return scalarLogical(cx.re !== 0 || cx.im !== 0);
      return scalarLogical();
    }
    const cx = exactComplex(inputType);
    if (cx !== undefined) return scalarComplex(cx);
    return scalarComplex();
  }

  // Pick the reduction axis.
  let resolved: AxisAll | AxisFixed;
  if (axis.kind === "default") {
    resolved = chooseDefaultAxis(spec.name, inputType);
  } else {
    resolved = axis;
  }

  // Compute the output shape; mirror the real path's shape rules but
  // produce a complex tensor (or scalar) for the numeric reducers and
  // a real tensor (or scalar) for the logical reducers.
  if (resolved.kind === "all") {
    if (isLogical) return scalarLogical();
    return scalarComplex();
  }
  if (inputType.shape !== undefined) {
    const r = reduceConcreteShape(inputType.shape, resolved.dim);
    if (r.scalar) {
      if (isLogical) return scalarLogical();
      return scalarComplex();
    }
    if (isLogical) {
      const out = tensorDouble(r.shape);
      out.elem = "logical";
      out.sign = "nonneg";
      return out;
    }
    return tensorComplex(r.shape);
  }
  // Lattice-only.
  const r = reduceLatticeDims(inputType.dims, resolved.dim);
  if (r.scalar) {
    if (isLogical) return scalarLogical();
    return scalarComplex();
  }
  if (isLogical) {
    const out = tensorDoubleFromDims(r.dims);
    out.elem = "logical";
    out.sign = "nonneg";
    return out;
  }
  return tensorComplexFromDims(r.dims);
}

/** Sign rule for `max(a, b)` / `min(a, b)` on two scalars. */
function elementwiseMinMaxSign(name: "min" | "max", sa: Sign, sb: Sign): Sign {
  if (name === "max") {
    // max(a, b) >= a and >= b, so any "≥ 0" constraint on either
    // operand carries through.
    if (sa === "positive" || sb === "positive") return "positive";
    if (sa === "nonneg" || sb === "nonneg") return "nonneg";
    if (sa === "zero" && sb === "zero") return "zero";
    if (
      (sa === "zero" || sa === "negative" || sa === "nonpositive") &&
      (sb === "zero" || sb === "negative" || sb === "nonpositive")
    ) {
      // Both operands are ≤ 0; max is also ≤ 0.
      // If either is exactly zero we already know max ≥ 0, so combined
      // with ≤ 0 we'd get zero — but the caller doesn't reach here
      // because the positive/nonneg cases short-circuited above.
      if (sa === "negative" && sb === "negative") return "negative";
      return "nonpositive";
    }
    return "unknown";
  }
  // min: symmetric. min(a, b) <= a and <= b.
  if (sa === "negative" || sb === "negative") return "negative";
  if (sa === "nonpositive" || sb === "nonpositive") return "nonpositive";
  if (sa === "zero" && sb === "zero") return "zero";
  if (
    (sa === "zero" || sa === "positive" || sa === "nonneg") &&
    (sb === "zero" || sb === "positive" || sb === "nonneg")
  ) {
    if (sa === "positive" && sb === "positive") return "positive";
    return "nonneg";
  }
  return "unknown";
}

// JS-side tensor reducers: keyed by builtin name. Each entry holds
// the `_all` (scalar return) and `_dim` (tensor return) kernels.
// Names match `mtoc2_<name>_all` / `mtoc2_<name>_dim` on the C side
// so the emitJs path and `call` path stay structurally aligned.
import * as TENSOR_REDUCE from "../../runtime/tensor_ops/tensor_reduce_real.js";
import * as TENSOR_REDUCE_C from "../../runtime/tensor_ops/tensor_reduce_complex.js";
import type { RuntimeTensor, RuntimeValue } from "../../../runtime/value.js";

type ReduceAll = (t: RuntimeTensor) => number;
type ReduceDim = (t: RuntimeTensor, d: number) => RuntimeTensor;

const TENSOR_REDUCERS: Record<string, { all: ReduceAll; dim: ReduceDim }> = {
  sum: {
    all: TENSOR_REDUCE.mtoc2_sum_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_sum_dim as unknown as ReduceDim,
  },
  prod: {
    all: TENSOR_REDUCE.mtoc2_prod_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_prod_dim as unknown as ReduceDim,
  },
  mean: {
    all: TENSOR_REDUCE.mtoc2_mean_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_mean_dim as unknown as ReduceDim,
  },
  min: {
    all: TENSOR_REDUCE.mtoc2_min_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_min_dim as unknown as ReduceDim,
  },
  max: {
    all: TENSOR_REDUCE.mtoc2_max_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_max_dim as unknown as ReduceDim,
  },
  any: {
    all: TENSOR_REDUCE.mtoc2_any_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_any_dim as unknown as ReduceDim,
  },
  all: {
    all: TENSOR_REDUCE.mtoc2_all_all as unknown as ReduceAll,
    dim: TENSOR_REDUCE.mtoc2_all_dim as unknown as ReduceDim,
  },
};

// Complex-tensor reducers: same per-name shape as the real table.
// `_all` returns `{re, im}` for sum/prod/mean/min/max, a plain number
// for any/all (logical reducers always produce real). `_dim` returns
// a complex tensor for sum/prod/mean/min/max, a real tensor for
// any/all.
type ReduceAllComplex = (
  t: RuntimeTensor
) => { re: number; im: number } | number;
const COMPLEX_TENSOR_REDUCERS: Record<
  string,
  { all: ReduceAllComplex; dim: ReduceDim }
> = {
  sum: {
    all: TENSOR_REDUCE_C.mtoc2_sum_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_sum_complex_dim as unknown as ReduceDim,
  },
  prod: {
    all: TENSOR_REDUCE_C.mtoc2_prod_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_prod_complex_dim as unknown as ReduceDim,
  },
  mean: {
    all: TENSOR_REDUCE_C.mtoc2_mean_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_mean_complex_dim as unknown as ReduceDim,
  },
  min: {
    all: TENSOR_REDUCE_C.mtoc2_min_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_min_complex_dim as unknown as ReduceDim,
  },
  max: {
    all: TENSOR_REDUCE_C.mtoc2_max_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_max_complex_dim as unknown as ReduceDim,
  },
  any: {
    all: TENSOR_REDUCE_C.mtoc2_any_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_any_complex_dim as unknown as ReduceDim,
  },
  all: {
    all: TENSOR_REDUCE_C.mtoc2_all_complex_all as unknown as ReduceAllComplex,
    dim: TENSOR_REDUCE_C.mtoc2_all_complex_dim as unknown as ReduceDim,
  },
};

/** Resolve the dim arg into a concrete axis verdict for the call
 *  hook. Mirrors `reductionEmit`'s branching but reads runtime
 *  values rather than `argTypes[..].exact`. */
function resolveCallAxis(
  spec: { name: string; dimArgIndex: 1 | 2 },
  args: ReadonlyArray<unknown>,
  argTypes: ReadonlyArray<Type>,
  inputT: NumericType
): { kind: "all" } | { kind: "fixed"; dim: number } {
  const dimArg = args[spec.dimArgIndex];
  const dimType = argTypes[spec.dimArgIndex];
  if (dimArg === undefined) {
    return chooseDefaultAxis(spec.name, inputT);
  }
  if (
    dimType !== undefined &&
    (dimType.kind === "String" || dimType.kind === "Char") &&
    dimType.exact === "all"
  ) {
    return { kind: "all" };
  }
  if (typeof dimArg === "string" && dimArg === "all") {
    return { kind: "all" };
  }
  const n =
    typeof dimArg === "number"
      ? dimArg
      : Number(dimArg as { toString: () => string });
  return { kind: "fixed", dim: Math.trunc(n) };
}

export function reductionEmitJs(spec: {
  name: string;
  dimArgIndex: 1 | 2;
  outputElem: "double" | "logical";
}): Builtin["emitJs"] {
  return ({ argsJs, argTypes, useRuntime }) => {
    if (spec.dimArgIndex === 2 && argTypes.length === 2) {
      // MATLAB / numbl 2-arg min/max: NaN compares as "missing", so
      // `min(NaN, 4)` returns 4 (the non-NaN). JS `Math.min(NaN, 4)`
      // is NaN. Use an explicit NaN-skip wrapper instead.
      const fn = spec.name === "max" ? "Math.max" : "Math.min";
      const a = argsJs[0];
      const b = argsJs[1];
      return `(((_a, _b) => isNaN(_a) ? _b : isNaN(_b) ? _a : ${fn}(_a, _b))(${a}, ${b}))`;
    }
    const inputT = argTypes[0];
    if (!isNumeric(inputT)) {
      throw new Error(`internal: ${spec.name} emitJs got non-numeric arg`);
    }
    const isComplex = inputT.isComplex;
    if (isScalar(inputT)) {
      if (spec.outputElem === "logical") {
        if (isComplex) {
          useRuntime("mtoc2_cscalar");
          return `mtoc2_cnonzero(${argsJs[0]})`;
        }
        // Coerce booleans through `Number` before the `!== 0` check.
        // A bare `${x} !== 0` would mis-classify a JS boolean argument:
        // `false !== 0` is `true` in JS strict-equality (boolean !==
        // number) — incorrect for `all(false)`. `Number(NaN) !== 0`
        // is `true`, matching MATLAB's `all(NaN) == 1` semantics
        // (anything non-zero counts as truthy).
        return `(Number(${argsJs[0]}) !== 0)`;
      }
      return argsJs[0];
    }
    // Tensor path. Mirror `reductionEmit`'s axis resolution exactly
    // so the emitted JS call shape matches the C side per case.
    if (isComplex) {
      useRuntime("mtoc2_tensor_reduce_complex");
      useRuntime("mtoc2_cscalar");
    } else {
      useRuntime("mtoc2_tensor_reduce_real");
    }
    const suffixAll = isComplex ? "_complex_all" : "_all";
    const suffixDim = isComplex ? "_complex_dim" : "_dim";
    const dimType: Type | undefined = argTypes[spec.dimArgIndex];
    let axis: { kind: "all" } | { kind: "fixed"; dim: number };
    if (dimType === undefined) {
      axis = chooseDefaultAxis(spec.name, inputT);
    } else if (
      (dimType.kind === "String" || dimType.kind === "Char") &&
      dimType.exact === "all"
    ) {
      axis = { kind: "all" };
    } else if (isNumeric(dimType)) {
      const v = exactDouble(dimType);
      if (v === undefined) {
        throw new UnsupportedConstruct(
          `'${spec.name}' tensor emitJs requires a statically-known dim ` +
            `(non-exact dim args aren't wired yet)`
        );
      }
      axis = { kind: "fixed", dim: v };
    } else {
      throw new UnsupportedConstruct(
        `'${spec.name}' tensor emitJs got an unexpected dim type ` +
          `'${dimType.kind}'`
      );
    }
    // Scalar-output (axis=all OR axis collapses to scalar) returns
    // a number 0/1 from the runtime helper; wrap to a JS bool when
    // the static output type is logical so the spec result round-
    // trips through `mtoc2ToNumbl` as a `boolean`. Tensor-output
    // (axis dim returning a 1×N etc.) keeps Float64Array storage
    // with an isLogical flag.
    const wrapLogical = (callExpr: string): string =>
      spec.outputElem === "logical" ? `(${callExpr} !== 0)` : callExpr;
    if (axis.kind === "all") {
      return wrapLogical(`mtoc2_${spec.name}${suffixAll}(${argsJs[0]})`);
    }
    // Mirror the C path's scalar-collapse check: if reducing on this
    // axis squeezes the shape down to a scalar, route to `_all`
    // (returning a number) instead of `_dim` (returning a 1×1 tensor).
    if (inputT.shape !== undefined) {
      const r = reduceConcreteShape(inputT.shape, axis.dim);
      if (r.scalar)
        return wrapLogical(`mtoc2_${spec.name}${suffixAll}(${argsJs[0]})`);
    } else {
      const r = reduceLatticeDims(inputT.dims, axis.dim);
      if (r.scalar)
        return wrapLogical(`mtoc2_${spec.name}${suffixAll}(${argsJs[0]})`);
    }
    return `mtoc2_${spec.name}${suffixDim}(${argsJs[0]}, ${axis.dim})`;
  };
}

export function reductionCall(spec: {
  name: string;
  dimArgIndex: 1 | 2;
  outputElem: "double" | "logical";
}): Builtin["call"] {
  return ({ args, argTypes }) => {
    if (spec.dimArgIndex === 2 && argTypes.length === 2) {
      const av = typeof args[0] === "number" ? args[0] : Number(args[0]);
      const bv = typeof args[1] === "number" ? args[1] : Number(args[1]);
      // NaN-skip: MATLAB / numbl returns the non-NaN side when one
      // operand is NaN. JS `Math.min/max` propagates NaN.
      if (Number.isNaN(av)) return [bv];
      if (Number.isNaN(bv)) return [av];
      return [spec.name === "max" ? Math.max(av, bv) : Math.min(av, bv)];
    }
    const inputT = argTypes[0];
    if (!isNumeric(inputT)) {
      throw new Error(`internal: ${spec.name} call got non-numeric arg`);
    }
    const isComplex = inputT.isComplex;
    if (isScalar(inputT)) {
      if (isComplex) {
        const v = args[0];
        const cx =
          typeof v === "object" && v !== null && "re" in v
            ? (v as { re: number; im: number })
            : { re: typeof v === "number" ? v : Number(v), im: 0 };
        if (spec.outputElem === "logical") {
          return [cx.re !== 0 || cx.im !== 0];
        }
        return [cx];
      }
      // Scalar-typed input may arrive as a plain number, a boolean,
      // or a 1×1 RuntimeTensor (the interpreter doesn't strip the
      // tensor wrapper just because the static type is scalar — a
      // value like `c = b(2:end)` keeps its 1×1 tensor shape at
      // runtime). Unwrap each shape carefully; `Number(tensor)`
      // would yield NaN.
      const raw = args[0];
      let v: number;
      if (typeof raw === "number") v = raw;
      else if (typeof raw === "boolean") v = raw ? 1 : 0;
      else if (
        raw !== null &&
        typeof raw === "object" &&
        (raw as { mtoc2Tag?: string }).mtoc2Tag === "tensor"
      ) {
        v = (raw as { data: { [k: number]: number } }).data[0];
      } else {
        v = Number(raw);
      }
      if (spec.outputElem === "logical") return [v !== 0];
      return [v];
    }
    // Tensor input — dispatch through the JS reduce table.
    const reducer = isComplex
      ? COMPLEX_TENSOR_REDUCERS[spec.name]
      : TENSOR_REDUCERS[spec.name];
    if (!reducer) {
      throw new UnsupportedConstruct(
        `'${spec.name}' tensor 'call' has no JS reducer registered`
      );
    }
    const t = args[0] as RuntimeTensor;
    const axis = resolveCallAxis(spec, args, argTypes, inputT);
    // Tensor reducers return 0/1 from the helper; coerce to a JS
    // bool when the output type is logical so the interpreter path
    // stays consistent with the emitJs / mtoc2-to-numbl shape.
    const wrap = (v: RuntimeValue): RuntimeValue =>
      spec.outputElem === "logical" && typeof v === "number" ? v !== 0 : v;
    if (axis.kind === "all") {
      return [wrap(reducer.all(t) as RuntimeValue)];
    }
    // Mirror the codegen scalar-collapse check (see reductionEmitJs).
    if (inputT.shape !== undefined) {
      const r = reduceConcreteShape(inputT.shape, axis.dim);
      if (r.scalar) return [wrap(reducer.all(t) as RuntimeValue)];
    } else {
      const r = reduceLatticeDims(inputT.dims, axis.dim);
      if (r.scalar) return [wrap(reducer.all(t) as RuntimeValue)];
    }
    return [reducer.dim(t, axis.dim) as RuntimeValue];
  };
}

export function reductionEmit(spec: {
  name: string;
  dimArgIndex: 1 | 2;
  outputElem: "double" | "logical";
}): Builtin["emitC"] {
  return ({ argsC, argTypes, useRuntime }) => {
    useRuntime("mtoc2_tensor_reduce_real");
    useRuntime("mtoc2_tensor_reduce_complex");
    useRuntime("mtoc2_cscalar");
    if (spec.dimArgIndex === 2 && argTypes.length === 2) {
      return `${spec.name === "max" ? "fmax" : "fmin"}((double)(${argsC[0]}), (double)(${argsC[1]}))`;
    }
    const inputT = argTypes[0];
    if (!isNumeric(inputT)) {
      throw new Error(`internal: ${spec.name} codegen got non-numeric arg`);
    }
    const isComplex = inputT.isComplex;
    const suffixAll = isComplex ? "_complex_all" : "_all";
    const suffixDim = isComplex ? "_complex_dim" : "_dim";
    if (isScalar(inputT)) {
      if (spec.outputElem === "logical") {
        if (isComplex) {
          return `(mtoc2_cnonzero(${argsC[0]}) ? 1.0 : 0.0)`;
        }
        return `((${argsC[0]}) != 0.0 ? 1.0 : 0.0)`;
      }
      return argsC[0];
    }
    const dimType: Type | undefined = argTypes[spec.dimArgIndex];
    let axis: AxisAll | AxisFixed;
    if (dimType === undefined) {
      axis = chooseDefaultAxis(spec.name, inputT);
    } else if (
      (dimType.kind === "String" || dimType.kind === "Char") &&
      dimType.exact === "all"
    ) {
      axis = { kind: "all" };
    } else if (isNumeric(dimType)) {
      const v = exactDouble(dimType);
      if (v === undefined) {
        throw new Error(
          `internal: ${spec.name} codegen reached with non-exact dim`
        );
      }
      axis = { kind: "fixed", dim: v };
    } else {
      throw new Error(`internal: ${spec.name} codegen unexpected dim type`);
    }
    if (axis.kind === "all") {
      return `mtoc2_${spec.name}${suffixAll}(${argsC[0]})`;
    }
    if (inputT.shape !== undefined) {
      const r = reduceConcreteShape(inputT.shape, axis.dim);
      if (r.scalar) {
        return `mtoc2_${spec.name}${suffixAll}(${argsC[0]})`;
      }
    } else {
      const r = reduceLatticeDims(inputT.dims, axis.dim);
      if (r.scalar) {
        return `mtoc2_${spec.name}${suffixAll}(${argsC[0]})`;
      }
    }
    return `mtoc2_${spec.name}${suffixDim}(${argsC[0]}, ${axis.dim})`;
  };
}

// ── Builtin assembly helper ────────────────────────────────────────────

/** Build a complete `Builtin` registration from a `KernelSpec`. */
export function defineReducer(spec: KernelSpec): Builtin {
  const isMinMax = spec.dimArgIndex === 2;
  const minArgs = 1;
  const maxArgs = isMinMax ? 3 : 2;
  return {
    name: spec.name,
    transfer(argTypes, nargout) {
      if (argTypes.length < minArgs || argTypes.length > maxArgs) {
        throw new TypeError(
          `'${spec.name}' expects ${minArgs}..${maxArgs} arg(s), got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${spec.name}' does not support multi-output (nargout=${nargout})`
        );
      }
      return [reductionTransfer(argTypes, spec)];
    },
    emitC: reductionEmit({
      name: spec.name,
      dimArgIndex: spec.dimArgIndex,
      outputElem: spec.outputElem,
    }),
    emitJs: reductionEmitJs({
      name: spec.name,
      dimArgIndex: spec.dimArgIndex,
      outputElem: spec.outputElem,
    }),
    call: reductionCall({
      name: spec.name,
      dimArgIndex: spec.dimArgIndex,
      outputElem: spec.outputElem,
    }),
  };
}

// ── Per-op sign rules (shared by the registrations) ────────────────────

export function sumSign(t: NumericType, nonEmpty: boolean): Sign {
  // sum returns 0 on empty input (numbl semantics). So a "positive"
  // input on a maybe-empty fiber still has 0 as a possible value.
  switch (t.sign) {
    case "positive":
      return nonEmpty ? "positive" : "nonneg";
    case "nonneg":
    case "zero":
      return "nonneg";
    case "negative":
      return nonEmpty ? "negative" : "nonpositive";
    case "nonpositive":
      return "nonpositive";
    default:
      return "unknown";
  }
}

export function meanSign(t: NumericType, nonEmpty: boolean): Sign {
  // mean of an empty fiber is NaN (unknown sign). If non-empty, the
  // mean preserves the input's bound.
  if (!nonEmpty) return "unknown";
  switch (t.sign) {
    case "positive":
      return "positive";
    case "nonneg":
    case "zero":
      return "nonneg";
    case "negative":
      return "negative";
    case "nonpositive":
      return "nonpositive";
    default:
      return "unknown";
  }
}

export function prodSign(t: NumericType, nonEmpty: boolean): Sign {
  // prod returns 1 on empty input.
  switch (t.sign) {
    case "positive":
      return "positive"; // 1 or positive — both positive.
    case "nonneg":
      return "nonneg"; // 1 (positive) or nonneg.
    case "zero":
      return nonEmpty ? "zero" : "unknown"; // empty → 1, non-empty → 0.
    default:
      return "unknown"; // sign-mixing can flip everything.
  }
}

/** Sign transfer for `min` and `max` reductions on a real fiber.
 *  min and max preserve the same sign classes: both stay in the
 *  half-line their input occupies (positive→positive,
 *  negative→negative, nonneg/zero→nonneg, nonpositive→nonpositive),
 *  so the two ops share one transfer. */
export function minMaxSign(t: NumericType, nonEmpty: boolean): Sign {
  // Empty min/max → NaN → unknown sign.
  if (!nonEmpty) return "unknown";
  switch (t.sign) {
    case "positive":
      return "positive";
    case "nonneg":
    case "zero":
      return "nonneg";
    case "negative":
      return "negative";
    case "nonpositive":
      return "nonpositive";
    default:
      return "unknown";
  }
}

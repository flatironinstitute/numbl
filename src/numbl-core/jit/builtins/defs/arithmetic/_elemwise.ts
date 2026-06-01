/**
 * Shared infrastructure for elementwise real binary builtins
 * (`plus`, `minus`, `times`, `rdivide`).
 */

import { UnsupportedConstruct, TypeError } from "../../../lowering/errors.js";
import {
  type DimInfo,
  DIM_ONE,
  type NumericType,
  type Sign,
  scalarDouble,
  scalarComplex,
  tensorDouble,
  tensorDoubleFromDims,
  tensorComplexFromDims,
  signFromNumber,
  flipSign,
  isDimOne,
  isMultiElement,
  shapeNumel,
  EXACT_ARRAY_MAX_ELEMENTS,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  requireRealOrComplex,
  exactDouble,
  exactRealArray,
  exactScalarAsComplex,
} from "../_shared.js";
import {
  isComplexValue,
  isTensor,
  type RuntimeTensor,
  type RuntimeValue,
} from "../../../runtime/value.js";
import {
  mtoc2_tensor_plus_tt,
  mtoc2_tensor_plus_ts,
  mtoc2_tensor_plus_bcast_tt,
  mtoc2_tensor_minus_tt,
  mtoc2_tensor_minus_ts,
  mtoc2_tensor_minus_st,
  mtoc2_tensor_minus_bcast_tt,
  mtoc2_tensor_times_tt,
  mtoc2_tensor_times_ts,
  mtoc2_tensor_times_bcast_tt,
  mtoc2_tensor_rdivide_tt,
  mtoc2_tensor_rdivide_ts,
  mtoc2_tensor_rdivide_st,
  mtoc2_tensor_rdivide_bcast_tt,
  mtoc2_tensor_atan2_tt,
  mtoc2_tensor_atan2_ts,
  mtoc2_tensor_atan2_st,
  mtoc2_tensor_atan2_bcast_tt,
  mtoc2_tensor_hypot_tt,
  mtoc2_tensor_hypot_ts,
  mtoc2_tensor_hypot_bcast_tt,
  mtoc2_tensor_rem_tt,
  mtoc2_tensor_rem_ts,
  mtoc2_tensor_rem_st,
  mtoc2_tensor_rem_bcast_tt,
  mtoc2_tensor_mod_tt,
  mtoc2_tensor_mod_ts,
  mtoc2_tensor_mod_st,
  mtoc2_tensor_mod_bcast_tt,
  mtoc2_tensor_plus_complex_tt,
  mtoc2_tensor_plus_complex_ts,
  mtoc2_tensor_plus_complex_bcast_tt,
  mtoc2_tensor_minus_complex_tt,
  mtoc2_tensor_minus_complex_ts,
  mtoc2_tensor_minus_complex_st,
  mtoc2_tensor_minus_complex_bcast_tt,
  mtoc2_tensor_times_complex_tt,
  mtoc2_tensor_times_complex_ts,
  mtoc2_tensor_times_complex_bcast_tt,
  mtoc2_tensor_rdivide_complex_tt,
  mtoc2_tensor_rdivide_complex_ts,
  mtoc2_tensor_rdivide_complex_st,
  mtoc2_tensor_rdivide_complex_bcast_tt,
} from "../../runtime/snippets.gen.js";

/** Project a runtime scalar into `{re, im}` form for complex
 *  arithmetic. Real `number` operands pad with `im=0`. Used by every
 *  scalar-complex `call` path. */
function scalarAsComplexValue(v: RuntimeValue): { re: number; im: number } {
  if (typeof v === "number") return { re: v, im: 0 };
  if (typeof v === "boolean") return { re: v ? 1 : 0, im: 0 };
  if (isTensor(v) && v.data.length === 1) {
    return { re: v.data[0], im: v.imag ? v.imag[0] : 0 };
  }
  if (isComplexValue(v)) return { re: v.re, im: v.im };
  throw new Error(`scalarAsComplexValue: not a scalar (got ${typeof v})`);
}

type TensorHelper2 = (a: RuntimeTensor, b: RuntimeTensor) => RuntimeTensor;
type TensorHelperTS = (a: RuntimeTensor, s: number) => RuntimeTensor;
type TensorHelperST = (s: number, a: RuntimeTensor) => RuntimeTensor;

interface TensorHelperSet {
  tt: TensorHelper2;
  ts: TensorHelperTS;
  /** Non-commutative ops define this. */
  st?: TensorHelperST;
  bcast_tt: TensorHelper2;
}

// `.js` snippet bodies infer their return type as `{ mtoc2Tag: string,
// ... }` (no literal "tensor" discriminator). Cast through `unknown`
// at the registration site so the typed dispatch table below stays
// the source of truth for what these helpers return at runtime.
// Complex-tensor helper set. Each entry mirrors a `tensor_<base>_complex_*`
// JS runtime helper that takes / returns RuntimeTensor objects with both
// `data` (real lane) and `imag` (imaginary lane) populated. Empty today —
// helpers land alongside the per-builtin tensor-complex `emitJs`/`call`
// rollout. When a name is missing, the complex-tensor branch in the
// generated `call` hook raises with a clear "not yet wired" message.
type TensorHelperComplexScalar = (
  a: RuntimeTensor,
  s: { re: number; im: number }
) => RuntimeTensor;
type TensorHelperScalarComplex = (
  s: { re: number; im: number },
  a: RuntimeTensor
) => RuntimeTensor;
interface ComplexTensorHelperSet {
  tt: TensorHelper2;
  ts: TensorHelperComplexScalar;
  st?: TensorHelperScalarComplex;
  bcast_tt: TensorHelper2;
}
const COMPLEX_TENSOR_HELPERS: Record<string, ComplexTensorHelperSet> = {
  plus: {
    tt: mtoc2_tensor_plus_complex_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_plus_complex_ts as unknown as TensorHelperComplexScalar,
    bcast_tt: mtoc2_tensor_plus_complex_bcast_tt as unknown as TensorHelper2,
  },
  minus: {
    tt: mtoc2_tensor_minus_complex_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_minus_complex_ts as unknown as TensorHelperComplexScalar,
    st: mtoc2_tensor_minus_complex_st as unknown as TensorHelperScalarComplex,
    bcast_tt: mtoc2_tensor_minus_complex_bcast_tt as unknown as TensorHelper2,
  },
  times: {
    tt: mtoc2_tensor_times_complex_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_times_complex_ts as unknown as TensorHelperComplexScalar,
    bcast_tt: mtoc2_tensor_times_complex_bcast_tt as unknown as TensorHelper2,
  },
  rdivide: {
    tt: mtoc2_tensor_rdivide_complex_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_rdivide_complex_ts as unknown as TensorHelperComplexScalar,
    st: mtoc2_tensor_rdivide_complex_st as unknown as TensorHelperScalarComplex,
    bcast_tt: mtoc2_tensor_rdivide_complex_bcast_tt as unknown as TensorHelper2,
  },
};

const TENSOR_HELPERS: Record<string, TensorHelperSet> = {
  plus: {
    tt: mtoc2_tensor_plus_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_plus_ts as unknown as TensorHelperTS,
    bcast_tt: mtoc2_tensor_plus_bcast_tt as unknown as TensorHelper2,
  },
  minus: {
    tt: mtoc2_tensor_minus_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_minus_ts as unknown as TensorHelperTS,
    st: mtoc2_tensor_minus_st as unknown as TensorHelperST,
    bcast_tt: mtoc2_tensor_minus_bcast_tt as unknown as TensorHelper2,
  },
  times: {
    tt: mtoc2_tensor_times_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_times_ts as unknown as TensorHelperTS,
    bcast_tt: mtoc2_tensor_times_bcast_tt as unknown as TensorHelper2,
  },
  rdivide: {
    tt: mtoc2_tensor_rdivide_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_rdivide_ts as unknown as TensorHelperTS,
    st: mtoc2_tensor_rdivide_st as unknown as TensorHelperST,
    bcast_tt: mtoc2_tensor_rdivide_bcast_tt as unknown as TensorHelper2,
  },
  atan2: {
    tt: mtoc2_tensor_atan2_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_atan2_ts as unknown as TensorHelperTS,
    st: mtoc2_tensor_atan2_st as unknown as TensorHelperST,
    bcast_tt: mtoc2_tensor_atan2_bcast_tt as unknown as TensorHelper2,
  },
  hypot: {
    tt: mtoc2_tensor_hypot_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_hypot_ts as unknown as TensorHelperTS,
    bcast_tt: mtoc2_tensor_hypot_bcast_tt as unknown as TensorHelper2,
  },
  rem: {
    tt: mtoc2_tensor_rem_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_rem_ts as unknown as TensorHelperTS,
    st: mtoc2_tensor_rem_st as unknown as TensorHelperST,
    bcast_tt: mtoc2_tensor_rem_bcast_tt as unknown as TensorHelper2,
  },
  mod: {
    tt: mtoc2_tensor_mod_tt as unknown as TensorHelper2,
    ts: mtoc2_tensor_mod_ts as unknown as TensorHelperTS,
    st: mtoc2_tensor_mod_st as unknown as TensorHelperST,
    bcast_tt: mtoc2_tensor_mod_bcast_tt as unknown as TensorHelper2,
  },
};

/** Result of broadcast shape resolution for the elementwise binary path.
 *  `outDims` is the output's per-axis dim info (MATLAB-style implicit
 *  expansion: align on axis 1, pad with 1s on the right, axis-wise
 *  `da == db || da == 1 || db == 1`). `bcast` is true when the C runtime
 *  needs the broadcasting helper rather than the fast `_tt` path —
 *  either the ndims differ or at least one statically-singleton axis
 *  needs to be replicated against a non-singleton sibling. */
type ResolvedShape = {
  outDims: DimInfo[];
  /** True when broadcast (any axis where one side is statically 1 but
   *  the other is not, OR ndims differ). False when the two arg shapes
   *  match dim-for-dim (modulo unknowns the runtime trusts). */
  bcast: boolean;
};

/** Shape resolution for elementwise binary. Returns `null` for scalar+
 *  scalar; otherwise computes the broadcast output shape and whether
 *  broadcasting is required. Throws on statically incompatible axes
 *  (no span — the framework's `withSpan` wrapper attaches the call-site
 *  span). Exported for `power.ts`, which shares the same broadcast rules. */
export function elemwiseResultShape(
  a: NumericType,
  b: NumericType,
  name: string
): ResolvedShape | null {
  const aMulti = isMultiElement(a);
  const bMulti = isMultiElement(b);
  if (!aMulti && !bMulti) return null; // scalar OP scalar
  if (!aMulti) return { outDims: b.dims.slice(), bcast: false };
  if (!bMulti) return { outDims: a.dims.slice(), bcast: false };
  // Both tensor. Pad to common ndim (trailing 1s) and check axis-wise.
  const rnd = Math.max(a.dims.length, b.dims.length);
  const outDims: DimInfo[] = new Array(rnd);
  let bcast = a.dims.length !== b.dims.length;
  for (let i = 0; i < rnd; i++) {
    const da: DimInfo = i < a.dims.length ? a.dims[i] : DIM_ONE;
    const db: DimInfo = i < b.dims.length ? b.dims[i] : DIM_ONE;
    const aOne = isDimOne(da);
    const bOne = isDimOne(db);
    if (aOne && bOne) {
      outDims[i] = DIM_ONE;
      continue;
    }
    if (aOne) {
      outDims[i] = db;
      bcast = true;
      continue;
    }
    if (bOne) {
      outDims[i] = da;
      bcast = true;
      continue;
    }
    // Neither statically 1. If both exact, must match.
    if (da.kind === "exact" && db.kind === "exact") {
      if (da.value !== db.value) {
        throw new UnsupportedConstruct(
          `'${name}' shape mismatch (${dimsToStr(a.dims)} vs ${dimsToStr(b.dims)}); axis ${i + 1} is ${da.value} vs ${db.value} — incompatible for implicit expansion`
        );
      }
      outDims[i] = da;
    } else if (da.kind === "exact") {
      outDims[i] = da;
    } else if (db.kind === "exact") {
      outDims[i] = db;
    } else {
      outDims[i] = { kind: "unknown" };
    }
  }
  return { outDims, bcast };
}

function dimsToStr(dims: ReadonlyArray<DimInfo>): string {
  return dims.map(d => (d.kind === "exact" ? `${d.value}` : "?")).join("×");
}

/** True iff an elementwise binary op on tensor operands `a` and `b`
 *  needs the broadcasting `_bcast_tt` helper rather than the same-shape
 *  `_tt` fast path. Mirrors the per-axis compatibility logic inside
 *  `elemwiseResultShape` but returns just the bool — codegen call
 *  sites don't need to thread the resolved output shape from transfer.
 *  Exported for `power.ts`, which shares the same codegen discipline. */
export function needsBroadcast(a: NumericType, b: NumericType): boolean {
  if (a.dims.length !== b.dims.length) return true;
  for (let i = 0; i < a.dims.length; i++) {
    if (isDimOne(a.dims[i]) !== isDimOne(b.dims[i])) return true;
  }
  return false;
}

/** Compute the broadcast-aware elementwise fold over `a` and `b` for
 *  the given (fully-exact) output shape, applying `step` per slot.
 *  Both sides must already carry exact data (a `Float64Array` or a
 *  scalar `number`); the caller is responsible for the
 *  `EXACT_ARRAY_MAX_ELEMENTS` cap. Singleton axes on either side get
 *  stride 0 so they broadcast against non-singleton output axes;
 *  output is filled in column-major order to match the runtime layout.
 *  Exported for `power.ts`, which shares the same broadcast rules. */
export function broadcastFoldExact(
  a: NumericType,
  b: NumericType,
  outShape: number[],
  step: (av: number, bv: number) => number
): Float64Array {
  const total = shapeNumel(outShape);
  const rnd = outShape.length;
  const aArr = exactRealArray(a);
  const bArr = exactRealArray(b);
  const ax = exactDouble(a);
  const bx = exactDouble(b);
  const aShape: number[] = new Array(rnd);
  const bShape: number[] = new Array(rnd);
  for (let i = 0; i < rnd; i++) {
    const da = i < a.dims.length ? a.dims[i] : DIM_ONE;
    const db = i < b.dims.length ? b.dims[i] : DIM_ONE;
    aShape[i] = da.kind === "exact" ? da.value : 1;
    bShape[i] = db.kind === "exact" ? db.value : 1;
  }
  const aStride: number[] = new Array(rnd);
  const bStride: number[] = new Array(rnd);
  let aAcc = 1;
  let bAcc = 1;
  for (let i = 0; i < rnd; i++) {
    aStride[i] = aShape[i] === 1 ? 0 : aAcc;
    bStride[i] = bShape[i] === 1 ? 0 : bAcc;
    aAcc *= aShape[i];
    bAcc *= bShape[i];
  }
  const data = new Float64Array(total);
  const ix = new Array(rnd).fill(0);
  for (let k = 0; k < total; k++) {
    let ai = 0;
    let bi = 0;
    for (let i = 0; i < rnd; i++) {
      ai += ix[i] * aStride[i];
      bi += ix[i] * bStride[i];
    }
    const av = aArr ? aArr[ai] : (ax as number);
    const bv = bArr ? bArr[bi] : (bx as number);
    data[k] = step(av, bv);
    for (let i = 0; i < rnd; i++) {
      ix[i]++;
      if (ix[i] < outShape[i]) break;
      ix[i] = 0;
    }
  }
  return data;
}

/** Build a real elementwise binary builtin: scalar, scalar+tensor,
 *  tensor+scalar, tensor+tensor. Folds when every input is exact.
 *  `helperBase` names the matching set of C runtime helpers
 *  (mtoc2_tensor_<helperBase>_tt / _ts / _st). Commutative ops set
 *  `commutative=true` so the scalar-first path reuses `_ts` with
 *  swapped operands.
 *
 *  The four shape combos and what they emit:
 *
 *  | a       | b       | C output                                   |
 *  |---------|---------|--------------------------------------------|
 *  | scalar  | scalar  | `(argsC[0] cOp argsC[1])`                  |
 *  | tensor  | scalar  | `mtoc2_tensor_<helperBase>_ts(a, b)`       |
 *  | scalar  | tensor  | commutative → `_ts(b, a)`; else `_st(a,b)` |
 *  | tensor  | tensor  | `mtoc2_tensor_<helperBase>_tt(a, b)`       |
 *
 *  `complexFold` / `complexScalarExpr` / `complexRuntimeDeps` are
 *  optional; when set, a scalar+scalar op with at least one complex
 *  operand routes through them (and `transfer` returns a complex
 *  result). Phase 1 wires complex scalars; tensor+complex paths are
 *  rejected until Phase 3 lands the runtime helpers.
 */
export function defineElemwiseRealBinary(opts: {
  name: string;
  cOp: string;
  helperBase: string;
  commutative: boolean;
  fold: (a: number, b: number) => number;
  signRule: (a: NumericType, b: NumericType) => Sign;
  /** Scalar-complex fold. Receives two `{re, im}` operands. */
  complexFold?: (
    a: { re: number; im: number },
    b: { re: number; im: number }
  ) => { re: number; im: number };
  /** Scalar-complex C expression. Defaults to the same `(a cOp b)`
   *  infix as the real path — works for `+ - *` because C99 promotes
   *  real↔complex automatically. Divide overrides to route through
   *  `mtoc2_cdiv(a, b)`. */
  complexScalarExpr?: (aC: string, bC: string) => string;
  /** Extra runtime snippet activations needed by the complex scalar
   *  path (e.g. `mtoc2_cdiv` for division). The real-only `_real`
   *  helper still activates unconditionally for the tensor path. */
  complexRuntimeDeps?: string[];
}): Builtin {
  // Default scalar-complex emit: route through the per-op helper from
  // `cscalar.h` instead of relying on C99 operator overloading on
  // `_Complex` operands. This keeps mtoc2's user-code emission
  // operator-free for complex, which lets the c2js JS backend ship
  // matching `{re, im}` helpers without learning to type-track
  // expressions. The C native path inlines `mtoc2_c*` back to the
  // same instructions C99 would have generated, so there's no cost.
  const defaultComplexHelper: Record<string, string | undefined> = {
    "+": "mtoc2_cadd",
    "-": "mtoc2_csub",
    "*": "mtoc2_cmul",
  };
  const helperName = defaultComplexHelper[opts.cOp];
  const defaultComplexScalarExpr = helperName
    ? (a: string, b: string) => `${helperName}(${a}, ${b})`
    : (a: string, b: string) => `(${a} ${opts.cOp} ${b})`;
  const defaultComplexDeps = helperName ? ["mtoc2_cscalar"] : [];
  return buildElemwiseRealBinary({
    name: opts.name,
    helperBase: opts.helperBase,
    commutative: opts.commutative,
    fold: opts.fold,
    signRule: opts.signRule,
    scalarExpr: (a, b) => `(${a} ${opts.cOp} ${b})`,
    runtimeDep: "mtoc2_tensor_elemwise_real",
    complexFold: opts.complexFold,
    complexScalarExpr: opts.complexScalarExpr ?? defaultComplexScalarExpr,
    complexRuntimeDeps: opts.complexRuntimeDeps ?? defaultComplexDeps,
  });
}

/** Same as `defineElemwiseRealBinary`, but the scalar path emits a C
 *  function call (`fmod(a,b)`, `atan2(a,b)`, …) instead of an infix
 *  operator. The tensor helpers still follow the `_tt`/`_ts`/`_st`
 *  naming convention; supply the appropriate `runtimeDep` for the
 *  snippet that defines them (`mtoc2_tensor_elemwise_real_fn`
 *  covers `mod`/`rem`/`atan2`/`hypot`). */
export function defineElemwiseRealBinaryFn(opts: {
  name: string;
  cFn: string;
  helperBase: string;
  commutative: boolean;
  fold: (a: number, b: number) => number;
  signRule: (a: NumericType, b: NumericType) => Sign;
  runtimeDep: string;
  /** Optional JS scalar form. Defaults to `${cFn}(${a}, ${b})` (works
   *  when `cFn` happens to be a JS function on the surrounding scope
   *  — almost never the case). Pass `(a, b) => "Math.atan2(...)"`
   *  for `atan2`/`hypot`, an explicit JS-modulo expression for
   *  `mod`/`rem`, etc. */
  jsScalarExpr?: (aJs: string, bJs: string) => string;
}): Builtin {
  // When the scalar `cFn` is a mtoc2_-prefixed helper (e.g.
  // `mtoc2_mod_real`), the scalar emit needs to activate the runtime
  // snippet that defines it — even when no tensor path uses it.
  // libc functions (`fmod`, `atan2`, `hypot`, …) are declared via
  // `<math.h>` and don't need activation.
  const scalarRuntimeDep = opts.cFn.startsWith("mtoc2_")
    ? opts.runtimeDep
    : undefined;
  return buildElemwiseRealBinary({
    name: opts.name,
    helperBase: opts.helperBase,
    commutative: opts.commutative,
    fold: opts.fold,
    signRule: opts.signRule,
    scalarExpr: (a, b) => `${opts.cFn}(${a}, ${b})`,
    runtimeDep: opts.runtimeDep,
    ...(scalarRuntimeDep !== undefined ? { scalarRuntimeDep } : {}),
    ...(opts.jsScalarExpr !== undefined
      ? { jsScalarExpr: opts.jsScalarExpr }
      : {}),
  });
}

function buildElemwiseRealBinary(opts: {
  name: string;
  helperBase: string;
  commutative: boolean;
  fold: (a: number, b: number) => number;
  signRule: (a: NumericType, b: NumericType) => Sign;
  scalarExpr: (aC: string, bC: string) => string;
  /** Optional JS scalar form (for `emitJs`). When omitted, the
   *  factory uses `scalarExpr` — which is correct only when the C
   *  expression is also valid JS (i.e. infix `+`/`-`/`*`/`/`). The
   *  function-call forms (`atan2`, `hypot`, `fmod`, mtoc2's custom
   *  mod) must supply this. */
  jsScalarExpr?: (aJs: string, bJs: string) => string;
  runtimeDep: string;
  /** Runtime snippet to activate when emitting the C scalar form.
   *  Used when `scalarExpr` calls a mtoc2_-prefixed helper that lives
   *  in the same snippet as the tensor helpers. Without this, the
   *  scalar path emits a call to an undeclared function. */
  scalarRuntimeDep?: string;
  complexFold?: (
    a: { re: number; im: number },
    b: { re: number; im: number }
  ) => { re: number; im: number };
  complexScalarExpr?: (aC: string, bC: string) => string;
  complexRuntimeDeps?: string[];
}): Builtin {
  const {
    name,
    helperBase,
    commutative,
    fold,
    signRule,
    scalarExpr,
    runtimeDep,
    scalarRuntimeDep,
    complexFold,
    complexScalarExpr,
    complexRuntimeDeps,
  } = opts;
  const jsScalarExpr = opts.jsScalarExpr ?? scalarExpr;
  return {
    name,
    transfer(argTypes, nargout) {
      if (argTypes.length !== 2) {
        throw new TypeError(
          `'${name}' expects 2 arg(s), got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      const a = argTypes[0];
      const b = argTypes[1];
      requireRealOrComplex(a, `'${name}' arg 1`);
      requireRealOrComplex(b, `'${name}' arg 2`);
      const aN = a as NumericType;
      const bN = b as NumericType;
      // Complex contamination: any complex operand makes the result
      // complex. The complex elemwise helpers tolerate an `imag ==
      // NULL` operand (treated as zero imag), so we can hand a real
      // tensor straight into the complex_tt path — no promote step.
      const aCx = aN.isComplex;
      const bCx = bN.isComplex;
      const anyComplex = aCx || bCx;
      if (anyComplex) {
        if (complexFold === undefined) {
          throw new UnsupportedConstruct(
            `'${name}' is not defined for complex scalars`
          );
        }
        if (!isMultiElement(aN) && !isMultiElement(bN)) {
          // Scalar+scalar (any mix) — fold or emit runtime.
          const ax = exactScalarAsComplex(aN);
          const bx = exactScalarAsComplex(bN);
          if (ax !== undefined && bx !== undefined) {
            const v = complexFold(ax, bx);
            if (Number.isFinite(v.re) && Number.isFinite(v.im)) {
              return [scalarComplex(v)];
            }
          }
          return [scalarComplex()];
        }
        // At least one is a complex tensor (the all-real-tensor +
        // complex-scalar case is also here). Build the result shape
        // via the same broadcast logic; the result is a complex
        // tensor (or scalar if both are scalar — handled above).
        const resolved = elemwiseResultShape(aN, bN, name);
        if (resolved === null) {
          // Unreachable: at least one is multi-element by the check
          // above. Defensive.
          return [scalarComplex()];
        }
        return [tensorComplexFromDims(resolved.outDims)];
      }
      const resolved = elemwiseResultShape(aN, bN, name);

      if (resolved === null) {
        // Pure scalar op — fold if exact.
        const ax = exactDouble(aN);
        const bx = exactDouble(bN);
        if (ax !== undefined && bx !== undefined) {
          const v = fold(ax, bx);
          if (Number.isFinite(v)) return [scalarDouble(signFromNumber(v), v)];
        }
        return [scalarDouble(signRule(aN, bN))];
      }

      // Build the result NumericType (carries `shape` automatically when
      // every output dim is exact).
      const outTy = tensorDoubleFromDims(resolved.outDims);

      // Try to fold when every input is exact AND every output dim is
      // known (so the result fits the exact-array cap).
      const aIsExact =
        exactRealArray(aN) !== undefined || exactDouble(aN) !== undefined;
      const bIsExact =
        exactRealArray(bN) !== undefined || exactDouble(bN) !== undefined;
      if (
        aIsExact &&
        bIsExact &&
        outTy.shape !== undefined &&
        shapeNumel(outTy.shape) <= EXACT_ARRAY_MAX_ELEMENTS
      ) {
        const data = broadcastFoldExact(aN, bN, outTy.shape, fold);
        return [tensorDouble(outTy.shape, data)];
      }
      outTy.sign = signRule(aN, bN);
      return [outTy];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      const aN = argTypes[0] as NumericType;
      const bN = argTypes[1] as NumericType;
      const aMulti = isMultiElement(aN);
      const bMulti = isMultiElement(bN);
      const anyComplex = aN.isComplex || bN.isComplex;
      if (!aMulti && !bMulti) {
        if (anyComplex) {
          if (complexScalarExpr === undefined) {
            throw new Error(
              `internal: '${name}' missing complexScalarExpr but reached complex codegen`
            );
          }
          if (complexRuntimeDeps) {
            for (const d of complexRuntimeDeps) useRuntime(d);
          }
          return complexScalarExpr(argsC[0], argsC[1]);
        }
        if (scalarRuntimeDep !== undefined) {
          useRuntime(scalarRuntimeDep);
        }
        return scalarExpr(argsC[0], argsC[1]);
      }
      // Tensor paths — activate the runtime helper for this op.
      useRuntime(runtimeDep);
      if (anyComplex) {
        // Tensor + scalar / scalar + tensor with at least one
        // complex operand. The runtime helpers take a `double _Complex`
        // for the scalar slot; project a real scalar to that via
        // `mtoc2_cmake(re, 0.0)` at emit. The receiver tensor is
        // already complex (we reject mixed real-tensor + complex-
        // tensor at the transfer layer).
        useRuntime("mtoc2_tensor_elemwise_complex");
        useRuntime("mtoc2_cscalar");
        const base = `mtoc2_tensor_${helperBase}_complex`;
        const promote = (c: string, isComplexArg: boolean): string =>
          isComplexArg ? c : `mtoc2_cmake(${c}, 0.0)`;
        if (aMulti && bMulti) {
          return needsBroadcast(aN, bN)
            ? `${base}_bcast_tt(${argsC[0]}, ${argsC[1]})`
            : `${base}_tt(${argsC[0]}, ${argsC[1]})`;
        }
        if (aMulti) {
          return `${base}_ts(${argsC[0]}, ${promote(argsC[1], bN.isComplex)})`;
        }
        // scalar OP tensor
        if (commutative) {
          return `${base}_ts(${argsC[1]}, ${promote(argsC[0], aN.isComplex)})`;
        }
        return `${base}_st(${promote(argsC[0], aN.isComplex)}, ${argsC[1]})`;
      }
      if (aMulti && bMulti) {
        return needsBroadcast(aN, bN)
          ? `mtoc2_tensor_${helperBase}_bcast_tt(${argsC[0]}, ${argsC[1]})`
          : `mtoc2_tensor_${helperBase}_tt(${argsC[0]}, ${argsC[1]})`;
      }
      if (aMulti) {
        return `mtoc2_tensor_${helperBase}_ts(${argsC[0]}, ${argsC[1]})`;
      }
      // scalar OP tensor
      if (commutative) {
        return `mtoc2_tensor_${helperBase}_ts(${argsC[1]}, ${argsC[0]})`;
      }
      return `mtoc2_tensor_${helperBase}_st(${argsC[0]}, ${argsC[1]})`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const aN = argTypes[0] as NumericType;
      const bN = argTypes[1] as NumericType;
      const aMulti = isMultiElement(aN);
      const bMulti = isMultiElement(bN);
      const anyComplex = aN.isComplex || bN.isComplex;
      if (!aMulti && !bMulti) {
        if (anyComplex) {
          if (complexScalarExpr === undefined) {
            throw new UnsupportedConstruct(
              `'${name}' is not defined for complex scalars`
            );
          }
          if (complexRuntimeDeps) {
            for (const d of complexRuntimeDeps) useRuntime(d);
          }
          // JS has no implicit `double → double _Complex` promotion
          // the way C99 does, so promote real operands to `{re, im}`
          // form at the call site. The C path passes plain doubles
          // and lets the C compiler do the cast; here we do it
          // explicitly so the helper sees {re,im} on both sides.
          useRuntime("mtoc2_cscalar");
          const promote = (j: string, isComplexArg: boolean): string =>
            isComplexArg ? j : `mtoc2_cmake(${j}, 0.0)`;
          return complexScalarExpr(
            promote(argsJs[0], aN.isComplex),
            promote(argsJs[1], bN.isComplex)
          );
        }
        return jsScalarExpr(argsJs[0], argsJs[1]);
      }
      const helpers = TENSOR_HELPERS[helperBase];
      if (!helpers) {
        throw new UnsupportedConstruct(
          `'${name}' tensor emitJs not yet wired (no helper for helperBase '${helperBase}')`
        );
      }
      // Activate the runtime snippet this builtin's C path activates —
      // its JS sibling carries the matching kernels. Plus/minus/times/
      // rdivide route through `tensor_elemwise_real`; atan2/hypot/mod/
      // rem/power route through `tensor_elemwise_real_fn`.
      useRuntime(runtimeDep);
      if (anyComplex) {
        useRuntime("mtoc2_tensor_elemwise_complex");
        useRuntime("mtoc2_cscalar");
        const base = `mtoc2_tensor_${helperBase}_complex`;
        const promote = (j: string, isComplexArg: boolean): string =>
          isComplexArg ? j : `mtoc2_cmake(${j}, 0.0)`;
        if (aMulti && bMulti) {
          return needsBroadcast(aN, bN)
            ? `${base}_bcast_tt(${argsJs[0]}, ${argsJs[1]})`
            : `${base}_tt(${argsJs[0]}, ${argsJs[1]})`;
        }
        if (aMulti) {
          return `${base}_ts(${argsJs[0]}, ${promote(argsJs[1], bN.isComplex)})`;
        }
        // scalar OP tensor
        if (commutative) {
          return `${base}_ts(${argsJs[1]}, ${promote(argsJs[0], aN.isComplex)})`;
        }
        return `${base}_st(${promote(argsJs[0], aN.isComplex)}, ${argsJs[1]})`;
      }
      if (aMulti && bMulti) {
        if (needsBroadcast(aN, bN)) {
          return `mtoc2_tensor_${helperBase}_bcast_tt(${argsJs[0]}, ${argsJs[1]})`;
        }
        return `mtoc2_tensor_${helperBase}_tt(${argsJs[0]}, ${argsJs[1]})`;
      }
      if (aMulti) {
        return `mtoc2_tensor_${helperBase}_ts(${argsJs[0]}, ${argsJs[1]})`;
      }
      if (commutative) {
        return `mtoc2_tensor_${helperBase}_ts(${argsJs[1]}, ${argsJs[0]})`;
      }
      return `mtoc2_tensor_${helperBase}_st(${argsJs[0]}, ${argsJs[1]})`;
    },
    call({ args, argTypes }) {
      const aN = argTypes[0] as NumericType;
      const bN = argTypes[1] as NumericType;
      const aMulti = isMultiElement(aN);
      const bMulti = isMultiElement(bN);
      const anyComplex = aN.isComplex || bN.isComplex;
      const aVal = args[0] as RuntimeValue;
      const bVal = args[1] as RuntimeValue;
      if (!aMulti && !bMulti) {
        if (anyComplex) {
          if (complexFold === undefined) {
            throw new UnsupportedConstruct(
              `'${name}' is not defined for complex scalars`
            );
          }
          const ax = scalarAsComplexValue(aVal);
          const bx = scalarAsComplexValue(bVal);
          const out = complexFold(ax, bx);
          // Collapse to a real number when the imag lane is zero,
          // so downstream `disp(...)` and arithmetic stay on the
          // real path. This matches numbl's runtime collapse rule.
          if (out.im === 0 && !aN.isComplex && !bN.isComplex) {
            return [out.re];
          }
          return [out];
        }
        const av = typeof aVal === "number" ? aVal : Number(aVal);
        const bv = typeof bVal === "number" ? bVal : Number(bVal);
        return [fold(av, bv)];
      }
      const helpers = TENSOR_HELPERS[helperBase];
      if (!helpers) {
        throw new UnsupportedConstruct(
          `'${name}' tensor 'call' not yet wired (no helper for helperBase '${helperBase}')`
        );
      }
      if (anyComplex) {
        const cxHelpers = COMPLEX_TENSOR_HELPERS[helperBase];
        if (!cxHelpers) {
          throw new UnsupportedConstruct(
            `'${name}' complex-tensor 'call' not yet wired`
          );
        }
        const promote = (v: RuntimeValue): { re: number; im: number } =>
          scalarAsComplexValue(v);
        if (aMulti && bMulti) {
          const at = aVal as RuntimeTensor;
          const bt = bVal as RuntimeTensor;
          const op = needsBroadcast(aN, bN) ? cxHelpers.bcast_tt : cxHelpers.tt;
          return [op(at, bt)];
        }
        if (aMulti) {
          const at = aVal as RuntimeTensor;
          return [cxHelpers.ts(at, promote(bVal))];
        }
        // scalar OP tensor
        const bt = bVal as RuntimeTensor;
        if (commutative) return [cxHelpers.ts(bt, promote(aVal))];
        if (cxHelpers.st === undefined) {
          throw new UnsupportedConstruct(
            `internal: '${name}' is non-commutative but has no _st complex JS helper`
          );
        }
        return [cxHelpers.st(promote(aVal), bt)];
      }
      if (aMulti && bMulti) {
        const at = aVal as RuntimeTensor;
        const bt = bVal as RuntimeTensor;
        const op = needsBroadcast(aN, bN) ? helpers.bcast_tt : helpers.tt;
        return [op(at, bt)];
      }
      if (aMulti) {
        const at = aVal as RuntimeTensor;
        const bv = typeof bVal === "number" ? bVal : Number(bVal);
        return [helpers.ts(at, bv)];
      }
      // scalar OP tensor
      const av = typeof aVal === "number" ? aVal : Number(aVal);
      const bt = bVal as RuntimeTensor;
      if (commutative) return [helpers.ts(bt, av)];
      if (helpers.st === undefined) {
        throw new UnsupportedConstruct(
          `internal: '${name}' is non-commutative but has no _st JS helper`
        );
      }
      return [helpers.st(av, bt)];
    },
    elementwise: true,
  };
}

export function signSum(a: NumericType, b: NumericType): Sign {
  // Conservative: only positive+positive ⇒ positive, nonneg+nonneg ⇒ nonneg.
  if (a.sign === "positive" && b.sign === "positive") return "positive";
  if (
    (a.sign === "positive" || a.sign === "nonneg" || a.sign === "zero") &&
    (b.sign === "positive" || b.sign === "nonneg" || b.sign === "zero")
  ) {
    return a.sign === "positive" || b.sign === "positive"
      ? "positive"
      : "nonneg";
  }
  if (a.sign === "negative" && b.sign === "negative") return "negative";
  if (
    (a.sign === "negative" || a.sign === "nonpositive" || a.sign === "zero") &&
    (b.sign === "negative" || b.sign === "nonpositive" || b.sign === "zero")
  ) {
    return a.sign === "negative" || b.sign === "negative"
      ? "negative"
      : "nonpositive";
  }
  return "unknown";
}

export function signDiff(a: NumericType, b: NumericType): Sign {
  return signSum(a, { ...b, sign: flipSign(b.sign) });
}

export function signProd(a: NumericType, b: NumericType): Sign {
  if (a.sign === "zero" || b.sign === "zero") return "zero";
  if (a.sign === "positive" && b.sign === "positive") return "positive";
  if (a.sign === "negative" && b.sign === "negative") return "positive";
  if (
    (a.sign === "positive" && b.sign === "negative") ||
    (a.sign === "negative" && b.sign === "positive")
  ) {
    return "negative";
  }
  // Lift the bounds to nonneg/nonpositive when both operands share a
  // sign. `0 * Inf = NaN` is the edge case; NaN is not `< 0` so it
  // doesn't violate the "≥ 0" interpretation our sqrt-domain check
  // relies on (and it doesn't violate `≤ 0` either).
  const aNonneg = a.sign === "positive" || a.sign === "nonneg";
  const bNonneg = b.sign === "positive" || b.sign === "nonneg";
  const aNonpos = a.sign === "negative" || a.sign === "nonpositive";
  const bNonpos = b.sign === "negative" || b.sign === "nonpositive";
  if (aNonneg && bNonneg) return "nonneg";
  if (aNonpos && bNonpos) return "nonneg";
  if ((aNonneg && bNonpos) || (aNonpos && bNonneg)) return "nonpositive";
  return "unknown";
}

/** Sign of `a / b`. Almost the same shape as `signProd`, but the
 *  divisor's sign tells a different story: `x / 0` is ±Inf (or NaN
 *  when `x` is also 0), NOT zero, so a zero divisor degrades the
 *  result to `unknown`. A zero numerator with a non-zero divisor is
 *  zero. */
export function signQuot(a: NumericType, b: NumericType): Sign {
  if (b.sign === "zero") return "unknown";
  if (a.sign === "zero") return "zero";
  if (a.sign === "positive" && b.sign === "positive") return "positive";
  if (a.sign === "negative" && b.sign === "negative") return "positive";
  if (
    (a.sign === "positive" && b.sign === "negative") ||
    (a.sign === "negative" && b.sign === "positive")
  ) {
    return "negative";
  }
  // Same "nonneg / nonneg = nonneg (NaN allowed)" relaxation as
  // signProd. Required for chains like `sqrt(errs / errs0 / k)` where
  // the numerator/denominator are sum-of-squares (nonneg) and we want
  // sqrt to accept the quotient.
  const aNonneg = a.sign === "positive" || a.sign === "nonneg";
  const bNonneg = b.sign === "positive" || b.sign === "nonneg";
  const aNonpos = a.sign === "negative" || a.sign === "nonpositive";
  const bNonpos = b.sign === "negative" || b.sign === "nonpositive";
  if (aNonneg && bNonneg) return "nonneg";
  if (aNonpos && bNonpos) return "nonneg";
  if ((aNonneg && bNonpos) || (aNonpos && bNonneg)) return "nonpositive";
  return "unknown";
}

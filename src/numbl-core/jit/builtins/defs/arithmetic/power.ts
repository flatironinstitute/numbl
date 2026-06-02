/**
 * `power` builtin — backs the `.^` (elementwise power) binary
 * operator.
 *
 * Mirrors `mElemPow` in `../numbl/src/numbl-core/helpers/arithmetic.ts`
 * (line 1075) modulo the complex-result paths. Numbl produces a
 * complex result for `(neg)^(non-integer)`; mtoc2 has no complex
 * type and rejects that combination at lowering time (same precedent
 * as `sqrt` / `log` rejecting out-of-domain real inputs).
 *
 * Shape rules follow the rest of the elementwise binary family:
 *   - scalar OP scalar → scalar; tensor OP scalar / scalar OP tensor /
 *     same-shape tensor OP tensor.
 *   - exact-fold within EXACT_ARRAY_MAX_ELEMENTS when every input
 *     element is exact AND every output element is finite.
 *
 * Sign refinement:
 *   - exponent exact zero → `positive` (x^0 == 1 for every x, even 0
 *     and Inf per IEEE-754 / MATLAB).
 *   - base statically `positive` → `positive`.
 *   - base statically `nonneg` AND exponent statically `nonneg` → `nonneg`.
 *   - otherwise → `unknown`.
 *
 * Domain rejection: when the base could be negative (i.e. its sign
 * isn't statically nonneg) AND the exponent isn't an exact integer
 * (or any exact value at all), reject with a clear span — that's the
 * "complex-result" case numbl handles but mtoc2 v1 doesn't.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  type NumericType,
  type Sign,
  EXACT_ARRAY_MAX_ELEMENTS,
  isMultiElement,
  scalarDouble,
  scalarComplex,
  signFromNumber,
  signIsNonneg,
  shapeNumel,
  tensorDouble,
  tensorDoubleFromDims,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  exactDouble,
  exactRealArray,
  exactScalarAsComplex,
  requireRealOrComplex,
  requireRealDouble,
} from "../_shared.js";
import {
  broadcastFoldExact,
  elemwiseResultShape,
  needsBroadcast,
} from "./_elemwise.js";
import { isComplexValue, type RuntimeTensor } from "../../../runtime/value.js";
import {
  mtoc2_cpow,
  mtoc2_tensor_power_tt,
  mtoc2_tensor_power_ts,
  mtoc2_tensor_power_st,
  mtoc2_tensor_power_bcast_tt,
} from "../../runtime/snippets.gen.js";

function isExactInteger(t: NumericType): boolean {
  const v = exactDouble(t);
  return v !== undefined && Number.isFinite(v) && Number.isInteger(v);
}

function exactArrayAllInteger(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v) || !Number.isInteger(v)) return false;
  }
  return true;
}

function exactArrayAllNonneg(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!(arr[i] >= 0)) return false;
  }
  return true;
}

/** Sign rule for power. See header comment. */
function powerSign(a: NumericType, b: NumericType): Sign {
  // exponent exact zero → result is exactly 1 (positive).
  const bx = exactDouble(b);
  if (bx === 0) return "positive";
  // Even-integer exponent → result is x^(2k), always ≥ 0 regardless
  // of base sign. `nonneg` (becomes `positive` only if the base is
  // provably nonzero, which we don't track).
  if (
    bx !== undefined &&
    Number.isFinite(bx) &&
    Number.isInteger(bx) &&
    bx % 2 === 0
  ) {
    return "nonneg";
  }
  if (signIsNonneg(a.sign) && signIsNonneg(b.sign)) {
    // base positive → positive; base nonneg → nonneg.
    if (a.sign === "positive") return "positive";
    return "nonneg";
  }
  return "unknown";
}

/** Check the base/exponent pair for a possible complex-result case
 *  that mtoc2 v1 doesn't support. Returns null when accepted; an
 *  error message when rejected. */
function checkDomain(a: NumericType, b: NumericType): string | null {
  // Easy accept: base is statically nonneg.
  if (signIsNonneg(a.sign)) return null;

  // Base may be negative — then exponent must be a statically-known
  // integer for the result to stay real.
  if (isExactInteger(b)) return null;

  // Exponent is an exact tensor with all-integer values: also OK.
  const bArr = exactRealArray(b);
  if (bArr !== undefined && exactArrayAllInteger(bArr)) return null;

  // If the base is an exact tensor that happens to be all-nonneg,
  // accept (we can prove no negative-base × fractional-exponent slot).
  const aArr = exactRealArray(a);
  if (aArr !== undefined && exactArrayAllNonneg(aArr)) return null;

  return (
    `'.^' with possibly-negative base and non-integer exponent is not yet ` +
    `supported (would produce a complex result; mtoc2 has no complex type). ` +
    `Use 'abs(base)' or ensure the base is statically nonneg, or pass an ` +
    `integer exponent.`
  );
}

/** Complex scalar power fold. Uses the standard
 *  `z^w = exp(w * log(z))` formula. Returns `undefined` when the
 *  result would be non-finite. */
function cpowFold(
  a: { re: number; im: number },
  b: { re: number; im: number }
): { re: number; im: number } | undefined {
  // log(z) = log|z| + i*arg(z)
  const r = Math.hypot(a.re, a.im);
  if (r === 0) {
    // 0^w: matches C99 — 0 when re(w) > 0, 1 when w == 0, else NaN.
    if (b.re === 0 && b.im === 0) return { re: 1, im: 0 };
    if (b.re > 0) return { re: 0, im: 0 };
    return undefined;
  }
  const logR = Math.log(r);
  const phi = Math.atan2(a.im, a.re);
  // w * log(z)
  const tRe = b.re * logR - b.im * phi;
  const tIm = b.re * phi + b.im * logR;
  // exp(t)
  const expRe = Math.exp(tRe);
  const re = expRe * Math.cos(tIm);
  const im = expRe * Math.sin(tIm);
  if (!Number.isFinite(re) || !Number.isFinite(im)) return undefined;
  return { re, im };
}

export const power: Builtin = {
  name: "power",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(`'power' expects 2 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'power' does not support multi-output (nargout=${nargout})`
      );
    }
    requireRealOrComplex(argTypes[0], `'.^' arg 1`);
    requireRealOrComplex(argTypes[1], `'.^' arg 2`);
    const a0 = argTypes[0] as NumericType;
    const b0 = argTypes[1] as NumericType;
    // Complex contamination — scalar path only (Phase 1).
    if (a0.isComplex || b0.isComplex) {
      if (isMultiElement(a0) || isMultiElement(b0)) {
        throw new UnsupportedConstruct(
          `'.^' on a complex tensor is not yet supported`
        );
      }
      const ax = exactScalarAsComplex(a0);
      const bx = exactScalarAsComplex(b0);
      if (ax !== undefined && bx !== undefined) {
        const v = cpowFold(ax, bx);
        if (v !== undefined) return [scalarComplex(v)];
      }
      return [scalarComplex()];
    }
    // Real path — both args must be real-double after the
    // contamination check above.
    requireRealDouble(argTypes[0], `'.^' arg 1`);
    requireRealDouble(argTypes[1], `'.^' arg 2`);
    const a = argTypes[0] as NumericType;
    const b = argTypes[1] as NumericType;

    const reject = checkDomain(a, b);
    if (reject !== null) throw new UnsupportedConstruct(reject);

    const resolved = elemwiseResultShape(a, b, ".^");

    if (resolved === null) {
      // Pure scalar OP scalar.
      const ax = exactDouble(a);
      const bx = exactDouble(b);
      if (ax !== undefined && bx !== undefined) {
        const v = Math.pow(ax, bx);
        if (Number.isFinite(v)) return [scalarDouble(signFromNumber(v), v)];
      }
      return [scalarDouble(powerSign(a, b))];
    }

    // Tensor result — try exact-fold when the output shape is fully
    // known and small enough.
    const outTy = tensorDoubleFromDims(resolved.outDims);
    const aIsExact =
      exactRealArray(a) !== undefined || exactDouble(a) !== undefined;
    const bIsExact =
      exactRealArray(b) !== undefined || exactDouble(b) !== undefined;
    if (
      aIsExact &&
      bIsExact &&
      outTy.shape !== undefined &&
      shapeNumel(outTy.shape) <= EXACT_ARRAY_MAX_ELEMENTS
    ) {
      const data = broadcastFoldExact(a, b, outTy.shape, Math.pow);
      // Drop the fold if any element went non-finite — defer to runtime
      // (which may surface a clearer NaN/Inf at the right span).
      if (data.every(Number.isFinite)) return [tensorDouble(outTy.shape, data)];
    }
    outTy.sign = powerSign(a, b);
    return [outTy];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const aN = argTypes[0] as NumericType;
    const bN = argTypes[1] as NumericType;
    const aMulti = isMultiElement(aN);
    const bMulti = isMultiElement(bN);
    if (!aMulti && !bMulti) {
      if (aN.isComplex || bN.isComplex) {
        useRuntime("mtoc2_cscalar");
        return `mtoc2_cpow(${argsC[0]}, ${argsC[1]})`;
      }
      // `mtoc2_pow_real`, not bare C `pow`: the interpreter and JS-JIT use
      // `Math.pow`, which returns NaN for |base|==1 with a non-finite
      // exponent (C99 `pow` returns 1). Keeps `--opt 2` in agreement.
      useRuntime("mtoc2_tensor_elemwise_real_fn");
      return `mtoc2_pow_real(${argsC[0]}, ${argsC[1]})`;
    }
    useRuntime("mtoc2_tensor_elemwise_real_fn");
    if (aMulti && bMulti) {
      return needsBroadcast(aN, bN)
        ? `mtoc2_tensor_power_bcast_tt(${argsC[0]}, ${argsC[1]})`
        : `mtoc2_tensor_power_tt(${argsC[0]}, ${argsC[1]})`;
    }
    if (aMulti) {
      return `mtoc2_tensor_power_ts(${argsC[0]}, ${argsC[1]})`;
    }
    // scalar .^ tensor — not commutative.
    return `mtoc2_tensor_power_st(${argsC[0]}, ${argsC[1]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const aN = argTypes[0] as NumericType;
    const bN = argTypes[1] as NumericType;
    const aMulti = isMultiElement(aN);
    const bMulti = isMultiElement(bN);
    if (!aMulti && !bMulti) {
      if (aN.isComplex || bN.isComplex) {
        useRuntime("mtoc2_cscalar");
        const promote = (j: string, c: boolean) =>
          c ? j : `mtoc2_cmake(${j}, 0.0)`;
        return `mtoc2_cpow(${promote(argsJs[0], aN.isComplex)}, ${promote(argsJs[1], bN.isComplex)})`;
      }
      return `Math.pow(${argsJs[0]}, ${argsJs[1]})`;
    }
    if (aN.isComplex || bN.isComplex) {
      throw new UnsupportedConstruct(
        `'power' complex-tensor emitJs not yet wired`
      );
    }
    useRuntime("mtoc2_tensor_elemwise_real_fn");
    if (aMulti && bMulti) {
      return needsBroadcast(aN, bN)
        ? `mtoc2_tensor_power_bcast_tt(${argsJs[0]}, ${argsJs[1]})`
        : `mtoc2_tensor_power_tt(${argsJs[0]}, ${argsJs[1]})`;
    }
    if (aMulti) {
      return `mtoc2_tensor_power_ts(${argsJs[0]}, ${argsJs[1]})`;
    }
    // scalar .^ tensor — non-commutative.
    return `mtoc2_tensor_power_st(${argsJs[0]}, ${argsJs[1]})`;
  },
  call({ args, argTypes }) {
    const aN = argTypes[0] as NumericType;
    const bN = argTypes[1] as NumericType;
    const aMulti = isMultiElement(aN);
    const bMulti = isMultiElement(bN);
    if (!aMulti && !bMulti) {
      if (aN.isComplex || bN.isComplex) {
        const av = args[0];
        const bv = args[1];
        const ax = isComplexValue(av) ? av : { re: Number(av), im: 0 };
        const bx = isComplexValue(bv) ? bv : { re: Number(bv), im: 0 };
        const out = mtoc2_cpow(ax, bx);
        if (out.im === 0 && !aN.isComplex && !bN.isComplex) return [out.re];
        return [out];
      }
      const av = typeof args[0] === "number" ? args[0] : Number(args[0]);
      const bv = typeof args[1] === "number" ? args[1] : Number(args[1]);
      return [Math.pow(av, bv)];
    }
    if (aN.isComplex || bN.isComplex) {
      throw new UnsupportedConstruct(
        `'power' complex-tensor 'call' not yet wired`
      );
    }
    if (aMulti && bMulti) {
      const at = args[0] as RuntimeTensor;
      const bt = args[1] as RuntimeTensor;
      const op = needsBroadcast(aN, bN)
        ? mtoc2_tensor_power_bcast_tt
        : mtoc2_tensor_power_tt;
      return [op(at, bt) as unknown as RuntimeTensor];
    }
    if (aMulti) {
      const at = args[0] as RuntimeTensor;
      const bv = typeof args[1] === "number" ? args[1] : Number(args[1]);
      return [mtoc2_tensor_power_ts(at, bv) as unknown as RuntimeTensor];
    }
    const av = typeof args[0] === "number" ? args[0] : Number(args[0]);
    const bt = args[1] as RuntimeTensor;
    return [mtoc2_tensor_power_st(av, bt) as unknown as RuntimeTensor];
  },
  elementwise: true,
};

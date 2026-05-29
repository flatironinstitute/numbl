/**
 * `norm(v)` / `norm(v, p)` — vector norms (real or complex).
 *
 * Scope today:
 *   - Scalar (real or complex): `abs(v)`. The optional `p` arg is
 *     accepted but ignored — matches numbl's `norm(3+4i, 1) = 5`.
 *   - 1-D vector (row or column, real or complex):
 *       - `norm(v)`                 = 2-norm (Euclidean)
 *       - `norm(v, p)` finite p > 0 = `(sum |x_i|^p)^(1/p)`
 *       - `norm(v, Inf)`            = `max |x_i|`
 *       - `norm(v, -Inf)`           = `min |x_i|`
 *       - `norm(v, 'fro')`          = 2-norm (frobenius == 2-norm on
 *                                     a vector; numbl uses this alias)
 *       - `norm(v, 'inf')`          = `+Inf` (case-insensitive)
 *   - Matrix / N-D tensor: not yet supported (numbl uses LAPACK SVD
 *     for the matrix 2-norm; the other matrix orders are
 *     finite-sum but need a per-axis walk that mtoc2's runtime
 *     doesn't expose yet).
 *
 * Result type is always real scalar `nonneg`.
 */
import {
  type NumericType,
  isMultiElement,
  isNumeric,
  isScalar,
  scalarDouble,
  signFromNumber,
  type Type,
  typeToString,
} from "../../../lowering/types.js";
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";
import { isComplexValue, type RuntimeTensor } from "../../../runtime/value.js";
import {
  mtoc2_cabs,
  mtoc2_norm2_real as jsNorm2Real,
  mtoc2_norm2_complex as jsNorm2Complex,
  mtoc2_norm_p_real as jsNormPReal,
  mtoc2_norm_p_complex as jsNormPComplex,
} from "../../runtime/snippets.gen.js";
import {
  exactDouble,
  exactComplex,
  exactComplexArray,
  exactRealArray,
} from "../_shared.js";

/** Resolve the optional second arg to a finite or infinite `p` value.
 *  Returns `2` when no second arg is present (the default 2-norm).
 *  Throws `TypeError` / `UnsupportedConstruct` on bad shape; returns
 *  `undefined` when the second arg is a runtime-only scalar (transfer
 *  has to fall through to the runtime helper without folding). */
function resolveP(argTypes: Type[]): number | undefined {
  if (argTypes.length < 2) return 2;
  const p = argTypes[1];
  if (p.kind === "Char" || p.kind === "String") {
    if (p.exact === undefined) {
      throw new UnsupportedConstruct(
        `'norm' order arg must be a literal; non-literal text is not supported`
      );
    }
    const m = p.exact.toLowerCase();
    if (m === "fro") return 2;
    if (m === "inf") return Infinity;
    throw new TypeError(
      `'norm' string order arg must be 'fro' or 'inf' (got '${p.exact}')`
    );
  }
  if (!isNumeric(p) || p.isComplex || !isScalar(p)) {
    throw new TypeError(
      `'norm' order arg must be a real scalar or 'fro' / 'inf' (got ${typeToString(p)})`
    );
  }
  const v = exactDouble(p);
  if (v === undefined) return undefined; // runtime-only p
  return v;
}

function requireSupportedShape(a: NumericType): "scalar" | "vector" {
  if (isScalar(a)) return "scalar";
  const isVecShape =
    a.dims.length === 2 &&
    ((a.dims[0].kind === "exact" && a.dims[0].value === 1) ||
      (a.dims[1].kind === "exact" && a.dims[1].value === 1));
  if (!isVecShape) {
    throw new UnsupportedConstruct(
      `'norm' input must be a scalar or vector (got ${typeToString(a)}); ` +
        `matrix-norm forms are not yet supported`
    );
  }
  return "vector";
}

function realFold(p: number, arr: Float64Array): number | undefined {
  if (arr.length === 0) return 0;
  let v: number;
  if (p === Infinity) {
    v = 0;
    for (let i = 0; i < arr.length; i++) {
      const x = Math.abs(arr[i]);
      if (x > v) v = x;
    }
  } else if (p === -Infinity) {
    v = Math.abs(arr[0]);
    for (let i = 1; i < arr.length; i++) {
      const x = Math.abs(arr[i]);
      if (x < v) v = x;
    }
  } else if (p === 1) {
    v = 0;
    for (let i = 0; i < arr.length; i++) v += Math.abs(arr[i]);
  } else if (p === 2) {
    let acc = 0;
    for (let i = 0; i < arr.length; i++) acc += arr[i] * arr[i];
    v = Math.sqrt(acc);
  } else {
    let acc = 0;
    for (let i = 0; i < arr.length; i++) acc += Math.pow(Math.abs(arr[i]), p);
    v = Math.pow(acc, 1 / p);
  }
  return Number.isFinite(v) ? v : undefined;
}

function complexFold(
  p: number,
  re: Float64Array,
  im: Float64Array
): number | undefined {
  if (re.length === 0) return 0;
  const abs = (i: number) => Math.hypot(re[i], im[i]);
  let v: number;
  if (p === Infinity) {
    v = 0;
    for (let i = 0; i < re.length; i++) {
      const x = abs(i);
      if (x > v) v = x;
    }
  } else if (p === -Infinity) {
    v = abs(0);
    for (let i = 1; i < re.length; i++) {
      const x = abs(i);
      if (x < v) v = x;
    }
  } else if (p === 2) {
    let acc = 0;
    for (let i = 0; i < re.length; i++) acc += re[i] * re[i] + im[i] * im[i];
    v = Math.sqrt(acc);
  } else if (p === 1) {
    v = 0;
    for (let i = 0; i < re.length; i++) v += abs(i);
  } else {
    let acc = 0;
    for (let i = 0; i < re.length; i++) acc += Math.pow(abs(i), p);
    v = Math.pow(acc, 1 / p);
  }
  return Number.isFinite(v) ? v : undefined;
}

/** C expression yielding the chosen `p` as a `double` literal. Uses
 *  `INFINITY` (from `<math.h>`, already pulled in by `tensor_norm`)
 *  for the unbounded cases. */
function pAsC(p: number): string {
  if (p === Infinity) return "INFINITY";
  if (p === -Infinity) return "-INFINITY";
  return String(p);
}

function pAsJs(p: number): string {
  if (p === Infinity) return "Infinity";
  if (p === -Infinity) return "-Infinity";
  return String(p);
}

export const norm: Builtin = {
  name: "norm",
  transfer(argTypes, nargout) {
    if (argTypes.length < 1 || argTypes.length > 2) {
      throw new TypeError(`'norm' expects 1..2 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'norm' does not support multi-output (nargout=${nargout})`
      );
    }
    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `'norm' arg must be a real or complex numeric (got ${typeToString(a)})`
      );
    }
    if (a.elem !== "double") {
      throw new TypeError(`'norm' arg must be double (got ${a.elem})`);
    }
    const shape = requireSupportedShape(a);
    const p = resolveP(argTypes);
    // Scalar: norm(s, *) == abs(s) — the p arg is ignored.
    if (shape === "scalar") {
      if (!a.isComplex) {
        const x = exactDouble(a);
        if (x !== undefined) {
          const v = Math.abs(x);
          if (Number.isFinite(v)) return [scalarDouble(signFromNumber(v), v)];
        }
        return [scalarDouble("nonneg")];
      }
      const cx = exactComplex(a);
      if (cx !== undefined) {
        const v = Math.hypot(cx.re, cx.im);
        if (Number.isFinite(v)) return [scalarDouble(signFromNumber(v), v)];
      }
      return [scalarDouble("nonneg")];
    }
    // Vector path.
    if (p !== undefined) {
      if (!a.isComplex) {
        const arr = exactRealArray(a);
        if (arr !== undefined) {
          const v = realFold(p, arr);
          if (v !== undefined) return [scalarDouble(signFromNumber(v), v)];
        }
      } else {
        const cx = exactComplexArray(a);
        if (cx !== undefined) {
          const v = complexFold(p, cx.re, cx.im);
          if (v !== undefined) return [scalarDouble(signFromNumber(v), v)];
        }
      }
    }
    return [scalarDouble("nonneg")];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    useRuntime("mtoc2_cscalar");
    if (isMultiElement(a)) {
      useRuntime("mtoc2_tensor_norm");
      const p = resolveP(argTypes);
      if (p === 2 && argTypes.length < 2) {
        return a.isComplex
          ? `mtoc2_norm2_complex(${argsC[0]})`
          : `mtoc2_norm2_real(${argsC[0]})`;
      }
      const pC = p !== undefined ? pAsC(p) : `(double)(${argsC[1]})`;
      return a.isComplex
        ? `mtoc2_norm_p_complex(${argsC[0]}, ${pC})`
        : `mtoc2_norm_p_real(${argsC[0]}, ${pC})`;
    }
    return a.isComplex ? `mtoc2_cabs(${argsC[0]})` : `fabs(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      useRuntime("mtoc2_tensor_norm");
      const p = resolveP(argTypes);
      if (p === 2 && argTypes.length < 2) {
        return a.isComplex
          ? `mtoc2_norm2_complex(${argsJs[0]})`
          : `mtoc2_norm2_real(${argsJs[0]})`;
      }
      const pJs = p !== undefined ? pAsJs(p) : argsJs[1];
      return a.isComplex
        ? `mtoc2_norm_p_complex(${argsJs[0]}, ${pJs})`
        : `mtoc2_norm_p_real(${argsJs[0]}, ${pJs})`;
    }
    if (a.isComplex) {
      useRuntime("mtoc2_cscalar");
      return `mtoc2_cabs(${argsJs[0]})`;
    }
    return `Math.abs(${argsJs[0]})`;
  },
  call({ args, argTypes }) {
    const a = argTypes[0] as NumericType;
    if (isMultiElement(a)) {
      const p = resolveP(argTypes);
      const fallbackP = p !== undefined ? p : Number(args[1]);
      if (fallbackP === 2 && argTypes.length < 2) {
        const fn = a.isComplex ? jsNorm2Complex : jsNorm2Real;
        return [fn(args[0] as RuntimeTensor)];
      }
      const fn = a.isComplex ? jsNormPComplex : jsNormPReal;
      return [fn(args[0] as RuntimeTensor, fallbackP)];
    }
    if (a.isComplex) {
      const v = args[0];
      const cx = isComplexValue(v) ? v : { re: Number(v), im: 0 };
      return [mtoc2_cabs(cx)];
    }
    const v = typeof args[0] === "number" ? args[0] : Number(args[0]);
    return [Math.abs(v)];
  },
};

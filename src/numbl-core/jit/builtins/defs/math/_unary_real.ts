/**
 * Shared scaffold for elementwise unary real-valued math builtins
 * (`cos`, `sin`, `sqrt`, `abs`, …).
 *
 * Mirrors `arithmetic/_elemwise.ts`'s binary factory: one transfer +
 * codegen pair, parametrized over the (`cFnReal`, `jsFn`, `signRule`,
 * `realDomainOk`) quadruple. Scalar path emits a bare C `<math.h>`
 * call; tensor path emits a per-name runtime helper that lives in
 * `runtime/tensor_unary_real_math.h`. Builtins whose real domain is
 * limited (`sqrt`, `log`, …) opt in to a real→complex lift via the
 * `complex.liftOnDomainMiss` flag instead of throwing at translate
 * time.
 *
 * Exact-fold rule: when every input element is exact AND every output
 * element is finite, attach the result as `exact` on the returned
 * type. Anything else drops the exact and the C side does the work.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  type NumericType,
  type Sign,
  scalarDouble,
  scalarComplex,
  tensorDouble,
  tensorComplex,
  tensorDoubleFromDims,
  tensorComplexFromDims,
  shapeNumel,
  signFromNumber,
  isScalar,
  isMultiElement,
  isNumeric,
  EXACT_ARRAY_MAX_ELEMENTS,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  requireRealDouble,
  requireRealOrComplex,
  exactDouble,
  exactRealArray,
  exactComplex,
  exactComplexArray,
} from "../_shared.js";
/** Builtin names with JS tensor kernels (`jit_tensor_<name>` in
 *  `tensor_unary_real_math.js`, `jit_tensor_<name>_complex` in
 *  `tensor_unary_complex_math.js`). */
const JS_TENSOR_UNARY = new Set([
  "cos",
  "sin",
  "tan",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "exp",
  "log",
  "log2",
  "log10",
  "sqrt",
  "realsqrt",
  "abs",
  "floor",
  "ceil",
  "fix",
  "round",
  "sign",
]);

const JS_TENSOR_UNARY_COMPLEX = new Set([
  "cos",
  "sin",
  "tan",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "exp",
  "log",
  "log2",
  "log10",
  "sqrt",
  "floor",
  "ceil",
  "fix",
  "round",
  "sign",
]);

export interface UnaryRealMathOpts {
  /** Source-level builtin name (also the runtime helper suffix). */
  name: string;
  /** C `<math.h>` function name for the scalar path (e.g. `"cos"`). */
  cFnReal: string;
  /** JS-side scalar fn for compile-time fold. */
  jsFn: (x: number) => number;
  /** Optional JS expression form (textual) for `emitJs`'s scalar real
   *  path. Defaults to `Math.${name}(arg)` which works for most names
   *  (sin/cos/tan/sqrt/exp/log/log2/log10/abs/atan/floor/ceil/sign).
   *  Override for `fix` (`Math.trunc`) and `round`
   *  (custom half-away-from-zero form). */
  jsExpr?: (arg: string) => string;
  /** Optional `jit_`-prefixed helper from `tensor_unary_real_math.js`
   *  for `emitJs`'s scalar real path. Activates that snippet and takes
   *  precedence over `jsExpr`. */
  jsFnReal?: string;
  /** Sign refinement on the result type. Called with the (validated)
   *  real-numeric input type. */
  signRule: (t: NumericType) => Sign;
  /** Optional real-domain predicate. `true` means the real input stays
   *  on the real path; `false` means it leaves the real domain (would
   *  produce NaN / -Inf / complex). When `false` AND `complex.liftOnDomainMiss`
   *  is set, the call lifts to the complex path (real-input,
   *  complex-output); otherwise the factory throws `TypeError`. */
  realDomainOk?: (t: NumericType) => boolean;
  /** Optional complex-input support. When set, complex scalars route
   *  through `cFnComplex` (a `jit_c*` helper); complex tensors
   *  route through `jit_tensor_<name>_complex`. `jsFnComplex`
   *  folds at the type-system layer when the input has an exact
   *  `{re, im}` carrier. `liftOnDomainMiss` makes the factory lift
   *  a real-typed input through the same complex path when
   *  `realDomainOk` returns false (e.g. `sqrt(-1)`). */
  complex?: {
    cFnComplex: string;
    jsFnComplex: (z: { re: number; im: number }) => { re: number; im: number };
    liftOnDomainMiss?: boolean;
  };
}

/** Sign rule for rounding-toward-zero builtins (`fix`, `round`, `ceil`,
 *  `floor`). Captures the "may collapse to zero" pattern: if a side of
 *  the number line can land on 0, its strict-sign input weakens to the
 *  corresponding non-strict sign. The flags say whether the operation's
 *  rounding direction can reach 0 from that side. */
export function roundingSignRule(
  positiveCanLand: boolean,
  negativeCanLand: boolean
): (t: NumericType) => Sign {
  return t => {
    if (t.sign === "positive" && positiveCanLand) return "nonneg";
    if (t.sign === "negative" && negativeCanLand) return "nonpositive";
    if (t.sign === "nonzero" && (positiveCanLand || negativeCanLand)) {
      return "unknown";
    }
    return t.sign;
  };
}

export function defineUnaryRealMath(opts: UnaryRealMathOpts): Builtin {
  const { name, cFnReal, jsFn, signRule, realDomainOk, complex } = opts;
  const jsExpr = opts.jsExpr ?? ((a: string) => `Math.${name}(${a})`);
  /** True when a real-typed input should route through the complex
   *  path because it leaves the real domain (e.g. `sqrt(-1)`,
   *  `log(-2)`). Requires both `realDomainOk` (predicate) and
   *  `complex.liftOnDomainMiss` (opt-in). Pure on `ty`, so emit / call
   *  can recompute it without extra state. */
  const liftRealToComplex = (ty: NumericType): boolean => {
    if (ty.isComplex) return false;
    if (realDomainOk === undefined) return false;
    if (realDomainOk(ty)) return false;
    return complex !== undefined && complex.liftOnDomainMiss === true;
  };
  return {
    name,
    transfer(argTypes, nargout) {
      if (argTypes.length !== 1) {
        throw new TypeError(
          `'${name}' expects 1 arg(s), got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      if (complex !== undefined) {
        requireRealOrComplex(argTypes[0], `'${name}' arg`);
      } else {
        requireRealDouble(argTypes[0], `'${name}' arg`);
      }
      const a = argTypes[0] as NumericType;
      // Complex result path: either input is already complex, or input
      // is real but leaves the real domain and we're configured to lift.
      const lifting = liftRealToComplex(a);
      if (a.isComplex || lifting) {
        if (isScalar(a)) {
          // Pick up the exact value in `{re, im}` form regardless of
          // which side fed us (complex carries `{re, im}`; real lift
          // projects `re=x, im=0`).
          const cx = a.isComplex
            ? exactComplex(a)
            : (() => {
                const ax = exactDouble(a);
                return ax !== undefined ? { re: ax, im: 0 } : undefined;
              })();
          if (cx !== undefined) {
            const v = complex!.jsFnComplex(cx);
            if (Number.isFinite(v.re) && Number.isFinite(v.im)) {
              return [scalarComplex(v)];
            }
          }
          return [scalarComplex()];
        }
        if (a.isComplex) {
          const cx = exactComplexArray(a);
          if (cx !== undefined && a.shape !== undefined) {
            const total = shapeNumel(a.shape);
            if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
              const re = new Float64Array(total);
              const im = new Float64Array(total);
              let allFinite = true;
              for (let i = 0; i < total; i++) {
                const v = complex!.jsFnComplex({
                  re: cx.re[i],
                  im: cx.im[i],
                });
                if (!Number.isFinite(v.re) || !Number.isFinite(v.im)) {
                  allFinite = false;
                  break;
                }
                re[i] = v.re;
                im[i] = v.im;
              }
              if (allFinite) return [tensorComplex(a.shape, { re, im })];
            }
          }
        } else {
          // Real-tensor lift: project each element through `jsFnComplex`
          // with im=0 to populate the exact complex result.
          const arr = exactRealArray(a);
          if (arr !== undefined && a.shape !== undefined) {
            const total = shapeNumel(a.shape);
            if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
              const re = new Float64Array(total);
              const im = new Float64Array(total);
              let allFinite = true;
              for (let i = 0; i < total; i++) {
                const v = complex!.jsFnComplex({ re: arr[i], im: 0 });
                if (!Number.isFinite(v.re) || !Number.isFinite(v.im)) {
                  allFinite = false;
                  break;
                }
                re[i] = v.re;
                im[i] = v.im;
              }
              if (allFinite) return [tensorComplex(a.shape, { re, im })];
            }
          }
        }
        return [tensorComplexFromDims(a.dims.slice())];
      }
      if (realDomainOk !== undefined && !realDomainOk(a)) {
        throw new TypeError(
          `'${name}' of input that may leave the real domain is not yet ` +
            `supported for real-typed input (would produce NaN / -Inf or ` +
            `a complex result). Guard upstream or make the input complex ` +
            `(e.g. '${name}(x + 0i)').`
        );
      }

      if (isScalar(a)) {
        const ax = exactDouble(a);
        if (ax !== undefined) {
          const v = jsFn(ax);
          if (Number.isFinite(v)) return [scalarDouble(signFromNumber(v), v)];
        }
        return [scalarDouble(signRule(a))];
      }

      if (a.shape !== undefined) {
        const arr = exactRealArray(a);
        const total = shapeNumel(a.shape);
        if (arr !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const out = new Float64Array(arr.length);
          let allFinite = true;
          for (let i = 0; i < arr.length; i++) {
            const v = jsFn(arr[i]);
            if (!Number.isFinite(v)) {
              allFinite = false;
              break;
            }
            out[i] = v;
          }
          if (allFinite) return [tensorDouble(a.shape, out)];
        }
      }
      const out = tensorDoubleFromDims(a.dims.slice());
      out.sign = signRule(a);
      return [out];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      const ty = argTypes[0] as NumericType;
      const lifting = liftRealToComplex(ty);
      if ((isNumeric(ty) && ty.isComplex) || lifting) {
        useRuntime("jit_cscalar");
        if (isMultiElement(ty)) {
          // Complex tensor helpers tolerate a real-tensor input
          // (`imag == NULL`), so the lift path passes the real tensor
          // straight through with no promote step.
          useRuntime("jit_tensor_unary_complex_math");
          return `jit_tensor_${name}_complex(${argsC[0]})`;
        }
        // Scalar lift: promote the `double` arg into `double _Complex`
        // via `jit_cmake(arg, 0.0)` to match the binary elemwise
        // convention. C99 would auto-promote here, but the explicit
        // form keeps the emitted C self-evidently complex-typed.
        const arg = lifting ? `jit_cmake(${argsC[0]}, 0.0)` : argsC[0];
        return `${complex!.cFnComplex}(${arg})`;
      }
      if (isMultiElement(ty)) {
        useRuntime("jit_tensor_unary_real_math");
        return `jit_tensor_${name}(${argsC[0]})`;
      }
      // `cFnReal` for scalar path. If it's a jit_-prefixed helper
      // (e.g. `jit_round_half_away`, `jit_signum`), it lives in
      // `tensor_unary_real_math.h`; activate that snippet so the
      // helper is declared even when no tensor path uses it.
      // libc math functions (cos, sin, sqrt, …) don't need this.
      if (cFnReal.startsWith("jit_")) {
        useRuntime("jit_tensor_unary_real_math");
      }
      return `${cFnReal}(${argsC[0]})`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const ty = argTypes[0] as NumericType;
      const lifting = liftRealToComplex(ty);
      if ((isNumeric(ty) && ty.isComplex) || lifting) {
        if (isMultiElement(ty)) {
          if (!JS_TENSOR_UNARY_COMPLEX.has(name)) {
            throw new UnsupportedConstruct(
              `'${name}' complex-tensor emitJs has no JS kernel registered`
            );
          }
          useRuntime("jit_tensor_unary_complex_math");
          useRuntime("jit_cscalar");
          return `jit_tensor_${name}_complex(${argsJs[0]})`;
        }
        useRuntime("jit_cscalar");
        // Real-scalar lift: JS has no implicit real→complex promotion,
        // so wrap explicitly in `{re, im}` via `jit_cmake`.
        const arg = lifting ? `jit_cmake(${argsJs[0]}, 0.0)` : argsJs[0];
        return `${complex!.cFnComplex}(${arg})`;
      }
      if (isMultiElement(ty)) {
        if (!JS_TENSOR_UNARY.has(name)) {
          throw new UnsupportedConstruct(
            `'${name}' tensor emitJs has no JS kernel registered`
          );
        }
        useRuntime("jit_tensor_unary_real_math");
        return `jit_tensor_${name}(${argsJs[0]})`;
      }
      if (opts.jsFnReal !== undefined) {
        useRuntime("jit_tensor_unary_real_math");
        return `${opts.jsFnReal}(${argsJs[0]})`;
      }
      return jsExpr(argsJs[0]);
    },
    elementwise: true,
  };
}

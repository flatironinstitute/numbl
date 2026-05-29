/**
 * Factory for elementwise numeric → logical predicate builtins
 * (`isnan`, `isinf`, `isfinite`, `logical`). One numeric argument
 * (real or complex); the result is a logical value of the same
 * shape (scalar → logical scalar, tensor → logical tensor of 1s / 0s).
 *
 * The per-element rule is supplied two ways:
 *  - real path: `cScalar` / `jsScalar` text + `jsFn` (compile fold,
 *    interpreter); real tensor uses `tensorHelper` (`mtoc2_tensor_*`).
 *  - complex path (optional): `cScalarComplex` / `jsScalarComplex`
 *    text + `jsFnComplex`; complex tensor uses
 *    `tensorHelperComplex` (`mtoc2_tensor_*_complex`).
 *
 * When `complex` is omitted, complex inputs are rejected by
 * `requireRealDouble`. Per the project rule (CLAUDE.md), new
 * predicates should declare the complex spec unless MATLAB itself
 * doesn't accept complex for the op.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  scalarLogical,
  tensorDouble,
  tensorDoubleFromDims,
  shapeNumel,
  isMultiElement,
  isScalar,
  EXACT_ARRAY_MAX_ELEMENTS,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import {
  requireRealDouble,
  requireRealOrComplex,
  exactDouble,
  exactRealArray,
  exactComplex,
  exactComplexArray,
} from "../_shared.js";
import type { Builtin } from "../../registry.js";
import {
  isTensor,
  isComplexValue,
  type RuntimeValue,
} from "../../../runtime/value.js";

/** Re-tag a double-typed result as logical (same Float64 buffer, the
 *  values are all 0 / 1). */
function asLogical(t: NumericType): NumericType {
  return { ...t, elem: "logical", sign: "nonneg" };
}

interface ComplexPredSpec {
  /** C scalar expression: given `(re, im)` text, return the per-element
   *  boolean expression. The factory wraps it in a ternary that maps
   *  to `1.0` / `0.0`. */
  cScalarComplex: (re: string, im: string) => string;
  /** JS scalar expression: same shape as `cScalarComplex` but for
   *  the js-aot path; operates on real-and-imag JS expressions. */
  jsScalarComplex: (re: string, im: string) => string;
  /** JS fold / interpreter rule on a `{re, im}` value. */
  jsFnComplex: (re: number, im: number) => boolean;
  /** Runtime tensor helper name (in `tensor_predicate_complex.h`). */
  tensorHelperComplex: string;
}

export interface UnaryPredOpts {
  name: string;
  /** C scalar expression given the arg's C text. */
  cScalar: (arg: string) => string;
  /** JS scalar expression given the arg's JS text. */
  jsScalar: (arg: string) => string;
  /** JS fold / interpreter rule. */
  jsFn: (x: number) => boolean;
  /** Runtime tensor helper name (in `tensor_predicate.h`). */
  tensorHelper: string;
  /** Optional complex-input support. Predicates that MATLAB accepts
   *  on complex (`isnan`, `isinf`, `isfinite`, …) declare this. */
  complex?: ComplexPredSpec;
}

export function defineUnaryPred(opts: UnaryPredOpts): Builtin {
  const { name, cScalar, jsScalar, jsFn, tensorHelper, complex } = opts;
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
      if (a.isComplex) {
        // Complex path (only reachable when `complex !== undefined`).
        if (isScalar(a)) {
          const cx = exactComplex(a);
          if (cx !== undefined)
            return [scalarLogical(complex!.jsFnComplex(cx.re, cx.im))];
          return [scalarLogical()];
        }
        const cx = exactComplexArray(a);
        if (cx !== undefined && a.shape !== undefined) {
          const total = shapeNumel(a.shape);
          if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
            const out = new Float64Array(total);
            for (let i = 0; i < total; i++) {
              out[i] = complex!.jsFnComplex(cx.re[i], cx.im[i]) ? 1 : 0;
            }
            return [asLogical(tensorDouble(a.shape, out))];
          }
        }
        return [asLogical(tensorDoubleFromDims(a.dims.slice()))];
      }
      // Real path.
      if (isScalar(a)) {
        const ex = exactDouble(a);
        if (ex !== undefined) return [scalarLogical(jsFn(ex))];
        return [scalarLogical()];
      }
      const arr = exactRealArray(a);
      if (arr !== undefined && a.shape !== undefined) {
        const total = shapeNumel(a.shape);
        if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
          const out = new Float64Array(total);
          for (let i = 0; i < total; i++) out[i] = jsFn(arr[i]) ? 1 : 0;
          return [asLogical(tensorDouble(a.shape, out))];
        }
      }
      return [asLogical(tensorDoubleFromDims(a.dims.slice()))];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      const a = argTypes[0] as NumericType;
      if (a.isComplex) {
        useRuntime("mtoc2_cscalar");
        if (isMultiElement(a)) {
          useRuntime("mtoc2_tensor_predicate");
          return `${complex!.tensorHelperComplex}(${argsC[0]})`;
        }
        const re = `creal(${argsC[0]})`;
        const im = `cimag(${argsC[0]})`;
        return `(${complex!.cScalarComplex(re, im)})`;
      }
      if (isMultiElement(a)) {
        useRuntime(tensorHelper);
        return `mtoc2_tensor_${name}(${argsC[0]})`;
      }
      return cScalar(argsC[0]);
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const a = argTypes[0] as NumericType;
      if (a.isComplex) {
        if (isMultiElement(a)) {
          useRuntime("mtoc2_tensor_predicate");
          return `${complex!.tensorHelperComplex}(${argsJs[0]})`;
        }
        // Scalar complex: `{re, im}` shape. Result is a bare JS bool.
        return `(${complex!.jsScalarComplex(`${argsJs[0]}.re`, `${argsJs[0]}.im`)})`;
      }
      if (isMultiElement(a)) {
        useRuntime(tensorHelper);
        return `mtoc2_tensor_${name}(${argsJs[0]})`;
      }
      return jsScalar(argsJs[0]);
    },
    call({ args, argTypes }) {
      const a = argTypes[0] as Type;
      const v = args[0];
      if (isMultiElement(a) && isTensor(v)) {
        const out = new Float64Array(v.data.length);
        if (v.imag !== undefined && complex !== undefined) {
          for (let i = 0; i < v.data.length; i++) {
            out[i] = complex.jsFnComplex(v.data[i], v.imag[i]) ? 1 : 0;
          }
        } else {
          for (let i = 0; i < v.data.length; i++) {
            out[i] = jsFn(v.data[i]) ? 1 : 0;
          }
        }
        const r: RuntimeValue = {
          mtoc2Tag: "tensor",
          shape: v.shape.slice(),
          data: out,
          isLogical: true,
        };
        return [r];
      }
      if (complex !== undefined && isComplexValue(v)) {
        return [complex.jsFnComplex(v.re, v.im)];
      }
      const n = typeof v === "number" ? v : Number(v);
      return [jsFn(n)];
    },
    elementwise: true,
  };
}

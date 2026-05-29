/**
 * Shared logic for `triu` / `tril` — the two builtins differ only by
 * the keep-predicate (column - row vs. row - column) and the runtime
 * helper they dispatch to. This module centralizes:
 *
 *   - argument validation (1..2 args, real-double tensor, k literal int)
 *   - `transfer` (exact-fold when input data fits)
 *   - hooks builders (emitC / emitJs / call) parameterized by the
 *     family-specific predicate + runtime helper.
 *
 * Mirrors `triPart` in numbl's
 * `interpreter/builtins/array-extras.ts`. Real and complex inputs
 * both supported; sparse inputs and rank > 2 are deferred / rejected.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  isNumeric,
  isScalar,
  scalarComplex,
  scalarDouble,
  signFromNumber,
  tensorComplex,
  tensorDouble,
  typeToString,
} from "../../../lowering/types.js";
import type { NumericType, Type } from "../../../lowering/types.js";
import type {
  Builtin,
  CallArgs,
  EmitCArgs,
  EmitJsArgs,
} from "../../registry.js";
import {
  exactComplex,
  exactComplexArray,
  exactDouble,
  exactRealArray,
} from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import { isTensor, makeTensor } from "../../../runtime/value.js";
import {
  mtoc2_tensor_triu as jsTriu,
  mtoc2_tensor_tril as jsTril,
  mtoc2_tensor_triu_complex as jsTriuComplex,
  mtoc2_tensor_tril_complex as jsTrilComplex,
} from "../../runtime/snippets.gen.js";

/** Resolve the optional `k` argument to a JS integer. Throws on
 *  dynamic / non-integer / non-scalar k. */
function resolveK(name: string, argTypes: Type[]): number {
  if (argTypes.length < 2) return 0;
  const t = argTypes[1];
  if (!isNumeric(t) || !isScalar(t) || t.isComplex) {
    throw new TypeError(
      `'${name}' second arg must be a scalar real integer (got ${typeToString(t)})`
    );
  }
  if (t.elem !== "double" && t.elem !== "logical") {
    throw new TypeError(
      `'${name}' second arg must be double or logical (got ${t.elem})`
    );
  }
  const kv = exactDouble(t);
  if (kv === undefined) {
    throw new UnsupportedConstruct(
      `'${name}' with a dynamic k argument is not yet supported (require a literal integer)`
    );
  }
  if (!Number.isFinite(kv) || !Number.isInteger(kv)) {
    throw new TypeError(`'${name}' k must be a finite integer (got ${kv})`);
  }
  return kv;
}

interface TriangularSpec {
  /** Source-level name (registry key). */
  name: string;
  /** C runtime helper name for real-input (e.g. `"mtoc2_tensor_triu"`). */
  cHelper: string;
  /** C runtime helper name for complex-input
   *  (e.g. `"mtoc2_tensor_triu_complex"`). */
  cHelperComplex: string;
  /** Keep predicate matching numbl's `triPart` (column-major i,j). */
  keep(i: number, j: number, k: number): boolean;
  /** JS-side helper for real input. */
  jsHelper: (a: RuntimeTensor, k: number) => unknown;
  /** JS-side helper for complex input. */
  jsHelperComplex: (a: RuntimeTensor, k: number) => unknown;
}

export function defineTriangular(spec: TriangularSpec): Builtin {
  const { name, cHelper, cHelperComplex, keep, jsHelper, jsHelperComplex } =
    spec;

  function transfer(argTypes: Type[], nargout: number): Type[] {
    if (argTypes.length < 1 || argTypes.length > 2) {
      throw new TypeError(
        `'${name}' expects 1..2 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'${name}' does not support multi-output (nargout=${nargout})`
      );
    }

    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `'${name}' first arg must be numeric (got ${typeToString(a)})`
      );
    }
    if (a.elem !== "double" && a.elem !== "logical") {
      throw new TypeError(
        `'${name}' first arg must be a double or logical tensor (got ${a.elem})`
      );
    }
    const k = resolveK(name, argTypes);

    // ── Scalar input ─────────────────────────────────────────
    if (isScalar(a)) {
      // Numbl behavior: scalar is treated as a 1×1 matrix; keep iff
      // predicate(0,0,k). Result stays scalar.
      if (a.isComplex) {
        const cx = exactComplex(a);
        if (cx !== undefined) {
          return [scalarComplex(keep(0, 0, k) ? cx : { re: 0, im: 0 })];
        }
        return [
          keep(0, 0, k) ? scalarComplex() : scalarComplex({ re: 0, im: 0 }),
        ];
      }
      const v = exactDouble(a);
      if (v !== undefined) {
        const kept = keep(0, 0, k) ? v : 0;
        return [scalarDouble(signFromNumber(kept), kept)];
      }
      if (keep(0, 0, k)) {
        return [scalarDouble(a.sign)];
      }
      // Predicate rejects this element → result is the literal 0.
      return [scalarDouble("zero", 0)];
    }

    // ── Tensor input ─────────────────────────────────────────
    if (a.dims.length !== 2) {
      throw new UnsupportedConstruct(
        `'${name}' requires a 2-D operand (got ${a.dims.length}-D)`
      );
    }
    if (a.shape === undefined) {
      throw new UnsupportedConstruct(
        `'${name}' of a tensor with unknown shape is not yet supported`
      );
    }

    const [rows, cols] = a.shape;
    const shape = [rows, cols];
    const total = rows * cols;
    if (a.isComplex) {
      const cx = exactComplexArray(a);
      if (cx !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
        const re = new Float64Array(total);
        const im = new Float64Array(total);
        for (let j = 0; j < cols; j++) {
          for (let i = 0; i < rows; i++) {
            if (keep(i, j, k)) {
              const idx = i + j * rows;
              re[idx] = cx.re[idx];
              im[idx] = cx.im[idx];
            }
          }
        }
        return [tensorComplex(shape, { re, im })];
      }
      return [tensorComplex(shape)];
    }
    const arr = exactRealArray(a);
    if (arr !== undefined && total <= EXACT_ARRAY_MAX_ELEMENTS) {
      const data = new Float64Array(total);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          if (keep(i, j, k)) {
            const idx = i + j * rows;
            data[idx] = arr[idx];
          }
        }
      }
      return [tensorDouble(shape, data)];
    }
    return [tensorDouble(shape)];
  }

  function emitC({ argsC, argTypes, useRuntime }: EmitCArgs): string {
    const a = argTypes[0] as NumericType;
    const k = resolveK(name, argTypes);

    if (isScalar(a)) {
      if (a.isComplex) {
        useRuntime("mtoc2_cscalar");
        return keep(0, 0, k) ? argsC[0] : `mtoc2_cmake(0.0, 0.0)`;
      }
      return keep(0, 0, k) ? argsC[0] : "0.0";
    }

    if (a.isComplex) {
      useRuntime("mtoc2_tensor_triangular");
      return `${cHelperComplex}(${argsC[0]}, ${k}L)`;
    }
    useRuntime("mtoc2_tensor_triangular");
    return `${cHelper}(${argsC[0]}, ${k}L)`;
  }

  function emitJs({ argsJs, argTypes, useRuntime }: EmitJsArgs): string {
    const a = argTypes[0] as NumericType;
    const k = resolveK(name, argTypes);

    if (isScalar(a)) {
      if (a.isComplex) {
        useRuntime("mtoc2_cscalar");
        return keep(0, 0, k) ? argsJs[0] : `mtoc2_cmake(0, 0)`;
      }
      return keep(0, 0, k) ? argsJs[0] : "0";
    }

    if (a.isComplex) {
      useRuntime("mtoc2_tensor_triangular");
      return `${cHelperComplex}(${argsJs[0]}, ${k})`;
    }
    useRuntime("mtoc2_tensor_triangular");
    return `${cHelper}(${argsJs[0]}, ${k})`;
  }

  function call({ args, argTypes }: CallArgs) {
    const a = argTypes[0] as NumericType;
    const k = resolveK(name, argTypes);

    if (isScalar(a)) {
      if (keep(0, 0, k)) return [args[0]];
      if (a.isComplex) return [{ re: 0, im: 0 }];
      return [0];
    }

    if (!isTensor(args[0])) {
      throw new TypeError(
        `'${name}' runtime arg has tensor type but runtime value isn't a tensor`
      );
    }
    const t = args[0] as RuntimeTensor;
    const rows = t.shape[0];
    const cols = t.shape[1];
    // Empty matrix degenerate: just return a fresh empty tensor with
    // the same shape (matches numbl's behavior — `triPart` allocates
    // a zero-length data array and returns it).
    if (rows === 0 || cols === 0) {
      return [makeTensor([rows, cols], new Float64Array(0))];
    }
    if (a.isComplex) {
      return [jsHelperComplex(t, k) as unknown as RuntimeTensor];
    }
    return [jsHelper(t, k) as unknown as RuntimeTensor];
  }

  return { name, transfer, emitC, emitJs, call };
}

// Re-export the JS helpers so the per-builtin files don't each have to
// re-import them.
export { jsTriu, jsTril, jsTriuComplex, jsTrilComplex };

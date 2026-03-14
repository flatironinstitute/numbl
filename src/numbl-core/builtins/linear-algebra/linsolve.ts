/**
 * linsolve(A, B) — solve the linear system A * X = B.
 *
 * If A is square (m × m), uses LU factorisation (dgesv / dgetrf+solve).
 * If A is non-square, uses QR / LQ factorisation (dgels / dgeqrf+solve):
 *   overdetermined  (m > n): least-squares solution minimising ||A*X - B||₂
 *   underdetermined (m < n): minimum-norm solution minimising ||X||₂
 *
 * Both the native LAPACK addon and the ts-lapack TypeScript fallback are
 * supported; the native addon is preferred when available.
 */

import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import { out, toF64, unknownMatrix } from "./check-helpers.js";
import { isNum, isTensor, isFullyUnknown } from "../../lowering/itemTypes.js";

// ── LAPACK helpers ────────────────────────────────────────────────────────────

/**
 * Dispatch linsolve to the native addon (preferred) or ts-lapack fallback.
 * Returns null only when neither bridge exposes `linsolve` (should not happen
 * in practice since ts-lapack always has it).
 */
export function linsolveLapack(
  A: FloatXArrayType,
  m: number,
  n: number,
  B: FloatXArrayType,
  nrhs: number
): Float64Array | null {
  const bridge = getEffectiveBridge("linsolve", "linsolve");
  if (!bridge?.linsolve) return null;

  return bridge.linsolve(toF64(A), m, n, toF64(B), nrhs);
}

/**
 * Dispatch complex linsolve to the native addon (preferred) or ts-lapack fallback.
 * The ts-lapack fallback throws — native addon is required for complex linsolve.
 */
export function linsolveComplexLapack(
  ARe: FloatXArrayType,
  AIm: FloatXArrayType,
  m: number,
  n: number,
  BRe: FloatXArrayType,
  BIm: FloatXArrayType,
  nrhs: number
): { re: Float64Array; im: Float64Array } {
  const bridge = getEffectiveBridge("linsolveComplex", "linsolveComplex");
  if (!bridge?.linsolveComplex)
    throw new RuntimeError(
      "linsolveComplex: no bridge available (should not happen)"
    );

  return bridge.linsolveComplex(
    toF64(ARe),
    toF64(AIm),
    m,
    n,
    toF64(BRe),
    toF64(BIm),
    nrhs
  );
}

// ── Builtin registration ──────────────────────────────────────────────────────

export function registerLinsolve(): void {
  /**
   * X = linsolve(A, B)
   *
   * A  — m×n real matrix
   * B  — m×p real matrix (or m-vector when p = 1)
   * X  — n×p solution matrix
   *
   * Square:        exact solution via LU
   * Overdetermined: least-squares via QR
   * Underdetermined: minimum-norm via LQ
   */
  register("linsolve", [
    {
      check: (argTypes, nargout) => {
        if (argTypes.length !== 2 || nargout !== 1) return null;

        const A = argTypes[0];
        const B = argTypes[1];

        // Unknown inputs → unknown output shape
        if (isFullyUnknown(A) || isFullyUnknown(B)) return out(unknownMatrix());

        // Scalar A or B is coerced to 1×1 at runtime
        if (isNum(A) === true || isNum(B) === true) return out(unknownMatrix());

        // Must be 2-D tensors (real or complex)
        if (isTensor(A) !== true || isTensor(B) !== true) return null;

        return out(unknownMatrix());
      },

      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("linsolve requires 2 arguments");

        // Scalars are treated as 1×1 matrices so linsolve(a, b) works for
        // scalar a/b and for row-vector A with scalar b (underdetermined case).
        const rawA = args[0];
        const rawB = args[1];

        const A = isRuntimeNumber(rawA)
          ? RTV.tensor(new FloatXArray([rawA]), [1, 1])
          : rawA;
        const B = isRuntimeNumber(rawB)
          ? RTV.tensor(new FloatXArray([rawB]), [1, 1])
          : rawB;

        if (!isRuntimeTensor(A) || !isRuntimeTensor(B))
          throw new RuntimeError(
            "linsolve: arguments must be numeric matrices"
          );

        const [m, n] = tensorSize2D(A);
        const [Bm, p] = tensorSize2D(B);

        if (Bm !== m)
          throw new RuntimeError(
            "linsolve: A and B must have the same number of rows"
          );

        if (A.imag || B.imag) {
          // Complex solve — native addon required
          const ARe = A.data;
          const AIm = A.imag ?? new FloatXArray(A.data.length);
          const BRe = B.data;
          const BIm = B.imag ?? new FloatXArray(B.data.length);
          const X = linsolveComplexLapack(ARe, AIm, m, n, BRe, BIm, p);
          return RTV.tensor(
            new FloatXArray(X.re),
            [n, p],
            new FloatXArray(X.im)
          );
        }

        const X = linsolveLapack(A.data, m, n, B.data, p);
        if (!X) throw new RuntimeError("linsolve: LAPACK bridge unavailable");

        return RTV.tensor(new FloatXArray(X), [n, p]);
      },
    },
  ]);
}

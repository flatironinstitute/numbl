/**
 * SVD (Singular Value Decomposition) builtin function
 */

import {
  colMajorIndex,
  RTV,
  RuntimeError,
  tensorSize2D,
} from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import {
  buildDiagMatrix,
  isMatrixLike,
  out,
  parseEconArg,
  parseEconArgRuntime,
  toF64,
  unknownMatrix,
} from "../check-helpers.js";

// ── LAPACK helper ─────────────────────────────────────────────────────────────

/**
 * SVD decomposition via LAPACK (dgesdd or dgesvd).
 * Returns null if the bridge or its svd method is unavailable.
 */
function svdLapack(
  data: FloatXArrayType,
  m: number,
  n: number,
  econ: boolean,
  computeUV: boolean
): { U?: Float64Array; S: Float64Array; V?: Float64Array } | null {
  const bridge = getEffectiveBridge("svd", "svd");
  if (!bridge || !bridge.svd) return null;
  return bridge.svd(toF64(data), m, n, econ, computeUV);
}

/**
 * Complex SVD via LAPACK zgesdd.
 */
function svdLapackComplex(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  m: number,
  n: number,
  econ: boolean,
  computeUV: boolean
): {
  S: Float64Array;
  URe?: Float64Array;
  UIm?: Float64Array;
  VRe?: Float64Array;
  VIm?: Float64Array;
} | null {
  const bridge = getLapackBridge();
  if (!bridge || !bridge.svdComplex) return null;
  return bridge.svdComplex(toF64(dataRe), toF64(dataIm), m, n, econ, computeUV);
}

export function registerSvd(): void {
  /**
   * Singular Value Decomposition.
   * Supports: S = svd(A)              — singular values only
   *           [U, S, V] = svd(A)      — full SVD: A = U*S*V'
   *           [U, S, V] = svd(A, 0)   — economy SVD
   *           [U, S, V] = svd(A, 'econ') — economy SVD
   */
  register("svd", [
    {
      check: (argTypes, nargout) => {
        if (
          nargout < 1 ||
          nargout > 3 ||
          argTypes.length < 1 ||
          argTypes.length > 2
        )
          return null;
        if (parseEconArg(argTypes[1]) === null) return null;
        if (!isMatrixLike(argTypes[0])) return null;
        if (nargout === 1) return out(unknownMatrix());
        return out(unknownMatrix(), unknownMatrix(), unknownMatrix());
      },
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("svd requires at least 1 argument");
        const A = args[0];

        if (isRuntimeNumber(A)) {
          // Scalar case
          const val = Math.abs(A);
          if (nargout === 1) {
            return RTV.tensor(new FloatXArray([val]), [1, 1]);
          }
          const U = RTV.tensor(new FloatXArray([A >= 0 ? 1 : -1]), [1, 1]);
          const S = RTV.tensor(new FloatXArray([val]), [1, 1]);
          const V = RTV.tensor(new FloatXArray([1]), [1, 1]);
          return [U, S, V];
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("svd: argument must be numeric");

        const econ = parseEconArgRuntime(args[1]);

        const [m, n] = tensorSize2D(A);
        const k = Math.min(m, n);

        // ── Complex SVD via LAPACK ──────────────────────────────────────────
        if (A.imag) {
          const lapackResult = svdLapackComplex(
            A.data,
            A.imag,
            m,
            n,
            econ,
            nargout === 3
          );
          if (!lapackResult) {
            throw new RuntimeError(
              "svd: complex SVD requires LAPACK (build the native addon)"
            );
          }
          if (nargout === 1) {
            return RTV.tensor(new FloatXArray(lapackResult.S), [k, 1]);
          }
          const uCols = econ ? k : m;
          const vCols = econ ? k : n;
          const U = RTV.tensor(
            new FloatXArray(lapackResult.URe!),
            [m, uCols],
            new FloatXArray(lapackResult.UIm!)
          );
          const S = buildDiagMatrix(
            lapackResult.S,
            undefined,
            econ ? k : [m, n]
          );
          const V = RTV.tensor(
            new FloatXArray(lapackResult.VRe!),
            [n, vCols],
            new FloatXArray(lapackResult.VIm!)
          );
          return [U, S, V];
        }

        // ── Real SVD via LAPACK ─────────────────────────────────────────────
        const lapackResult = svdLapack(A.data, m, n, econ, nargout === 3);
        if (lapackResult) {
          if (nargout === 1) {
            // Return singular values as column vector
            return RTV.tensor(new FloatXArray(lapackResult.S), [k, 1]);
          }
          // Full or economy decomposition
          const uCols = econ ? k : m;
          const vCols = econ ? k : n;
          const U = RTV.tensor(new FloatXArray(lapackResult.U!), [m, uCols]);
          const S = buildDiagMatrix(
            lapackResult.S,
            undefined,
            econ ? k : [m, n]
          );
          const V = RTV.tensor(new FloatXArray(lapackResult.V!), [n, vCols]);
          return [U, S, V];
        }

        // ── JS fallback ───────────────────────────────────────────────────
        // For now, we'll use a simple power iteration method for the largest
        // singular value, or throw an error suggesting LAPACK for full SVD
        if (nargout > 1) {
          throw new RuntimeError(
            "svd: full decomposition requires LAPACK (build the native addon)"
          );
        }

        // Compute singular values only using eigenvalues of A'*A
        // This is less efficient but works for the singular values only case
        const ATA = computeATA(A.data, m, n);
        const eigenvalues = powerIterationEigenvalues(ATA, n, k);
        const singularValues = eigenvalues.map(ev =>
          Math.sqrt(Math.max(0, ev))
        );

        return RTV.tensor(new FloatXArray(singularValues), [k, 1]);
      },
    },
  ]);
}

// ── JavaScript fallback helpers ──────────────────────────────────────────────

/**
 * Compute A'*A where A is m×n in column-major format.
 * Returns an n×n matrix in column-major format.
 */
function computeATA(
  A_data: FloatXArrayType,
  m: number,
  n: number
): FloatXArrayType {
  const result = new FloatXArray(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i <= j; i++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += A_data[colMajorIndex(k, i, m)] * A_data[colMajorIndex(k, j, m)];
      }
      result[colMajorIndex(i, j, n)] = sum;
      if (i !== j) result[colMajorIndex(j, i, n)] = sum; // symmetric
    }
  }
  return result;
}

/**
 * Simple power iteration to find the largest eigenvalues of a symmetric matrix.
 * This is a fallback and not particularly efficient, but works for small matrices.
 */
function powerIterationEigenvalues(
  A_data: FloatXArrayType,
  n: number,
  numEigenvalues: number
): number[] {
  const eigenvalues: number[] = [];
  const A_copy = new FloatXArray(A_data); // work on a copy

  for (let ev = 0; ev < numEigenvalues; ev++) {
    // Power iteration to find largest eigenvalue
    const v = new FloatXArray(n);
    for (let i = 0; i < n; i++) v[i] = Math.random();

    // Normalize
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < n; i++) v[i] /= norm;

    let lambda = 0;
    for (let iter = 0; iter < 100; iter++) {
      // A * v
      const Av = new FloatXArray(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          sum += A_copy[colMajorIndex(i, j, n)] * v[j];
        }
        Av[i] = sum;
      }

      // Compute Rayleigh quotient: v' * A * v
      lambda = 0;
      for (let i = 0; i < n; i++) lambda += v[i] * Av[i];

      // Normalize Av
      norm = 0;
      for (let i = 0; i < n; i++) norm += Av[i] * Av[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-14) break; // converged to zero
      for (let i = 0; i < n; i++) v[i] = Av[i] / norm;
    }

    eigenvalues.push(lambda);

    // Deflate: A := A - lambda * v * v'
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        A_copy[colMajorIndex(i, j, n)] -= lambda * v[i] * v[j];
      }
    }
  }

  return eigenvalues;
}

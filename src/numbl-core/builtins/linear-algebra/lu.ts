/**
 * LU factorization builtin function
 *
 * Supports:
 *   [L,U]   = lu(A)                — permuted L, upper U, A = L*U
 *   [L,U,P] = lu(A)                — unit lower L, upper U, permutation matrix P, P*A = L*U
 *   [L,U,P] = lu(A,'vector')       — P as permutation vector, A(P,:) = L*U
 */

import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeTensor,
  RuntimeValue,
} from "../../runtime/types.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import {
  matrix,
  out,
  parseStringArgLower,
  toF64,
  unknownMatrix,
} from "./check-helpers.js";
import { isNum, isTensor, isFullyUnknown } from "../../lowering/itemTypes.js";

// ── LAPACK helper ─────────────────────────────────────────────────────────────

function luLapack(
  data: FloatXArrayType,
  m: number,
  n: number
): { LU: Float64Array; ipiv: Int32Array } | null {
  const bridge = getEffectiveBridge("lu", "lu");
  if (!bridge?.lu) return null;
  return bridge.lu(toF64(data), m, n);
}

function luLapackComplex(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  m: number,
  n: number
): { LURe: Float64Array; LUIm: Float64Array; ipiv: Int32Array } | null {
  const bridge = getEffectiveBridge("lu", "luComplex");
  if (!bridge?.luComplex) return null;
  return bridge.luComplex(toF64(dataRe), toF64(dataIm), m, n);
}

/**
 * Convert 1-based LAPACK ipiv (sequential swaps) to a 0-based permutation vector.
 * ipiv[i] means row i was swapped with row ipiv[i]-1 (1-based).
 */
function ipivToPermVector(ipiv: Int32Array, m: number): Int32Array {
  const perm = new Int32Array(m);
  for (let i = 0; i < m; i++) perm[i] = i;
  for (let i = 0; i < ipiv.length; i++) {
    const j = ipiv[i] - 1; // convert to 0-based
    if (j !== i) {
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
  }
  return perm;
}

/**
 * Parse the optional outputForm argument ('matrix' or 'vector').
 * Returns 'matrix' (default), 'vector', or null if invalid.
 */
function parseOutputForm(
  arg: RuntimeValue | undefined
): "matrix" | "vector" | null {
  if (arg === undefined) return "matrix";
  const s = parseStringArgLower(arg);
  if (s === "vector") return "vector";
  if (s === "matrix") return "matrix";
  return null;
}

export function registerLu(): void {
  register("lu", [
    {
      check: (argTypes, nargout) => {
        if (nargout < 1 || nargout > 3) return null;
        if (argTypes.length < 1 || argTypes.length > 2) return null;

        const A = argTypes[0];

        // Validate optional outputForm arg (only meaningful for nargout >= 3)
        if (argTypes.length === 2) {
          const fmt = argTypes[1];
          if (
            !isFullyUnknown(fmt) &&
            fmt.kind !== "String" &&
            fmt.kind !== "Char"
          )
            return null;
        }

        if (nargout === 1) {
          // Single output not standard for lu — but MATLAB returns packed LU.
          // We'll support it as returning the packed matrix (same shape as A).
          if (isFullyUnknown(A)) return out(unknownMatrix());
          if (isNum(A) === true) return out(matrix([1, 1]));
          if (isTensor(A) !== true) return null;
          return out(unknownMatrix());
        }

        if (nargout === 2) {
          // [L, U] = lu(A)
          if (isFullyUnknown(A)) return out(unknownMatrix(), unknownMatrix());
          if (isNum(A) === true) return out(matrix([1, 1]), matrix([1, 1]));
          if (isTensor(A) !== true) return null;
          return out(unknownMatrix(), unknownMatrix());
        }

        // nargout === 3: [L, U, P] = lu(A) or lu(A, 'vector')
        if (isFullyUnknown(A))
          return out(unknownMatrix(), unknownMatrix(), unknownMatrix());
        if (isNum(A) === true)
          return out(matrix([1, 1]), matrix([1, 1]), matrix([1, 1]));
        if (isTensor(A) !== true || (A.kind === "Tensor" && A.isComplex))
          return null;
        // P is m×m (matrix form) or 1×m / m×1 (vector form) — use unknown for simplicity
        return out(unknownMatrix(), unknownMatrix(), unknownMatrix());
      },

      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("lu requires at least 1 argument");

        const A = args[0];

        // Parse outputForm
        const outputForm =
          args.length >= 2 ? parseOutputForm(args[1]) : "matrix";
        if (outputForm === null)
          throw new RuntimeError("lu: outputForm must be 'matrix' or 'vector'");

        // Scalar case
        if (isRuntimeNumber(A)) {
          const val = A as number;
          if (nargout <= 2) {
            const L = RTV.tensor(new FloatXArray([1]), [1, 1]);
            const U = RTV.tensor(new FloatXArray([val]), [1, 1]);
            if (nargout === 1) return L; // packed form = just the value
            return [L, U];
          }
          // nargout === 3
          const L = RTV.tensor(new FloatXArray([1]), [1, 1]);
          const U = RTV.tensor(new FloatXArray([val]), [1, 1]);
          if (outputForm === "vector") {
            return [L, U, RTV.num(1)];
          }
          const P = RTV.tensor(new FloatXArray([1]), [1, 1]);
          return [L, U, P];
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("lu: argument must be numeric");

        const [m, n] = tensorSize2D(A);
        const k = Math.min(m, n);
        const isComplex = A.imag !== undefined;

        // Call LAPACK (real or complex)
        let LU_re: FloatXArrayType;
        let LU_im: FloatXArrayType | undefined;
        let ipiv: Int32Array;

        if (isComplex) {
          const result = luLapackComplex(A.data, A.imag!, m, n);
          if (!result)
            throw new RuntimeError(
              "lu: complex LU requires the native LAPACK addon"
            );
          LU_re = result.LURe;
          LU_im = result.LUIm;
          ipiv = result.ipiv;
        } else {
          const result = luLapack(A.data, m, n);
          if (!result)
            throw new RuntimeError("lu: LAPACK bridge not available");
          LU_re = result.LU;
          ipiv = result.ipiv;
        }

        // Extract U (upper triangular, k×n)
        const U_re = new FloatXArray(k * n);
        const U_im = isComplex ? new FloatXArray(k * n) : undefined;
        for (let j = 0; j < n; j++) {
          const imax = Math.min(j, k - 1);
          for (let i = 0; i <= imax; i++) {
            U_re[i + j * k] = LU_re[i + j * m];
            if (U_im && LU_im) U_im[i + j * k] = LU_im[i + j * m];
          }
        }

        if (nargout === 1) {
          return RTV.tensor(
            new FloatXArray(LU_re),
            [m, n],
            LU_im ? new FloatXArray(LU_im) : undefined
          );
        }

        // Build permutation vector (0-based)
        const perm = ipivToPermVector(ipiv, m);

        if (nargout === 2) {
          // [L, U] = lu(A): L is permuted lower triangular such that A = L*U
          const L_unit_re = new FloatXArray(m * k);
          const L_unit_im = isComplex ? new FloatXArray(m * k) : undefined;
          for (let j = 0; j < k; j++) {
            L_unit_re[j + j * m] = 1;
            for (let i = j + 1; i < m; i++) {
              L_unit_re[i + j * m] = LU_re[i + j * m];
              if (L_unit_im && LU_im) L_unit_im[i + j * m] = LU_im[i + j * m];
            }
          }

          // Apply inverse permutation: L_permuted = P' * L_unit
          const L_re = new FloatXArray(m * k);
          const L_im = isComplex ? new FloatXArray(m * k) : undefined;
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < k; j++) {
              L_re[perm[i] + j * m] = L_unit_re[i + j * m];
              if (L_im && L_unit_im)
                L_im[perm[i] + j * m] = L_unit_im[i + j * m];
            }
          }

          return [
            RTV.tensor(L_re, [m, k], L_im),
            RTV.tensor(U_re, [k, n], U_im),
          ];
        }

        // nargout === 3: [L, U, P] = lu(A)
        const L_re = new FloatXArray(m * k);
        const L_im = isComplex ? new FloatXArray(m * k) : undefined;
        for (let j = 0; j < k; j++) {
          L_re[j + j * m] = 1;
          for (let i = j + 1; i < m; i++) {
            L_re[i + j * m] = LU_re[i + j * m];
            if (L_im && LU_im) L_im[i + j * m] = LU_im[i + j * m];
          }
        }

        if (outputForm === "vector") {
          const P_data = new FloatXArray(m);
          for (let i = 0; i < m; i++) {
            P_data[i] = perm[i] + 1;
          }
          return [
            RTV.tensor(L_re, [m, k], L_im),
            RTV.tensor(U_re, [k, n], U_im),
            RTV.tensor(P_data, [m, 1]),
          ];
        }

        const P_data = new FloatXArray(m * m);
        for (let i = 0; i < m; i++) {
          P_data[i + perm[i] * m] = 1;
        }

        return [
          RTV.tensor(L_re, [m, k], L_im),
          RTV.tensor(U_re, [k, n], U_im),
          RTV.tensor(P_data, [m, m]),
        ];
      },
    },
  ]);
}

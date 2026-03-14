/**
 * pinv (Moore-Penrose pseudoinverse) builtin function
 */

import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { register, builtinSingle } from "../registry.js";
import { toF64 } from "./check-helpers.js";

export function registerPinv(): void {
  register(
    "pinv",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("pinv requires 1 or 2 arguments");

      const A = args[0];

      // Scalar case
      if (isRuntimeNumber(A)) {
        return RTV.num(A === 0 ? 0 : 1 / A);
      }

      if (!isRuntimeTensor(A))
        throw new RuntimeError("pinv: argument must be numeric");

      const [m, n] = tensorSize2D(A);
      const k = Math.min(m, n);

      // Use SVD to compute pseudoinverse: pinv(A) = V * diag(1./s) * U'
      const bridge = getLapackBridge();
      if (!bridge || !bridge.svd) {
        // JS fallback: use the formula pinv(A) = (A'*A)^-1 * A' for tall matrices
        // or A' * (A*A')^-1 for wide matrices
        return pinvFallback(A.data, m, n);
      }

      const svdResult = bridge.svd(toF64(A.data), m, n, true, true);
      if (!svdResult || !svdResult.U || !svdResult.V)
        throw new RuntimeError("pinv: SVD computation failed");

      const { U, S, V } = svdResult;

      // Determine tolerance
      const tol =
        args.length >= 2
          ? isRuntimeNumber(args[1])
            ? args[1]
            : 0
          : Math.max(m, n) * S[0] * 2.220446049250313e-16;

      // Compute pinv = V * diag(1/s_i) * U' for s_i > tol
      // U is m x k, S is k-vector, V is n x k
      // pinv is n x m
      const result = new FloatXArray(n * m);

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          let sum = 0;
          for (let l = 0; l < k; l++) {
            if (S[l] > tol) {
              // V[i,l] * (1/S[l]) * U[j,l]
              sum += V[l * n + i] * (1 / S[l]) * U[l * m + j];
            }
          }
          result[j * n + i] = sum;
        }
      }

      return RTV.tensor(result, [n, m]);
    })
  );
}

/**
 * JS fallback for pinv using normal equations.
 * For m >= n: pinv(A) = (A'A)^-1 A'
 * For m < n:  pinv(A) = A' (AA')^-1
 */
function pinvFallback(
  data: FloatXArrayType,
  m: number,
  n: number
): ReturnType<typeof RTV.tensor> {
  // Simple implementation using the fact that for full-rank matrices:
  // For tall: pinv = (A'A)\A'
  // For wide: pinv = A'/(AA')
  // For general: use iterative approach

  // For zero matrix
  let allZero = true;
  for (let i = 0; i < data.length; i++) {
    if (data[i] !== 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return RTV.tensor(new FloatXArray(n * m), [n, m]);
  }

  // Compute A^T
  const AT = new FloatXArray(n * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      AT[i * n + j] = data[j * m + i];
    }
  }

  if (m >= n) {
    // Compute A'A (n x n)
    const ATA = new FloatXArray(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < m; k++) {
          sum += data[i * m + k] * data[j * m + k];
        }
        ATA[j * n + i] = sum;
      }
    }

    // Solve (A'A) X = A' using Gauss-Jordan
    const augmented = new FloatXArray(n * (n + m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        augmented[j * n + i] = ATA[j * n + i];
      }
      for (let j = 0; j < m; j++) {
        augmented[(n + j) * n + i] = AT[j * n + i];
      }
    }

    // Gauss-Jordan elimination
    for (let col = 0; col < n; col++) {
      // Partial pivoting
      let maxVal = Math.abs(augmented[col * n + col]);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(augmented[col * n + row]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        for (let j = 0; j < n + m; j++) {
          const tmp = augmented[j * n + col];
          augmented[j * n + col] = augmented[j * n + maxRow];
          augmented[j * n + maxRow] = tmp;
        }
      }

      const pivot = augmented[col * n + col];
      if (Math.abs(pivot) < 1e-14) continue;

      for (let j = 0; j < n + m; j++) {
        augmented[j * n + col] /= pivot;
      }

      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = augmented[col * n + row];
        for (let j = 0; j < n + m; j++) {
          augmented[j * n + row] -= factor * augmented[j * n + col];
        }
      }
    }

    // Extract result (n x m)
    const result = new FloatXArray(n * m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        result[j * n + i] = augmented[(n + j) * n + i];
      }
    }
    return RTV.tensor(result, [n, m]);
  } else {
    // m < n: pinv = A' * (AA')^-1
    // Compute AA' (m x m)
    const AAT = new FloatXArray(m * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += data[k * m + i] * data[k * m + j];
        }
        AAT[j * m + i] = sum;
      }
    }

    // Invert AA' using Gauss-Jordan
    const augmented = new FloatXArray(m * 2 * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        augmented[j * m + i] = AAT[j * m + i];
      }
      augmented[(m + i) * m + i] = 1;
    }

    for (let col = 0; col < m; col++) {
      let maxVal = Math.abs(augmented[col * m + col]);
      let maxRow = col;
      for (let row = col + 1; row < m; row++) {
        const val = Math.abs(augmented[col * m + row]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        for (let j = 0; j < 2 * m; j++) {
          const tmp = augmented[j * m + col];
          augmented[j * m + col] = augmented[j * m + maxRow];
          augmented[j * m + maxRow] = tmp;
        }
      }

      const pivot = augmented[col * m + col];
      if (Math.abs(pivot) < 1e-14) continue;

      for (let j = 0; j < 2 * m; j++) {
        augmented[j * m + col] /= pivot;
      }

      for (let row = 0; row < m; row++) {
        if (row === col) continue;
        const factor = augmented[col * m + row];
        for (let j = 0; j < 2 * m; j++) {
          augmented[j * m + row] -= factor * augmented[j * m + col];
        }
      }
    }

    // Extract (AA')^-1 (m x m)
    const AATinv = new FloatXArray(m * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        AATinv[j * m + i] = augmented[(m + j) * m + i];
      }
    }

    // result = A' * (AA')^-1, which is n x m
    const result = new FloatXArray(n * m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < m; k++) {
          // AT[i, k] * AATinv[k, j]
          sum += AT[k * n + i] * AATinv[j * m + k];
        }
        result[j * n + i] = sum;
      }
    }
    return RTV.tensor(result, [n, m]);
  }
}

/**
 * eig (Eigenvalues and eigenvectors) builtin function
 */

import {
  colMajorIndex,
  RTV,
  RuntimeError,
  RuntimeValue,
  tensorSize2D,
} from "../../runtime/index.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import {
  isMatrixLike,
  isOptionalStringArg,
  out,
  parseStringArgLower,
  toF64,
  unknownMatrix,
} from "./check-helpers.js";
import { type ItemType } from "../../lowering/itemTypes.js";

// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * Parse optional string arguments for eig():
 *   'nobalance' / 'balance'
 *   'vector' / 'matrix'
 * Returns null if an argument is invalid.
 */
function parseEigOptions(argTypes: ItemType[]): boolean {
  for (let i = 1; i < argTypes.length; i++) {
    if (!isOptionalStringArg(argTypes[i])) return false;
  }
  return true;
}

function parseEigOptionsRuntime(args: RuntimeValue[]): {
  balance: boolean;
  outputForm: "vector" | "matrix";
} {
  let balance = true;
  let outputForm: "vector" | "matrix" = "matrix";

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (isRuntimeString(arg) || isRuntimeChar(arg)) {
      const val = parseStringArgLower(arg);
      if (val === "nobalance") balance = false;
      else if (val === "balance") balance = true;
      else if (val === "vector") outputForm = "vector";
      else if (val === "matrix") outputForm = "matrix";
    }
  }
  return { balance, outputForm };
}

// ── LAPACK helper ────────────────────────────────────────────────────────────

function eigLapack(
  data: Float64Array,
  n: number,
  computeVL: boolean,
  computeVR: boolean,
  balance: boolean
): {
  wr: Float64Array;
  wi: Float64Array;
  VL?: Float64Array;
  VR?: Float64Array;
} | null {
  const bridge = getEffectiveBridge("eig", "eig");
  if (!bridge.eig) return null;
  return bridge.eig(data, n, computeVL, computeVR, balance);
}

function eigLapackComplex(
  dataRe: Float64Array,
  dataIm: Float64Array,
  n: number,
  computeVL: boolean,
  computeVR: boolean
): {
  wRe: Float64Array;
  wIm: Float64Array;
  VLRe?: Float64Array;
  VLIm?: Float64Array;
  VRRe?: Float64Array;
  VRIm?: Float64Array;
} | null {
  const bridge = getEffectiveBridge("eigComplex", "eigComplex");
  if (!bridge.eigComplex) return null;
  return bridge.eigComplex(dataRe, dataIm, n, computeVL, computeVR);
}

// ── registration ─────────────────────────────────────────────────────────────

export function registerEig(): void {
  register("eig", [
    {
      check: (argTypes, nargout) => {
        if (nargout < 1 || nargout > 3) return null;
        if (argTypes.length < 1 || argTypes.length > 3) return null;
        if (!parseEigOptions(argTypes)) return null;
        if (!isMatrixLike(argTypes[0])) return null;
        // Eigenvalues are generally complex
        const c = unknownMatrix(true);
        if (nargout === 1) return out(c);
        if (nargout === 2) return out(c, c);
        return out(c, c, c);
      },

      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("eig requires at least 1 argument");
        const A = args[0];

        const { balance, outputForm } = parseEigOptionsRuntime(args);

        // ── Scalar case ──────────────────────────────────────────────────
        if (isRuntimeNumber(A)) {
          const val = A;
          if (nargout === 1) {
            return RTV.num(val);
          }
          const V = RTV.tensor(new FloatXArray([1]), [1, 1]);
          const eigval = RTV.num(val);
          if (nargout === 2) {
            if (outputForm === "vector") {
              return [V, eigval];
            }
            const D = RTV.tensor(new FloatXArray([val]), [1, 1]);
            return [V, D];
          }
          // nargout === 3
          const D =
            outputForm === "vector"
              ? eigval
              : RTV.tensor(new FloatXArray([val]), [1, 1]);
          const W = RTV.tensor(new FloatXArray([1]), [1, 1]);
          return [V, D, W];
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("eig: argument must be numeric");

        const [m, n] = tensorSize2D(A);
        if (m !== n)
          throw new RuntimeError("eig: input must be a square matrix");

        const computeVL = nargout >= 3;
        const computeVR = nargout >= 2;

        // ── Complex input path ───────────────────────────────────────────
        if (A.imag) {
          const result = eigLapackComplex(
            toF64(A.data),
            toF64(A.imag),
            n,
            computeVL,
            computeVR
          );
          if (!result) {
            throw new RuntimeError(
              "eig: complex eig requires the native LAPACK addon"
            );
          }

          const { wRe, wIm, VLRe, VLIm, VRRe, VRIm } = result;

          // Check if result is actually all-real
          let hasComplex = false;
          for (let i = 0; i < n; i++) {
            if (Math.abs(wIm[i]) > 0) {
              hasComplex = true;
              break;
            }
          }

          if (nargout === 1) {
            if (hasComplex) {
              return RTV.tensor(
                new FloatXArray(wRe),
                [n, 1],
                new FloatXArray(wIm)
              );
            }
            return RTV.tensor(new FloatXArray(wRe), [n, 1]);
          }

          // Build V (right eigenvectors)
          let Vout;
          if (computeVR && VRRe && VRIm) {
            if (hasComplex) {
              Vout = RTV.tensor(
                new FloatXArray(VRRe),
                [n, n],
                new FloatXArray(VRIm)
              );
            } else {
              Vout = RTV.tensor(new FloatXArray(VRRe), [n, n]);
            }
          } else {
            Vout = RTV.tensor(new FloatXArray(n * n), [n, n]);
          }

          // Build D
          let Dout;
          if (outputForm === "vector") {
            if (hasComplex) {
              Dout = RTV.tensor(
                new FloatXArray(wRe),
                [n, 1],
                new FloatXArray(wIm)
              );
            } else {
              Dout = RTV.tensor(new FloatXArray(wRe), [n, 1]);
            }
          } else {
            const dReal = new FloatXArray(n * n);
            for (let i = 0; i < n; i++) {
              dReal[colMajorIndex(i, i, n)] = wRe[i];
            }
            if (hasComplex) {
              const dImag = new FloatXArray(n * n);
              for (let i = 0; i < n; i++) {
                dImag[colMajorIndex(i, i, n)] = wIm[i];
              }
              Dout = RTV.tensor(dReal, [n, n], dImag);
            } else {
              Dout = RTV.tensor(dReal, [n, n]);
            }
          }

          if (nargout === 2) {
            return [Vout, Dout];
          }

          // Build W (left eigenvectors)
          let Wout;
          if (computeVL && VLRe && VLIm) {
            if (hasComplex) {
              Wout = RTV.tensor(
                new FloatXArray(VLRe),
                [n, n],
                new FloatXArray(VLIm)
              );
            } else {
              Wout = RTV.tensor(new FloatXArray(VLRe), [n, n]);
            }
          } else {
            Wout = RTV.tensor(new FloatXArray(n * n), [n, n]);
          }

          return [Vout, Dout, Wout];
        }

        // ── Real input path ──────────────────────────────────────────────
        const result = eigLapack(
          toF64(A.data),
          n,
          computeVL,
          computeVR,
          balance
        );
        if (!result) {
          throw new RuntimeError("eig: LAPACK bridge not available");
        }

        const { wr, wi, VL, VR } = result;

        // Check if any eigenvalues are complex
        let hasComplex = false;
        for (let i = 0; i < n; i++) {
          if (Math.abs(wi[i]) > 0) {
            hasComplex = true;
            break;
          }
        }

        // ── nargout === 1: return eigenvalue vector ──────────────────────
        if (nargout === 1) {
          if (hasComplex) {
            return RTV.tensor(new FloatXArray(wr), [n, 1], new FloatXArray(wi));
          }
          return RTV.tensor(new FloatXArray(wr), [n, 1]);
        }

        // ── Build eigenvector matrix V (right eigenvectors) ──────────────
        let Vout;
        if (computeVR && VR) {
          Vout = buildEigenvectorMatrix(VR, wi, n, hasComplex);
        } else {
          Vout = RTV.tensor(new FloatXArray(n * n), [n, n]);
        }

        // ── Build D (eigenvalue matrix or vector) ────────────────────────
        let Dout;
        if (outputForm === "vector") {
          if (hasComplex) {
            Dout = RTV.tensor(new FloatXArray(wr), [n, 1], new FloatXArray(wi));
          } else {
            Dout = RTV.tensor(new FloatXArray(wr), [n, 1]);
          }
        } else {
          // Diagonal matrix
          const dReal = new FloatXArray(n * n);
          for (let i = 0; i < n; i++) {
            dReal[colMajorIndex(i, i, n)] = wr[i];
          }
          if (hasComplex) {
            const dImag = new FloatXArray(n * n);
            for (let i = 0; i < n; i++) {
              dImag[colMajorIndex(i, i, n)] = wi[i];
            }
            Dout = RTV.tensor(dReal, [n, n], dImag);
          } else {
            Dout = RTV.tensor(dReal, [n, n]);
          }
        }

        if (nargout === 2) {
          return [Vout, Dout];
        }

        // ── nargout === 3: also build W (left eigenvectors) ──────────────
        let Wout;
        if (computeVL && VL) {
          Wout = buildEigenvectorMatrix(VL, wi, n, hasComplex);
        } else {
          Wout = RTV.tensor(new FloatXArray(n * n), [n, n]);
        }

        return [Vout, Dout, Wout];
      },
    },
  ]);
}

/**
 * Build a complex eigenvector matrix from DGEEV's packed real format.
 *
 * DGEEV stores eigenvectors for complex conjugate pairs as:
 *   Column j:   real part of eigenvector for eigenvalue j
 *   Column j+1: imaginary part of eigenvector for eigenvalue j
 * The eigenvector for eigenvalue j is   V(:,j) + i*V(:,j+1)
 * The eigenvector for eigenvalue j+1 is V(:,j) - i*V(:,j+1)
 */
function buildEigenvectorMatrix(
  packedV: Float64Array,
  wi: Float64Array,
  n: number,
  hasComplex: boolean
) {
  if (!hasComplex) {
    return RTV.tensor(new FloatXArray(packedV), [n, n]);
  }

  const realPart = new FloatXArray(n * n);
  const imagPart = new FloatXArray(n * n);

  let j = 0;
  while (j < n) {
    if (Math.abs(wi[j]) === 0) {
      // Real eigenvalue — column j is real
      for (let i = 0; i < n; i++) {
        realPart[colMajorIndex(i, j, n)] = packedV[colMajorIndex(i, j, n)];
        // imagPart stays 0
      }
      j++;
    } else {
      // Complex conjugate pair at j and j+1
      for (let i = 0; i < n; i++) {
        const re = packedV[colMajorIndex(i, j, n)];
        const im = packedV[colMajorIndex(i, j + 1, n)];
        // Eigenvector for eigenvalue j: re + i*im
        realPart[colMajorIndex(i, j, n)] = re;
        imagPart[colMajorIndex(i, j, n)] = im;
        // Eigenvector for eigenvalue j+1: re - i*im
        realPart[colMajorIndex(i, j + 1, n)] = re;
        imagPart[colMajorIndex(i, j + 1, n)] = -im;
      }
      j += 2;
    }
  }

  return RTV.tensor(realPart, [n, n], imagPart);
}

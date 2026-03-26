/**
 * Cholesky factorization builtin function
 *
 * Supports:
 *   R = chol(A)                    — upper triangular R, A = R'*R
 *   R = chol(A, 'upper')           — same as above
 *   L = chol(A, 'lower')           — lower triangular L, A = L*L'
 *   [R, flag] = chol(A)            — flag=0 if positive definite, else index of failure
 *   [R, flag] = chol(A, triangle)  — with triangle option
 *   [R, flag, P] = chol(S)         — sparse only; also returns permutation matrix P
 *   [R, flag, p] = chol(S,'vector')— sparse only; permutation as vector
 *   [R, flag, P] = chol(S, triangle, outputForm) — sparse only; with both options
 */

import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeTensor,
  RuntimeSparseMatrix,
  RuntimeValue,
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
} from "../check-helpers.js";
import { IType, isTensor, isFullyUnknown } from "../../lowering/itemTypes.js";
import { sparseToDense } from "../sparse-arithmetic.js";

// ── LAPACK helpers ───────────────────────────────────────────────────────────

function cholLapack(
  data: FloatXArrayType,
  n: number,
  upper: boolean
): { R: Float64Array; info: number } | null {
  const bridge = getEffectiveBridge("chol", "chol");
  if (!bridge?.chol) return null;
  return bridge.chol(toF64(data), n, upper);
}

function cholLapackComplex(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  n: number,
  upper: boolean
): { RRe: Float64Array; RIm: Float64Array; info: number } | null {
  const bridge = getEffectiveBridge("chol", "cholComplex");
  if (!bridge?.cholComplex) return null;
  return bridge.cholComplex(toF64(dataRe), toF64(dataIm), n, upper);
}

/**
 * Parse the optional triangle argument ('upper' or 'lower').
 * Returns 'upper' (default) or 'lower', or null if invalid.
 */
function parseTriangleArg(
  arg: RuntimeValue | undefined
): "upper" | "lower" | null {
  if (arg === undefined) return "upper";
  const s = parseStringArgLower(arg);
  if (s === "upper") return "upper";
  if (s === "lower") return "lower";
  return null;
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

/**
 * Classify a string argument as triangle, outputForm, or invalid.
 * Used when a single string arg could be either.
 */
function classifyStringArg(
  arg: RuntimeValue
):
  | { kind: "triangle"; value: "upper" | "lower" }
  | { kind: "outputForm"; value: "matrix" | "vector" }
  | null {
  const s = parseStringArgLower(arg);
  if (s === "upper" || s === "lower") return { kind: "triangle", value: s };
  if (s === "matrix" || s === "vector") return { kind: "outputForm", value: s };
  return null;
}

/** Check if a type could be sparse */
function isSparseOrUnknown(
  t: import("../../lowering/itemTypes.js").ItemType
): boolean {
  return isFullyUnknown(t) || t.kind === "SparseMatrix";
}

export function registerChol(): void {
  register("chol", [
    {
      check: (argTypes, nargout) => {
        if (nargout < 1 || nargout > 3) return null;
        if (argTypes.length < 1 || argTypes.length > 3) return null;
        // Validate string args (positions 1 and 2)
        if (!isOptionalStringArg(argTypes[1])) return null;
        if (!isOptionalStringArg(argTypes[2])) return null;
        // Accept matrix-like or sparse
        const a0 = argTypes[0];
        if (!isMatrixLike(a0) && a0.kind !== "SparseMatrix") return null;
        // 3 outputs requires sparse (or unknown at compile time)
        if (nargout === 3) {
          if (!isSparseOrUnknown(a0) && isTensor(a0)) return null;
        }
        if (nargout === 1) return out(unknownMatrix());
        if (nargout === 2) return out(unknownMatrix(), IType.num());
        return out(unknownMatrix(), IType.num(), unknownMatrix());
      },

      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("chol requires at least 1 argument");

        let A = args[0];
        const inputIsSparse = isRuntimeSparseMatrix(A);

        // 3 outputs only for sparse
        if (nargout >= 3 && !inputIsSparse) {
          throw new RuntimeError(
            "Third output only available for sparse matrices."
          );
        }

        // Densify sparse input
        if (inputIsSparse) {
          A = sparseToDense(A as RuntimeSparseMatrix);
        }

        // Parse string arguments: can be triangle, outputForm, or both
        let triangle: "upper" | "lower" = "upper";
        let outputForm: "matrix" | "vector" = "matrix";

        if (args.length === 2) {
          // Single string arg: could be triangle or outputForm
          const classified = classifyStringArg(args[1]);
          if (classified === null)
            throw new RuntimeError(
              "chol: invalid option; expected 'upper', 'lower', 'matrix', or 'vector'"
            );
          if (classified.kind === "triangle") {
            triangle = classified.value;
          } else {
            outputForm = classified.value;
          }
        } else if (args.length === 3) {
          // Two string args: first is triangle, second is outputForm
          const tri = parseTriangleArg(args[1]);
          if (tri === null)
            throw new RuntimeError("chol: triangle must be 'upper' or 'lower'");
          triangle = tri;
          const form = parseOutputForm(args[2]);
          if (form === null)
            throw new RuntimeError(
              "chol: outputForm must be 'matrix' or 'vector'"
            );
          outputForm = form;
        }

        const upper = triangle === "upper";

        // Scalar case
        if (isRuntimeNumber(A)) {
          const val = A as number;
          if (val <= 0) {
            if (nargout >= 2) {
              // Return partial result with flag
              const R = RTV.tensor(new FloatXArray([0]), [1, 1]);
              return [R, RTV.num(1)];
            }
            throw new RuntimeError("chol: Matrix must be positive definite.");
          }
          const r = Math.sqrt(val);
          if (nargout >= 2) {
            return [RTV.tensor(new FloatXArray([r]), [1, 1]), RTV.num(0)];
          }
          return RTV.tensor(new FloatXArray([r]), [1, 1]);
        }

        if (!isRuntimeTensor(A))
          throw new RuntimeError("chol: argument must be numeric");

        const [m, n] = tensorSize2D(A);
        if (m !== n) throw new RuntimeError("chol: Matrix must be square.");

        const isComplex = A.imag !== undefined;

        let R_re: FloatXArrayType;
        let R_im: FloatXArrayType | undefined;
        let info_val: number;

        if (isComplex) {
          const result = cholLapackComplex(A.data, A.imag!, n, upper);
          if (!result)
            throw new RuntimeError(
              "chol: complex Cholesky requires the native LAPACK addon"
            );
          R_re = result.RRe;
          R_im = result.RIm;
          info_val = result.info;
        } else {
          const result = cholLapack(A.data, n, upper);
          if (!result)
            throw new RuntimeError("chol: LAPACK bridge not available");
          R_re = result.R;
          info_val = result.info;
        }

        // Helper to build identity permutation output (no AMD reordering)
        const buildPermOutput = (): RuntimeValue => {
          if (outputForm === "vector") {
            const p = new FloatXArray(n);
            for (let i = 0; i < n; i++) p[i] = i + 1; // 1-based
            return RTV.tensor(p, [n, 1]);
          }
          // Identity permutation matrix
          const P = new FloatXArray(n * n);
          for (let i = 0; i < n; i++) P[i + i * n] = 1;
          return RTV.tensor(P, [n, n]);
        };

        if (nargout >= 2) {
          if (info_val > 0) {
            // Return partial result up to row/col (info_val-1)
            const k = info_val - 1;
            let partialR: RuntimeValue;
            if (nargout >= 3) {
              // 3-output (sparse convention): R is k×n (upper) or n×k (lower)
              if (upper) {
                const partial_re = new FloatXArray(k * n);
                const partial_im = isComplex
                  ? new FloatXArray(k * n)
                  : undefined;
                for (let j = 0; j < n; j++) {
                  const imax = Math.min(j, k - 1);
                  for (let i = 0; i <= imax; i++) {
                    partial_re[i + j * k] = (R_re as FloatXArrayType)[
                      i + j * n
                    ];
                    if (partial_im && R_im)
                      partial_im[i + j * k] = (R_im as FloatXArrayType)[
                        i + j * n
                      ];
                  }
                }
                partialR = RTV.tensor(partial_re, [k, n], partial_im);
              } else {
                const partial_re = new FloatXArray(n * k);
                const partial_im = isComplex
                  ? new FloatXArray(n * k)
                  : undefined;
                for (let j = 0; j < k; j++) {
                  for (let i = j; i < n; i++) {
                    partial_re[i + j * n] = (R_re as FloatXArrayType)[
                      i + j * n
                    ];
                    if (partial_im && R_im)
                      partial_im[i + j * n] = (R_im as FloatXArrayType)[
                        i + j * n
                      ];
                  }
                }
                partialR = RTV.tensor(partial_re, [n, k], partial_im);
              }
              return [partialR, RTV.num(info_val), buildPermOutput()];
            }
            // 2-output (dense convention): R is k×k
            if (upper) {
              const partial_re = new FloatXArray(k * k);
              const partial_im = isComplex ? new FloatXArray(k * k) : undefined;
              for (let j = 0; j < k; j++) {
                for (let i = 0; i <= j; i++) {
                  partial_re[i + j * k] = (R_re as FloatXArrayType)[i + j * n];
                  if (partial_im && R_im)
                    partial_im[i + j * k] = (R_im as FloatXArrayType)[
                      i + j * n
                    ];
                }
              }
              partialR = RTV.tensor(partial_re, [k, k], partial_im);
            } else {
              const partial_re = new FloatXArray(k * k);
              const partial_im = isComplex ? new FloatXArray(k * k) : undefined;
              for (let j = 0; j < k; j++) {
                for (let i = j; i < k; i++) {
                  partial_re[i + j * k] = (R_re as FloatXArrayType)[i + j * n];
                  if (partial_im && R_im)
                    partial_im[i + j * k] = (R_im as FloatXArrayType)[
                      i + j * n
                    ];
                }
              }
              partialR = RTV.tensor(partial_re, [k, k], partial_im);
            }
            return [partialR, RTV.num(info_val)];
          }
          // Success
          const R = RTV.tensor(
            new FloatXArray(R_re as ArrayLike<number>),
            [n, n],
            R_im ? new FloatXArray(R_im as ArrayLike<number>) : undefined
          );
          if (nargout >= 3) {
            return [R, RTV.num(0), buildPermOutput()];
          }
          return [R, RTV.num(0)];
        }

        // Single output — error if not positive definite
        if (info_val > 0)
          throw new RuntimeError("chol: Matrix must be positive definite.");

        return RTV.tensor(
          new FloatXArray(R_re as ArrayLike<number>),
          [n, n],
          R_im ? new FloatXArray(R_im as ArrayLike<number>) : undefined
        );
      },
    },
  ]);
}

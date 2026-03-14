/**
 * Cholesky factorization builtin function
 *
 * Supports:
 *   R = chol(A)                — upper triangular R, A = R'*R
 *   R = chol(A, 'upper')       — same as above
 *   L = chol(A, 'lower')       — lower triangular L, A = L*L'
 *   [R, flag] = chol(A)        — flag=0 if positive definite, else index of failure
 *   [R, flag] = chol(A, triangle) — with triangle option
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
import {
  IType,
  isNum,
  isTensor,
  isFullyUnknown,
} from "../../lowering/itemTypes.js";

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

export function registerChol(): void {
  register("chol", [
    {
      check: (argTypes, nargout) => {
        if (nargout < 1 || nargout > 2) return null;
        if (argTypes.length < 1 || argTypes.length > 2) return null;

        const A = argTypes[0];

        // Validate optional triangle arg
        if (argTypes.length === 2) {
          const tri = argTypes[1];
          if (
            !isFullyUnknown(tri) &&
            tri.kind !== "String" &&
            tri.kind !== "Char"
          )
            return null;
        }

        if (nargout === 1) {
          if (isFullyUnknown(A)) return out(unknownMatrix());
          if (isNum(A) === true) return out(matrix([1, 1]));
          if (isTensor(A) !== true) return null;
          return out(unknownMatrix());
        }

        // nargout === 2: [R, flag]
        if (isFullyUnknown(A)) return out(unknownMatrix(), IType.num());
        if (isNum(A) === true) return out(matrix([1, 1]), IType.num());
        if (isTensor(A) !== true) return null;
        return out(unknownMatrix(), IType.num());
      },

      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("chol requires at least 1 argument");

        const A = args[0];

        // Parse triangle argument
        const triangle = args.length >= 2 ? parseTriangleArg(args[1]) : "upper";
        if (triangle === null)
          throw new RuntimeError("chol: triangle must be 'upper' or 'lower'");
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

        if (nargout >= 2) {
          if (info_val > 0) {
            // Return partial result up to row/col (info_val-1)
            const k = info_val - 1;
            if (upper) {
              // Return R(1:k, 1:k) — upper-left k×k block of R
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
              return [
                RTV.tensor(partial_re, [k, k], partial_im),
                RTV.num(info_val),
              ];
            } else {
              // Return L(1:k, 1:k) — upper-left k×k block of L
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
              return [
                RTV.tensor(partial_re, [k, k], partial_im),
                RTV.num(info_val),
              ];
            }
          }
          // Success
          return [
            RTV.tensor(
              new FloatXArray(R_re as ArrayLike<number>),
              [n, n],
              R_im ? new FloatXArray(R_im as ArrayLike<number>) : undefined
            ),
            RTV.num(0),
          ];
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

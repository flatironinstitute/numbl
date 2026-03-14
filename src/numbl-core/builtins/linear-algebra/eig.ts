/**
 * eig (Eigenvalues and eigenvectors) builtin function
 */

import {
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
  buildDiagMatrix,
  buildEigenvectorMatrix,
  isMatrixLike,
  isOptionalStringArg,
  maybeComplexTensor,
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

          if (nargout === 1) {
            return maybeComplexTensor(wRe, [n, 1], wIm);
          }

          const Vout =
            computeVR && VRRe && VRIm
              ? maybeComplexTensor(VRRe, [n, n], VRIm)
              : RTV.tensor(new FloatXArray(n * n), [n, n]);

          const Dout =
            outputForm === "vector"
              ? maybeComplexTensor(wRe, [n, 1], wIm)
              : buildDiagMatrix(wRe, wIm, n);

          if (nargout === 2) return [Vout, Dout];

          const Wout =
            computeVL && VLRe && VLIm
              ? maybeComplexTensor(VLRe, [n, n], VLIm)
              : RTV.tensor(new FloatXArray(n * n), [n, n]);

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
        const hasComplex = wi.some(v => v !== 0);

        // ── nargout === 1: return eigenvalue vector ──────────────────────
        if (nargout === 1) {
          return maybeComplexTensor(wr, [n, 1], wi);
        }

        // ── Build eigenvector matrix V (right eigenvectors) ──────────────
        const Vout =
          computeVR && VR
            ? buildEigenvectorMatrix(VR, wi, n, hasComplex)
            : RTV.tensor(new FloatXArray(n * n), [n, n]);

        // ── Build D (eigenvalue matrix or vector) ────────────────────────
        const Dout =
          outputForm === "vector"
            ? maybeComplexTensor(wr, [n, 1], wi)
            : buildDiagMatrix(wr, wi, n);

        if (nargout === 2) return [Vout, Dout];

        // ── nargout === 3: also build W (left eigenvectors) ──────────────
        const Wout =
          computeVL && VL
            ? buildEigenvectorMatrix(VL, wi, n, hasComplex)
            : RTV.tensor(new FloatXArray(n * n), [n, n]);

        return [Vout, Dout, Wout];
      },
    },
  ]);
}

/**
 * qz (Generalized Schur / QZ factorization) builtin function
 *
 * [AA,BB,Q,Z] = qz(A,B)        — generalized Schur decomposition
 * [AA,BB,Q,Z,V,W] = qz(A,B)    — also compute generalized eigenvectors
 * [...] = qz(A,B,'real')        — real decomposition (default for real matrices)
 * [...] = qz(A,B,'complex')     — complex decomposition
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
  RuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import {
  out,
  isMatrixLike,
  isOptionalStringArg,
  toF64,
  unknownMatrix,
} from "./check-helpers.js";

// ── LAPACK helpers ───────────────────────────────────────────────────────────

function qzLapack(
  dataA: Float64Array,
  dataB: Float64Array,
  n: number,
  computeEigvecs: boolean
) {
  const bridge = getEffectiveBridge("qz", "qz");
  if (!bridge.qz) return null;
  return bridge.qz(dataA, dataB, n, computeEigvecs);
}

function qzComplexLapack(
  dataARe: Float64Array,
  dataAIm: Float64Array,
  dataBRe: Float64Array,
  dataBIm: Float64Array,
  n: number,
  computeEigvecs: boolean
) {
  const bridge = getEffectiveBridge("qzComplex", "qzComplex");
  if (!bridge.qzComplex) return null;
  return bridge.qzComplex(
    dataARe,
    dataAIm,
    dataBRe,
    dataBIm,
    n,
    computeEigvecs
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────

function zeroF64(n: number): Float64Array {
  return new Float64Array(n);
}

function applyReal(
  A: RuntimeTensor,
  B: RuntimeTensor,
  n: number,
  nargout: number,
  computeEigvecs: boolean
): RuntimeValue[] {
  const result = qzLapack(toF64(A.data), toF64(B.data), n, computeEigvecs);
  if (!result) {
    throw new RuntimeError(
      "qz: LAPACK bridge not available (requires native addon)"
    );
  }

  const { AA, BB, Q, Z } = result;
  const AAout = RTV.tensor(new FloatXArray(AA), [n, n]);
  const BBout = RTV.tensor(new FloatXArray(BB), [n, n]);
  const Qout = RTV.tensor(new FloatXArray(Q), [n, n]);
  const Zout = RTV.tensor(new FloatXArray(Z), [n, n]);

  if (nargout === 4) {
    return [AAout, BBout, Qout, Zout];
  }

  // nargout === 6: also return V and W (generalized eigenvectors)
  const { alphai, V, W } = result;
  if (!V || !W) {
    throw new RuntimeError("qz: failed to compute generalized eigenvectors");
  }

  let hasComplex = false;
  for (let i = 0; i < n; i++) {
    if (Math.abs(alphai[i]) > 0) {
      hasComplex = true;
      break;
    }
  }

  const Vout = buildEigenvectorMatrix(V, alphai, n, hasComplex);
  const Wout = buildEigenvectorMatrix(W, alphai, n, hasComplex);
  return [AAout, BBout, Qout, Zout, Vout, Wout];
}

function applyComplex(
  A: RuntimeTensor,
  B: RuntimeTensor,
  n: number,
  nargout: number,
  computeEigvecs: boolean
): RuntimeValue[] {
  const nn = n * n;
  const aRe = toF64(A.data);
  const aIm = A.imag ? toF64(A.imag) : zeroF64(nn);
  const bRe = toF64(B.data);
  const bIm = B.imag ? toF64(B.imag) : zeroF64(nn);

  const result = qzComplexLapack(aRe, aIm, bRe, bIm, n, computeEigvecs);
  if (!result) {
    throw new RuntimeError(
      "qz: complex LAPACK bridge not available (requires native addon)"
    );
  }

  const AAout = RTV.tensor(
    new FloatXArray(result.AARe),
    [n, n],
    new FloatXArray(result.AAIm)
  );
  const BBout = RTV.tensor(
    new FloatXArray(result.BBRe),
    [n, n],
    new FloatXArray(result.BBIm)
  );
  const Qout = RTV.tensor(
    new FloatXArray(result.QRe),
    [n, n],
    new FloatXArray(result.QIm)
  );
  const Zout = RTV.tensor(
    new FloatXArray(result.ZRe),
    [n, n],
    new FloatXArray(result.ZIm)
  );

  if (nargout === 4) {
    return [AAout, BBout, Qout, Zout];
  }

  const { VRe, VIm, WRe, WIm } = result;
  if (!VRe || !VIm || !WRe || !WIm) {
    throw new RuntimeError("qz: failed to compute generalized eigenvectors");
  }

  const Vout = RTV.tensor(new FloatXArray(VRe), [n, n], new FloatXArray(VIm));
  const Wout = RTV.tensor(new FloatXArray(WRe), [n, n], new FloatXArray(WIm));
  return [AAout, BBout, Qout, Zout, Vout, Wout];
}

// ── registration ─────────────────────────────────────────────────────────────

export function registerQz(): void {
  register("qz", [
    {
      check: (argTypes, nargout) => {
        if (nargout !== 4 && nargout !== 6) return null;
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        if (!isOptionalStringArg(argTypes[2])) return null;
        if (!isMatrixLike(argTypes[0]) || !isMatrixLike(argTypes[1]))
          return null;
        const c = unknownMatrix(true);
        if (nargout === 4) return out(c, c, c, c);
        return out(c, c, c, c, c, c);
      },

      apply: (args, nargout) => {
        if (args.length < 2)
          throw new RuntimeError("qz requires at least 2 arguments");

        // Parse optional mode argument
        let mode: "real" | "complex" = "complex";
        if (args.length >= 3) {
          const modeArg = args[2];
          let modeStr = "";
          if (isRuntimeString(modeArg)) {
            modeStr = modeArg.replace(/^['"]|['"]$/g, "").toLowerCase();
          } else if (isRuntimeChar(modeArg)) {
            modeStr = modeArg.value.replace(/^['"]|['"]$/g, "").toLowerCase();
          }
          if (modeStr === "complex") mode = "complex";
          else if (modeStr === "real") mode = "real";
          else throw new RuntimeError(`qz: unknown mode '${modeStr}'`);
        }

        const A = args[0];
        const B = args[1];

        if (!isRuntimeTensor(A) || !isRuntimeTensor(B))
          throw new RuntimeError("qz: arguments must be numeric matrices");

        const [mA, nA] = tensorSize2D(A);
        const [mB, nB] = tensorSize2D(B);

        if (mA !== nA) throw new RuntimeError("qz: A must be a square matrix");
        if (mB !== nB) throw new RuntimeError("qz: B must be a square matrix");
        if (mA !== mB)
          throw new RuntimeError("qz: A and B must be the same size");

        const n = mA;
        const computeEigvecs = nargout >= 6;
        const isComplex = !!(A.imag || B.imag) || mode === "complex";

        if (isComplex) {
          return applyComplex(A, B, n, nargout, computeEigvecs);
        }

        return applyReal(A, B, n, nargout, computeEigvecs);
      },
    },
  ]);
}

/**
 * Build a (possibly complex) eigenvector matrix from LAPACK's packed real format.
 *
 * For complex conjugate eigenvalue pairs, LAPACK stores eigenvectors as:
 *   Column j:   real part
 *   Column j+1: imaginary part
 * Eigenvector for eigenvalue j   is V(:,j) + i*V(:,j+1)
 * Eigenvector for eigenvalue j+1 is V(:,j) - i*V(:,j+1)
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
      for (let i = 0; i < n; i++) {
        realPart[colMajorIndex(i, j, n)] = packedV[colMajorIndex(i, j, n)];
      }
      j++;
    } else {
      for (let i = 0; i < n; i++) {
        const re = packedV[colMajorIndex(i, j, n)];
        const im = packedV[colMajorIndex(i, j + 1, n)];
        realPart[colMajorIndex(i, j, n)] = re;
        imagPart[colMajorIndex(i, j, n)] = im;
        realPart[colMajorIndex(i, j + 1, n)] = re;
        imagPart[colMajorIndex(i, j + 1, n)] = -im;
      }
      j += 2;
    }
  }

  return RTV.tensor(realPart, [n, n], imagPart);
}

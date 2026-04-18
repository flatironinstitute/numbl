/**
 * Interpreter IBuiltins for linear algebra functions:
 * norm, vecnorm, dot, det, trace, cross, inv, svd, qr, lu, eig, chol,
 * linsolve, cond, rank, pinv, kron, blkdiag, pagemtimes, pagetranspose.
 *
 * These delegate to the same runtime implementations as the legacy builtins
 * but provide JitType-level resolve() for the interpreter/JIT pipeline.
 */

import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import {
  RTV,
  RuntimeError,
  tensorSize2D,
  toNumber,
  colMajorIndex,
} from "../../runtime/index.js";
import { rstr } from "../../runtime/runtime.js";
import type { JitType } from "../../jit/jitTypes.js";
import {
  defineBuiltin,
  registerIBuiltin,
  getIBuiltin,
  inferJitType,
} from "./types.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import {
  toF64,
  parseEconArgRuntime,
  parseStringArgLower,
  buildDiagMatrix,
  buildEigenvectorMatrix,
  maybeComplexTensor,
  gaussJordanEliminate,
} from "../../helpers/check-helpers.js";
import { forEachSlice, copyTensor } from "../../helpers/reduction-helpers.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import {
  isRuntimeSparseMatrix,
  type FloatXArrayType,
} from "../../runtime/types.js";
import {
  linsolveLapack,
  linsolveComplexLapack,
} from "../../helpers/linsolve.js";

// ── Type helpers ──────────────────────────────────────────────────────────

/** Check if arg is a numeric type that could be a matrix */
function isNumericJitType(t: JitType): boolean {
  return (
    t.kind === "number" ||
    t.kind === "boolean" ||
    t.kind === "complex_or_number" ||
    t.kind === "tensor" ||
    t.kind === "sparse_matrix"
  );
}

/** Return a "scalar number" JitType */
const NUM: JitType = { kind: "number" };
const COMPLEX_OR_NUM: JitType = { kind: "complex_or_number" };

/** Return a tensor type with unknown shape, optionally complex */
function tensorType(isComplex?: boolean, shape?: number[]): JitType {
  return {
    kind: "tensor",
    isComplex: isComplex ?? false,
    ...(shape ? { shape } : {}),
  };
}

/** Return a tensor type that may be complex (based on input) */
function tensorLikeInput(input: JitType): JitType {
  if (input.kind === "tensor") {
    return { kind: "tensor", isComplex: input.isComplex };
  }
  return { kind: "tensor", isComplex: false };
}

// ── norm ──────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "norm",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length < 1 || argTypes.length > 2)
          return null;
        if (!isNumericJitType(argTypes[0])) return null;
        return [NUM];
      },
      apply: args => {
        // Import the implementation inline to avoid circular deps
        return normApply(args);
      },
    },
  ],
});

function normApply(args: RuntimeValue[]): RuntimeValue {
  if (args.length < 1)
    throw new RuntimeError("norm requires at least 1 argument");
  const v = args[0];
  if (isRuntimeNumber(v)) return RTV.num(Math.abs(v));
  if (isRuntimeComplexNumber(v)) return RTV.num(Math.hypot(v.re, v.im));
  if (isRuntimeSparseMatrix(v))
    return normApply([sparseToDense(v), ...args.slice(1)]);
  if (!isRuntimeTensor(v))
    throw new RuntimeError("norm: argument must be numeric");
  return normImplTensor(v, args);
}

// LAPACK dlassq-style Euclidean norm: finite for any inputs in double range.
function scaledEuclid(magnitudes: ArrayLike<number>, count: number): number {
  let scale = 0;
  let ssq = 1;
  for (let i = 0; i < count; i++) {
    const ax = Math.abs(magnitudes[i]);
    if (ax === 0) continue;
    if (!isFinite(ax)) return isNaN(ax) ? NaN : Infinity;
    if (scale < ax) {
      const r = scale / ax;
      ssq = 1 + ssq * r * r;
      scale = ax;
    } else {
      const r = ax / scale;
      ssq += r * r;
    }
  }
  return scale === 0 ? 0 : scale * Math.sqrt(ssq);
}

function elemMag(
  re: ArrayLike<number>,
  im: ArrayLike<number> | undefined,
  i: number
): number {
  return im ? Math.hypot(re[i], im[i]) : Math.abs(re[i]);
}

function normImplTensor(v: RuntimeTensor, args: RuntimeValue[]): RuntimeValue {
  const shape = v.shape;
  const rows = shape[0] || 1;
  const cols = shape.length >= 2 ? shape[1] : 1;
  const isVec =
    shape.length <= 2
      ? rows === 1 || cols === 1
      : v.data.length === Math.max(...shape);

  let p: number | "fro" = 2;
  if (args.length >= 2) {
    const arg1 = args[1];
    if (isRuntimeChar(arg1)) {
      const s = arg1.value.toLowerCase();
      if (s === "inf") p = Infinity;
      else if (s === "fro") p = "fro";
      else throw new RuntimeError(`norm: invalid option '${arg1.value}'`);
    } else {
      p = toNumber(arg1);
    }
  }

  const imag = v.imag;
  if (isVec) {
    const vp = p === "fro" ? 2 : p;
    if (vp === Infinity) {
      let m = 0;
      for (let i = 0; i < v.data.length; i++) {
        const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
        m = Math.max(m, a);
      }
      return RTV.num(m);
    }
    if (vp === -Infinity) {
      let m = Infinity;
      for (let i = 0; i < v.data.length; i++) {
        const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
        m = Math.min(m, a);
      }
      return RTV.num(m);
    }
    if (vp === 2) {
      const mags = new Float64Array(v.data.length);
      for (let i = 0; i < v.data.length; i++)
        mags[i] = elemMag(v.data, imag, i);
      return RTV.num(scaledEuclid(mags, mags.length));
    }
    let s = 0;
    for (let i = 0; i < v.data.length; i++) {
      const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
      s += Math.pow(a, vp);
    }
    return RTV.num(Math.pow(s, 1 / vp));
  }

  // Matrix norms
  if (p === "fro" || (typeof p === "number" && isNaN(p))) {
    const mags = new Float64Array(v.data.length);
    for (let i = 0; i < v.data.length; i++) mags[i] = elemMag(v.data, imag, i);
    return RTV.num(scaledEuclid(mags, mags.length));
  }
  if (p === 1) {
    let maxColSum = 0;
    for (let j = 0; j < cols; j++) {
      let colSum = 0;
      for (let i = 0; i < rows; i++) {
        const idx = j * rows + i;
        colSum += imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
      }
      maxColSum = Math.max(maxColSum, colSum);
    }
    return RTV.num(maxColSum);
  }
  if (p === Infinity) {
    let maxRowSum = 0;
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        const idx = j * rows + i;
        rowSum += imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
      }
      maxRowSum = Math.max(maxRowSum, rowSum);
    }
    return RTV.num(maxRowSum);
  }
  if (p === 2) {
    const bridge = getEffectiveBridge("norm", "svd");
    if (bridge && bridge.svd) {
      const f64 =
        v.data instanceof Float64Array ? v.data : new Float64Array(v.data);
      const result = bridge.svd(f64, rows, cols, false, false);
      return RTV.num(result.S[0]);
    }
    throw new RuntimeError(
      "norm: matrix 2-norm requires LAPACK (build the native addon)"
    );
  }
  throw new RuntimeError("norm: for matrices, p must be 1, 2, Inf, or 'fro'");
}

// ── vecnorm ──────────────────────────────────────────────────────────────

defineBuiltin({
  name: "vecnorm",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length < 1 || argTypes.length > 3)
          return null;
        if (!isNumericJitType(argTypes[0])) return null;
        const a = argTypes[0];
        if (
          a.kind === "number" ||
          a.kind === "boolean" ||
          a.kind === "complex_or_number"
        )
          return [NUM];
        return [{ kind: "tensor", isComplex: false }];
      },
      apply: args => {
        if (args.length < 1)
          throw new RuntimeError("vecnorm requires at least 1 argument");
        const v = args[0];
        let p = 2;
        if (args.length >= 2) p = toNumber(args[1]);
        if (isRuntimeNumber(v)) return RTV.num(Math.abs(v));
        if (isRuntimeComplexNumber(v)) return RTV.num(Math.hypot(v.re, v.im));
        if (!isRuntimeTensor(v))
          throw new RuntimeError("vecnorm: argument must be numeric");
        let dim: number;
        if (args.length >= 3) {
          dim = Math.round(toNumber(args[2]));
        } else {
          const idx = v.shape.findIndex((d: number) => d > 1);
          dim = idx === -1 ? 1 : idx + 1;
        }
        return vecnormAlongDim(v, p, dim);
      },
    },
  ],
});

function vecnormAlongDim(
  v: RuntimeTensor,
  p: number,
  dim: number
): RuntimeValue {
  const dimIdx = dim - 1;
  if (dimIdx >= v.shape.length) {
    const result = new FloatXArray(v.data.length);
    const imag = v.imag;
    for (let i = 0; i < v.data.length; i++) {
      result[i] = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
    }
    return RTV.tensor(result, [...v.shape]);
  }
  const info = forEachSlice(v.shape, dim, () => {});
  if (!info) return copyTensor(v);
  const result = new FloatXArray(info.totalElems);
  const imag = v.imag;
  forEachSlice(v.shape, dim, (outIdx, srcIndices) => {
    if (p === Infinity) {
      let m = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (a > m) m = a;
      }
      result[outIdx] = m;
    } else if (p === -Infinity) {
      let m = Infinity;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (a < m) m = a;
      }
      result[outIdx] = m;
    } else if (p === 2) {
      let scale = 0;
      let ssq = 1;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const ax = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        if (ax === 0) continue;
        if (!isFinite(ax)) {
          scale = isNaN(ax) ? NaN : Infinity;
          ssq = 1;
          break;
        }
        if (scale < ax) {
          const r = scale / ax;
          ssq = 1 + ssq * r * r;
          scale = ax;
        } else {
          const r = ax / scale;
          ssq += r * r;
        }
      }
      result[outIdx] = scale === 0 ? 0 : scale * Math.sqrt(ssq);
    } else {
      let s = 0;
      for (let k = 0; k < srcIndices.length; k++) {
        const idx = srcIndices[k];
        const a = imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
        s += Math.pow(a, p);
      }
      result[outIdx] = Math.pow(s, 1 / p);
    }
  });
  return RTV.tensor(result, info.resultShape);
}

// ── dot ──────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "dot",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length !== 2) return null;
        if (!isNumericJitType(argTypes[0]) || !isNumericJitType(argTypes[1]))
          return null;
        const hasTensor = argTypes.some(t => t.kind === "tensor");
        const hasComplex = argTypes.some(
          t =>
            t.kind === "complex_or_number" ||
            (t.kind === "tensor" && t.isComplex)
        );
        if (hasTensor) return [hasComplex ? COMPLEX_OR_NUM : tensorType(false)];
        return [hasComplex ? COMPLEX_OR_NUM : NUM];
      },
      apply: args => {
        if (args.length !== 2)
          throw new RuntimeError("dot requires 2 arguments");
        const a = args[0],
          b = args[1];
        let aRe: FloatXArrayType | null = null,
          aIm: FloatXArrayType | null = null;
        let aShape: number[] = [1, 1];
        let bRe: FloatXArrayType | null = null,
          bIm: FloatXArrayType | null = null;

        if (isRuntimeTensor(a)) {
          aRe = a.data;
          aIm = a.imag ?? null;
          aShape = a.shape;
        } else if (isRuntimeNumber(a)) {
          aRe = new FloatXArray([a]);
        } else if (isRuntimeComplexNumber(a)) {
          aRe = new FloatXArray([a.re]);
          aIm = new FloatXArray([a.im]);
        }

        if (isRuntimeTensor(b)) {
          bRe = b.data;
          bIm = b.imag ?? null;
        } else if (isRuntimeNumber(b)) {
          bRe = new FloatXArray([b]);
        } else if (isRuntimeComplexNumber(b)) {
          bRe = new FloatXArray([b.re]);
          bIm = new FloatXArray([b.im]);
        }

        if (!aRe || !bRe)
          throw new RuntimeError("dot: arguments must be numeric");
        if (aRe.length !== bRe.length)
          throw new RuntimeError("dot: vectors must be same length");

        const hasComplex = aIm !== null || bIm !== null;
        const rows = aShape[0];
        const cols = aShape.length >= 2 ? aShape[1] : 1;
        const isMatrix = rows > 1 && cols > 1;

        if (isMatrix) {
          const resultRe = new FloatXArray(cols);
          const resultIm = hasComplex ? new FloatXArray(cols) : null;
          for (let c = 0; c < cols; c++) {
            let sRe = 0,
              sIm = 0;
            for (let r = 0; r < rows; r++) {
              const idx = c * rows + r;
              const aRei = aRe[idx],
                aImi = aIm ? aIm[idx] : 0;
              const bRei = bRe[idx],
                bImi = bIm ? bIm[idx] : 0;
              sRe += aRei * bRei + aImi * bImi;
              sIm += aRei * bImi - aImi * bRei;
            }
            resultRe[c] = sRe;
            if (resultIm) resultIm[c] = sIm;
          }
          return RTV.tensor(resultRe, [1, cols], resultIm ?? undefined);
        }

        if (!hasComplex) {
          let s = 0;
          for (let i = 0; i < aRe.length; i++) s += aRe[i] * bRe[i];
          return RTV.num(s);
        }
        let sRe = 0,
          sIm = 0;
        for (let i = 0; i < aRe.length; i++) {
          const aRei = aRe[i],
            aImi = aIm ? aIm[i] : 0;
          const bRei = bRe[i],
            bImi = bIm ? bIm[i] : 0;
          sRe += aRei * bRei + aImi * bImi;
          sIm += aRei * bImi - aImi * bRei;
        }
        if (sIm === 0) return RTV.num(sRe);
        return RTV.complex(sRe, sIm);
      },
    },
  ],
});

// ── det ──────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "det",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout > 1) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        const hasComplex =
          argTypes[0].kind === "complex_or_number" ||
          (argTypes[0].kind === "tensor" && argTypes[0].isComplex);
        return [hasComplex ? COMPLEX_OR_NUM : NUM];
      },
      apply: args => {
        if (args.length !== 1)
          throw new RuntimeError("det requires 1 argument");
        const A = args[0];
        if (isRuntimeNumber(A)) return A;
        if (typeof A === "boolean") return RTV.num(A ? 1 : 0);
        if (!isRuntimeTensor(A))
          throw new RuntimeError("det: argument must be a matrix");
        const [m, n] = tensorSize2D(A);
        if (m !== n) throw new RuntimeError("det: matrix must be square");
        if (A.imag) {
          const [detRe, detIm] = detComplexJS(toF64(A.data), toF64(A.imag), n);
          if (Math.abs(detIm) < 1e-15) return RTV.num(detRe);
          return RTV.complex(detRe, detIm);
        }
        return RTV.num(detJS(toF64(A.data), n));
      },
    },
  ],
});

function detJS(data: Float32Array | Float64Array, n: number): number {
  const a = new Float64Array(n * n);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) a[r * n + c] = data[r + c * n];
  let det = 1;
  for (let col = 0; col < n; col++) {
    let maxRow = col,
      maxVal = Math.abs(a[col * n + col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(a[row * n + col]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal === 0) return 0;
    if (maxRow !== col) {
      for (let k = 0; k < n; k++) {
        const tmp = a[col * n + k];
        a[col * n + k] = a[maxRow * n + k];
        a[maxRow * n + k] = tmp;
      }
      det *= -1;
    }
    const pivot = a[col * n + col];
    det *= pivot;
    for (let row = col + 1; row < n; row++) {
      const factor = a[row * n + col] / pivot;
      for (let k = col; k < n; k++) a[row * n + k] -= factor * a[col * n + k];
    }
  }
  return det;
}

function detComplexJS(
  dataRe: Float32Array | Float64Array,
  dataIm: Float32Array | Float64Array,
  n: number
): [number, number] {
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      re[r * n + c] = dataRe[r + c * n];
      im[r * n + c] = dataIm[r + c * n];
    }
  let detRe = 1,
    detIm = 0;
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal =
      re[col * n + col] * re[col * n + col] +
      im[col * n + col] * im[col * n + col];
    for (let row = col + 1; row < n; row++) {
      const v =
        re[row * n + col] * re[row * n + col] +
        im[row * n + col] * im[row * n + col];
      if (v > maxVal) {
        maxVal = v;
        maxRow = row;
      }
    }
    if (maxVal === 0) return [0, 0];
    if (maxRow !== col) {
      for (let k = 0; k < n; k++) {
        let tmp = re[col * n + k];
        re[col * n + k] = re[maxRow * n + k];
        re[maxRow * n + k] = tmp;
        tmp = im[col * n + k];
        im[col * n + k] = im[maxRow * n + k];
        im[maxRow * n + k] = tmp;
      }
      detRe = -detRe;
      detIm = -detIm;
    }
    const pivRe = re[col * n + col],
      pivIm = im[col * n + col];
    const newDetRe = detRe * pivRe - detIm * pivIm;
    const newDetIm = detRe * pivIm + detIm * pivRe;
    detRe = newDetRe;
    detIm = newDetIm;
    const pivMag2 = pivRe * pivRe + pivIm * pivIm;
    for (let row = col + 1; row < n; row++) {
      const rRe = re[row * n + col],
        rIm = im[row * n + col];
      const fRe = (rRe * pivRe + rIm * pivIm) / pivMag2;
      const fIm = (rIm * pivRe - rRe * pivIm) / pivMag2;
      for (let k = col; k < n; k++) {
        re[row * n + k] -= fRe * re[col * n + k] - fIm * im[col * n + k];
        im[row * n + k] -= fRe * im[col * n + k] + fIm * re[col * n + k];
      }
    }
  }
  return [detRe, detIm];
}

// ── trace ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "trace",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout > 1) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        return [NUM];
      },
      apply: args => {
        if (args.length !== 1)
          throw new RuntimeError("trace requires 1 argument");
        const A = args[0];
        if (isRuntimeNumber(A)) return A;
        if (!isRuntimeTensor(A))
          throw new RuntimeError("trace: argument must be a matrix");
        const [rows, cols] = tensorSize2D(A);
        const n = Math.min(rows, cols);
        let sum = 0;
        for (let i = 0; i < n; i++) sum += A.data[i + i * rows];
        return RTV.num(sum);
      },
    },
  ],
});

// ── cross ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cross",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1) return null;
        if (argTypes.length !== 2 && argTypes.length !== 3) return null;
        if (!isNumericJitType(argTypes[0]) || !isNumericJitType(argTypes[1]))
          return null;
        if (argTypes[0].kind === "tensor")
          return [tensorLikeInput(argTypes[0])];
        return [tensorType(false)];
      },
      apply: args => {
        if (args.length < 2 || args.length > 3)
          throw new RuntimeError("cross requires 2 or 3 arguments");
        const a = args[0],
          b = args[1];
        if (!isRuntimeTensor(a) || !isRuntimeTensor(b))
          throw new RuntimeError("cross: arguments must be vectors or arrays");
        const shape = a.shape;
        if (
          shape.length !== b.shape.length ||
          shape.some((s, i) => s !== b.shape[i])
        )
          throw new RuntimeError("cross: A and B must have the same size");
        let dim: number;
        if (args.length === 3) {
          dim = toNumber(args[2]);
          if (!Number.isInteger(dim) || dim < 1)
            throw new RuntimeError("cross: dim must be a positive integer");
        } else {
          dim = shape.indexOf(3) + 1;
          if (dim === 0)
            throw new RuntimeError(
              "cross: A and B must have at least one dimension of length 3"
            );
        }
        const dimIdx = dim - 1;
        if (dimIdx >= shape.length || shape[dimIdx] !== 3)
          throw new RuntimeError(
            `cross: size(A,${dim}) and size(B,${dim}) must be 3`
          );
        const totalLen = a.data.length;
        const result = new FloatXArray(totalLen);
        const strides = new Array(shape.length);
        strides[0] = 1;
        for (let d = 1; d < shape.length; d++)
          strides[d] = strides[d - 1] * shape[d - 1];
        const dimStride = strides[dimIdx];
        const outerStride =
          dimIdx + 1 < shape.length ? strides[dimIdx + 1] : totalLen;
        const innerSize = dimStride;
        const numOuter = totalLen / outerStride;
        for (let outer = 0; outer < numOuter; outer++) {
          const blockBase = outer * outerStride;
          for (let inner = 0; inner < innerSize; inner++) {
            const base = blockBase + inner;
            const i0 = base,
              i1 = base + dimStride,
              i2 = base + 2 * dimStride;
            const ax = a.data[i0],
              ay = a.data[i1],
              az = a.data[i2];
            const bx = b.data[i0],
              by = b.data[i1],
              bz = b.data[i2];
            result[i0] = ay * bz - az * by;
            result[i1] = az * bx - ax * bz;
            result[i2] = ax * by - ay * bx;
          }
        }
        return RTV.tensor(result, [...shape]);
      },
    },
  ],
});

// ── inv ──────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "inv",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 1 || nargout > 1) return null;
        if (!isNumericJitType(argTypes[0])) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "boolean") return [NUM];
        if (a.kind === "complex_or_number") return [COMPLEX_OR_NUM];
        if (a.kind === "tensor")
          return [{ kind: "tensor", isComplex: a.isComplex }];
        return null;
      },
      apply: args => {
        if (args.length < 1) throw new RuntimeError("inv requires 1 argument");
        const A = args[0];
        if (isRuntimeComplexNumber(A)) {
          const { re, im } = A;
          if (re === 0 && im === 0)
            throw new RuntimeError("inv: argument is singular");
          // Smith's algorithm for 1/(re + im*i) to avoid spurious over/underflow.
          if (Math.abs(re) >= Math.abs(im)) {
            const r = im / re;
            const d = re + im * r;
            return RTV.complex(1 / d, -r / d);
          }
          const r = re / im;
          const d = im + re * r;
          return RTV.complex(r / d, -1 / d);
        }
        if (isRuntimeNumber(A)) {
          if (A === 0) throw new RuntimeError("inv: argument is singular");
          return RTV.num(1 / A);
        }
        if (!isRuntimeTensor(A))
          throw new RuntimeError("inv: argument must be numeric");
        const [m, n] = tensorSize2D(A);
        if (m !== n) throw new RuntimeError("inv: matrix must be square");
        if (A.imag !== undefined) {
          const bridge = getLapackBridge();
          if (bridge?.invComplex) {
            const result = bridge.invComplex(toF64(A.data), toF64(A.imag), n);
            if (result)
              return RTV.tensor(
                new FloatXArray(result.re),
                [n, n],
                new FloatXArray(result.im)
              );
          }
          const result = invComplexJS(A.data, A.imag, n);
          return RTV.tensor(result.re, [n, n], result.im);
        }
        const bridge = getEffectiveBridge("inv");
        return RTV.tensor(new FloatXArray(bridge.inv(toF64(A.data), n)), [
          n,
          n,
        ]);
      },
    },
  ],
});

function invComplexJS(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  n: number
): { re: FloatXArrayType; im: FloatXArrayType } {
  const augRe = new Float64Array(n * 2 * n);
  const augIm = new Float64Array(n * 2 * n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      augRe[row * 2 * n + col] = dataRe[row + col * n];
      augIm[row * 2 * n + col] = dataIm[row + col * n];
    }
    augRe[row * 2 * n + n + row] = 1;
  }
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxMag = augRe[col * 2 * n + col] ** 2 + augIm[col * 2 * n + col] ** 2;
    for (let row = col + 1; row < n; row++) {
      const mag = augRe[row * 2 * n + col] ** 2 + augIm[row * 2 * n + col] ** 2;
      if (mag > maxMag) {
        maxMag = mag;
        maxRow = row;
      }
    }
    if (maxMag < 1e-300)
      throw new RuntimeError("inv: matrix is singular or nearly singular");
    if (maxRow !== col) {
      for (let k = 0; k < 2 * n; k++) {
        let tmp = augRe[col * 2 * n + k];
        augRe[col * 2 * n + k] = augRe[maxRow * 2 * n + k];
        augRe[maxRow * 2 * n + k] = tmp;
        tmp = augIm[col * 2 * n + k];
        augIm[col * 2 * n + k] = augIm[maxRow * 2 * n + k];
        augIm[maxRow * 2 * n + k] = tmp;
      }
    }
    const pivotRe = augRe[col * 2 * n + col],
      pivotIm = augIm[col * 2 * n + col];
    const pivotMagSq = pivotRe * pivotRe + pivotIm * pivotIm;
    const invPivotRe = pivotRe / pivotMagSq,
      invPivotIm = -pivotIm / pivotMagSq;
    for (let k = col; k < 2 * n; k++) {
      const re = augRe[col * 2 * n + k],
        im = augIm[col * 2 * n + k];
      augRe[col * 2 * n + k] = re * invPivotRe - im * invPivotIm;
      augIm[col * 2 * n + k] = re * invPivotIm + im * invPivotRe;
    }
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factorRe = augRe[row * 2 * n + col],
        factorIm = augIm[row * 2 * n + col];
      if (factorRe === 0 && factorIm === 0) continue;
      for (let k = col; k < 2 * n; k++) {
        const multRe =
          factorRe * augRe[col * 2 * n + k] - factorIm * augIm[col * 2 * n + k];
        const multIm =
          factorRe * augIm[col * 2 * n + k] + factorIm * augRe[col * 2 * n + k];
        augRe[row * 2 * n + k] -= multRe;
        augIm[row * 2 * n + k] -= multIm;
      }
    }
  }
  const resultRe = new FloatXArray(n * n);
  const resultIm = new FloatXArray(n * n);
  for (let row = 0; row < n; row++)
    for (let col = 0; col < n; col++) {
      resultRe[row + col * n] = augRe[row * 2 * n + n + col];
      resultIm[row + col * n] = augIm[row * 2 * n + n + col];
    }
  return { re: resultRe, im: resultIm };
}

// ── svd ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "svd",
  resolve: (argTypes, nargout) => {
    if (
      nargout < 0 ||
      nargout > 3 ||
      argTypes.length < 1 ||
      argTypes.length > 2
    )
      return null;
    if (!isNumericJitType(argTypes[0])) return null;
    const c = tensorType();
    if (nargout <= 1)
      return { outputTypes: [c], apply: (args, n) => svdApply(args, n) };
    return { outputTypes: [c, c, c], apply: (args, n) => svdApply(args, n) };
  },
});

function svdApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (args.length < 1)
    throw new RuntimeError("svd requires at least 1 argument");
  const A = args[0];
  if (isRuntimeNumber(A)) {
    const val = Math.abs(A);
    if (nargout <= 1) return RTV.tensor(new FloatXArray([val]), [1, 1]);
    return [
      RTV.tensor(new FloatXArray([A >= 0 ? 1 : -1]), [1, 1]),
      RTV.tensor(new FloatXArray([val]), [1, 1]),
      RTV.tensor(new FloatXArray([1]), [1, 1]),
    ];
  }
  if (!isRuntimeTensor(A))
    throw new RuntimeError("svd: argument must be numeric");
  const econ = parseEconArgRuntime(args[1]);
  const [m, n] = tensorSize2D(A);
  const k = Math.min(m, n);
  if (A.imag) {
    const bridge = getLapackBridge();
    if (!bridge?.svdComplex)
      throw new RuntimeError(
        "svd: complex SVD requires LAPACK (build the native addon)"
      );
    const result = bridge.svdComplex(
      toF64(A.data),
      toF64(A.imag),
      m,
      n,
      econ,
      nargout === 3
    );
    if (!result) throw new RuntimeError("svd: complex SVD failed");
    if (nargout <= 1) return RTV.tensor(new FloatXArray(result.S), [k, 1]);
    const uCols = econ ? k : m;
    const vCols = econ ? k : n;
    return [
      RTV.tensor(
        new FloatXArray(result.URe!),
        [m, uCols],
        new FloatXArray(result.UIm!)
      ),
      buildDiagMatrix(result.S, undefined, econ ? k : [m, n]),
      RTV.tensor(
        new FloatXArray(result.VRe!),
        [n, vCols],
        new FloatXArray(result.VIm!)
      ),
    ];
  }
  const bridge = getEffectiveBridge("svd", "svd");
  if (bridge?.svd) {
    const result = bridge.svd(toF64(A.data), m, n, econ, nargout === 3);
    if (result) {
      if (nargout <= 1) return RTV.tensor(new FloatXArray(result.S), [k, 1]);
      const uCols = econ ? k : m;
      const vCols = econ ? k : n;
      return [
        RTV.tensor(new FloatXArray(result.U!), [m, uCols]),
        buildDiagMatrix(result.S, undefined, econ ? k : [m, n]),
        RTV.tensor(new FloatXArray(result.V!), [n, vCols]),
      ];
    }
  }
  if (nargout > 1)
    throw new RuntimeError(
      "svd: full decomposition requires LAPACK (build the native addon)"
    );
  const ATA = computeATA(A.data, m, n);
  const eigenvalues = powerIterationEigenvalues(ATA, n, k);
  return RTV.tensor(
    new FloatXArray(eigenvalues.map(ev => Math.sqrt(Math.max(0, ev)))),
    [k, 1]
  );
}

function computeATA(
  A_data: FloatXArrayType,
  m: number,
  n: number
): FloatXArrayType {
  const result = new FloatXArray(n * n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i <= j; i++) {
      let sum = 0;
      for (let k = 0; k < m; k++)
        sum += A_data[colMajorIndex(k, i, m)] * A_data[colMajorIndex(k, j, m)];
      result[colMajorIndex(i, j, n)] = sum;
      if (i !== j) result[colMajorIndex(j, i, n)] = sum;
    }
  return result;
}

function powerIterationEigenvalues(
  A_data: FloatXArrayType,
  n: number,
  numEigenvalues: number
): number[] {
  const eigenvalues: number[] = [];
  const A_copy = new FloatXArray(A_data);
  for (let ev = 0; ev < numEigenvalues; ev++) {
    const v = new FloatXArray(n);
    for (let i = 0; i < n; i++) v[i] = Math.random();
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < n; i++) v[i] /= norm;
    let lambda = 0;
    for (let iter = 0; iter < 100; iter++) {
      const Av = new FloatXArray(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++)
          sum += A_copy[colMajorIndex(i, j, n)] * v[j];
        Av[i] = sum;
      }
      lambda = 0;
      for (let i = 0; i < n; i++) lambda += v[i] * Av[i];
      norm = 0;
      for (let i = 0; i < n; i++) norm += Av[i] * Av[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-14) break;
      for (let i = 0; i < n; i++) v[i] = Av[i] / norm;
    }
    eigenvalues.push(lambda);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        A_copy[colMajorIndex(i, j, n)] -= lambda * v[i] * v[j];
  }
  return eigenvalues;
}

// ── qr ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "qr",
  resolve: (argTypes, nargout) => {
    if (
      nargout < 0 ||
      nargout > 3 ||
      argTypes.length < 1 ||
      argTypes.length > 2
    )
      return null;
    if (!isNumericJitType(argTypes[0])) return null;
    const isComplex = argTypes[0].kind === "tensor" && argTypes[0].isComplex;
    const t = tensorType(isComplex || undefined);
    if (nargout <= 1)
      return { outputTypes: [t], apply: (args, n) => qrApply(args, n) };
    if (nargout === 3)
      return {
        outputTypes: [t, t, t],
        apply: (args, n) => qrApply(args, n),
      };
    return { outputTypes: [t, t], apply: (args, n) => qrApply(args, n) };
  },
});

function qrApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (args.length < 1)
    throw new RuntimeError("qr requires at least 1 argument");
  const A = args[0];
  if (isRuntimeNumber(A)) {
    const val = A;
    const s = val >= 0 ? 1 : -1;
    if (nargout <= 1) return RTV.tensor(new FloatXArray([s * val]), [1, 1]);
    if (nargout === 3)
      return [
        RTV.tensor(new FloatXArray([s]), [1, 1]),
        RTV.tensor(new FloatXArray([s * val]), [1, 1]),
        RTV.tensor(new FloatXArray([1]), [1, 1]),
      ];
    return [
      RTV.tensor(new FloatXArray([s]), [1, 1]),
      RTV.tensor(new FloatXArray([s * val]), [1, 1]),
    ];
  }
  if (!isRuntimeTensor(A))
    throw new RuntimeError("qr: argument must be numeric");
  const econ = parseEconArgRuntime(args[1]);
  const [m, n] = tensorSize2D(A);
  const k = Math.min(m, n);

  // 3-output: column-pivoted QR → [Q, R, E]
  if (nargout === 3) {
    return qrPivotApply(A, m, n, k, econ);
  }

  if (A.imag) {
    const bridge = getEffectiveBridge("qr", "qrComplex");
    if (!bridge?.qrComplex)
      throw new RuntimeError("qr: complex QR requires the native LAPACK addon");
    const result = bridge.qrComplex(
      toF64(A.data),
      toF64(A.imag),
      m,
      n,
      econ,
      nargout === 2
    );
    if (!result) throw new RuntimeError("qr: complex QR failed");
    if (nargout <= 1) {
      const rRows = econ ? k : m;
      return RTV.tensor(
        new FloatXArray(result.RRe),
        [rRows, n],
        new FloatXArray(result.RIm)
      );
    }
    const qCols = econ ? k : m;
    return [
      RTV.tensor(
        new FloatXArray(result.QRe!),
        [m, qCols],
        new FloatXArray(result.QIm!)
      ),
      RTV.tensor(
        new FloatXArray(result.RRe),
        [econ ? k : m, n],
        new FloatXArray(result.RIm)
      ),
    ];
  }

  const bridge = getEffectiveBridge("qr", "qr");
  if (bridge?.qr) {
    const result = bridge.qr(toF64(A.data), m, n, econ, nargout === 2);
    if (result) {
      if (nargout <= 1)
        return RTV.tensor(new FloatXArray(result.R), [econ ? k : m, n]);
      const qCols = econ ? k : m;
      return [
        RTV.tensor(new FloatXArray(result.Q), [m, qCols]),
        RTV.tensor(new FloatXArray(result.R), [econ ? k : m, n]),
      ];
    }
  }

  // JS fallback (Householder)
  const R_data = new FloatXArray(A.data);
  const vecs: FloatXArrayType[] = [];
  const taus: number[] = [];
  for (let j = 0; j < k; j++) {
    const len = m - j;
    const x = new FloatXArray(len);
    for (let i = 0; i < len; i++) x[i] = R_data[colMajorIndex(j + i, j, m)];
    let normx = 0;
    for (let i = 0; i < len; i++) normx += x[i] * x[i];
    normx = Math.sqrt(normx);
    if (normx === 0) {
      vecs.push(new FloatXArray(len));
      taus.push(0);
      continue;
    }
    const sign = x[0] >= 0 ? 1 : -1;
    const alpha = -sign * normx;
    const v = new FloatXArray(len);
    v[0] = x[0] - alpha;
    for (let i = 1; i < len; i++) v[i] = x[i];
    let vnorm = 0;
    for (let i = 0; i < len; i++) vnorm += v[i] * v[i];
    const tau = vnorm === 0 ? 0 : 2.0 / vnorm;
    vecs.push(v);
    taus.push(tau);
    for (let c = j; c < n; c++) {
      let dot = 0;
      for (let i = 0; i < len; i++)
        dot += v[i] * R_data[colMajorIndex(j + i, c, m)];
      const scale = tau * dot;
      for (let i = 0; i < len; i++)
        R_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
    }
  }

  if (nargout <= 1) {
    if (econ) {
      const R_econ = new FloatXArray(k * n);
      for (let r = 0; r < k; r++)
        for (let c = 0; c < n; c++)
          R_econ[colMajorIndex(r, c, k)] = R_data[colMajorIndex(r, c, m)];
      return RTV.tensor(R_econ, [k, n]);
    }
    return RTV.tensor(new FloatXArray(R_data.slice(0, m * n)), [m, n]);
  }

  if (econ) {
    const qCols = k;
    const Q_data = new FloatXArray(m * qCols);
    for (let i = 0; i < Math.min(m, qCols); i++)
      Q_data[colMajorIndex(i, i, m)] = 1;
    for (let j = k - 1; j >= 0; j--) {
      const v = vecs[j],
        tau = taus[j];
      if (tau === 0) continue;
      const len = m - j;
      for (let c = j; c < qCols; c++) {
        let dot = 0;
        for (let i = 0; i < len; i++)
          dot += v[i] * Q_data[colMajorIndex(j + i, c, m)];
        const scale = tau * dot;
        for (let i = 0; i < len; i++)
          Q_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
      }
    }
    const R_econ = new FloatXArray(k * n);
    for (let r = 0; r < k; r++)
      for (let c = 0; c < n; c++)
        R_econ[colMajorIndex(r, c, k)] = R_data[colMajorIndex(r, c, m)];
    return [RTV.tensor(Q_data, [m, qCols]), RTV.tensor(R_econ, [k, n])];
  }

  const Q_data = new FloatXArray(m * m);
  for (let i = 0; i < m; i++) Q_data[colMajorIndex(i, i, m)] = 1;
  for (let j = k - 1; j >= 0; j--) {
    const v = vecs[j],
      tau = taus[j];
    if (tau === 0) continue;
    const len = m - j;
    for (let c = j; c < m; c++) {
      let dot = 0;
      for (let i = 0; i < len; i++)
        dot += v[i] * Q_data[colMajorIndex(j + i, c, m)];
      const scale = tau * dot;
      for (let i = 0; i < len; i++)
        Q_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
    }
  }
  return [
    RTV.tensor(Q_data, [m, m]),
    RTV.tensor(new FloatXArray(R_data.slice(0, m * n)), [m, n]),
  ];
}

/** Convert 1-based permutation vector to permutation matrix or keep as vector. */
function permResult(
  jpvt: FloatXArrayType | Int32Array,
  n: number,
  econ: boolean
): RuntimeValue {
  const perm = new FloatXArray(n);
  for (let i = 0; i < n; i++) perm[i] = jpvt[i];
  if (econ) return RTV.tensor(perm, [1, n]);
  // Non-economy: return n×n permutation matrix
  const mat = new FloatXArray(n * n);
  for (let j = 0; j < n; j++) mat[colMajorIndex(jpvt[j] - 1, j, n)] = 1;
  return RTV.tensor(mat, [n, n]);
}

function qrPivotApply(
  A: RuntimeTensor,
  m: number,
  n: number,
  k: number,
  econ: boolean
): RuntimeValue[] {
  if (A.imag) {
    const bridge = getEffectiveBridge("qr", "qrPivotComplex");
    if (!bridge?.qrPivotComplex)
      throw new RuntimeError(
        "qr: complex column-pivoted QR requires the native LAPACK addon"
      );
    const result = bridge.qrPivotComplex(
      toF64(A.data),
      toF64(A.imag),
      m,
      n,
      econ
    );
    const rRows = econ ? k : m;
    const qCols = econ ? k : m;
    return [
      RTV.tensor(
        new FloatXArray(result.QRe),
        [m, qCols],
        new FloatXArray(result.QIm)
      ),
      RTV.tensor(
        new FloatXArray(result.RRe),
        [rRows, n],
        new FloatXArray(result.RIm)
      ),
      permResult(result.jpvt, n, econ),
    ];
  }

  const bridge = getEffectiveBridge("qr", "qrPivot");
  if (bridge?.qrPivot) {
    const result = bridge.qrPivot(toF64(A.data), m, n, econ);
    if (result) {
      const rRows = econ ? k : m;
      const qCols = econ ? k : m;
      return [
        RTV.tensor(new FloatXArray(result.Q), [m, qCols]),
        RTV.tensor(new FloatXArray(result.R), [rRows, n]),
        permResult(result.jpvt, n, econ),
      ];
    }
  }

  // JS fallback: Householder QR with column pivoting
  const R_data = new FloatXArray(A.data);
  const perm = new Float64Array(n);
  for (let i = 0; i < n; i++) perm[i] = i; // 0-based during computation

  // Compute initial column norms
  const colNorms = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < m; i++) {
      const v = R_data[colMajorIndex(i, j, m)];
      s += v * v;
    }
    colNorms[j] = s;
  }

  const vecs: FloatXArrayType[] = [];
  const taus: number[] = [];

  for (let j = 0; j < k; j++) {
    // Find column with max remaining norm
    let maxNorm = colNorms[j];
    let maxCol = j;
    for (let c = j + 1; c < n; c++) {
      if (colNorms[c] > maxNorm) {
        maxNorm = colNorms[c];
        maxCol = c;
      }
    }
    // Swap columns j and maxCol in R_data, perm, colNorms
    if (maxCol !== j) {
      for (let i = 0; i < m; i++) {
        const ij = colMajorIndex(i, j, m);
        const ic = colMajorIndex(i, maxCol, m);
        const tmp = R_data[ij];
        R_data[ij] = R_data[ic];
        R_data[ic] = tmp;
      }
      const tmpP = perm[j];
      perm[j] = perm[maxCol];
      perm[maxCol] = tmpP;
      const tmpN = colNorms[j];
      colNorms[j] = colNorms[maxCol];
      colNorms[maxCol] = tmpN;
    }

    // Householder reflector for column j
    const len = m - j;
    const x = new FloatXArray(len);
    for (let i = 0; i < len; i++) x[i] = R_data[colMajorIndex(j + i, j, m)];
    let normx = 0;
    for (let i = 0; i < len; i++) normx += x[i] * x[i];
    normx = Math.sqrt(normx);
    if (normx === 0) {
      vecs.push(new FloatXArray(len));
      taus.push(0);
      // Update remaining column norms
      for (let c = j + 1; c < n; c++) {
        const vj = R_data[colMajorIndex(j, c, m)];
        colNorms[c] -= vj * vj;
        if (colNorms[c] < 0) colNorms[c] = 0;
      }
      continue;
    }
    const sign = x[0] >= 0 ? 1 : -1;
    const alpha = -sign * normx;
    const v = new FloatXArray(len);
    v[0] = x[0] - alpha;
    for (let i = 1; i < len; i++) v[i] = x[i];
    let vnorm = 0;
    for (let i = 0; i < len; i++) vnorm += v[i] * v[i];
    const tau = vnorm === 0 ? 0 : 2.0 / vnorm;
    vecs.push(v);
    taus.push(tau);
    for (let c = j; c < n; c++) {
      let dot = 0;
      for (let i = 0; i < len; i++)
        dot += v[i] * R_data[colMajorIndex(j + i, c, m)];
      const scale = tau * dot;
      for (let i = 0; i < len; i++)
        R_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
    }
    // Update remaining column norms
    for (let c = j + 1; c < n; c++) {
      const vj = R_data[colMajorIndex(j, c, m)];
      colNorms[c] -= vj * vj;
      if (colNorms[c] < 0) colNorms[c] = 0;
    }
  }

  // Build Q
  const qCols = econ ? k : m;
  const Q_data = new FloatXArray(m * qCols);
  for (let i = 0; i < Math.min(m, qCols); i++)
    Q_data[colMajorIndex(i, i, m)] = 1;
  for (let j = k - 1; j >= 0; j--) {
    const v = vecs[j],
      tau = taus[j];
    if (tau === 0) continue;
    const len = m - j;
    for (let c = j; c < qCols; c++) {
      let dot = 0;
      for (let i = 0; i < len; i++)
        dot += v[i] * Q_data[colMajorIndex(j + i, c, m)];
      const scale = tau * dot;
      for (let i = 0; i < len; i++)
        Q_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
    }
  }

  // Build R
  const rRows = econ ? k : m;
  const R_out = new FloatXArray(rRows * n);
  for (let r = 0; r < rRows; r++)
    for (let c = r; c < n; c++)
      R_out[colMajorIndex(r, c, rRows)] = R_data[colMajorIndex(r, c, m)];

  // Convert perm to 1-based MATLAB convention
  const permOut = new FloatXArray(n);
  for (let i = 0; i < n; i++) permOut[i] = perm[i] + 1;

  return [
    RTV.tensor(Q_data, [m, qCols]),
    RTV.tensor(R_out, [rRows, n]),
    permResult(permOut, n, econ),
  ];
}

// ── lu ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "lu",
  resolve: (argTypes, nargout) => {
    if (nargout < 0 || nargout > 3) return null;
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (!isNumericJitType(argTypes[0])) return null;
    const t = tensorType();
    if (nargout <= 1)
      return { outputTypes: [t], apply: (args, n) => luApply(args, n) };
    if (nargout === 2)
      return { outputTypes: [t, t], apply: (args, n) => luApply(args, n) };
    return { outputTypes: [t, t, t], apply: (args, n) => luApply(args, n) };
  },
});

function luApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (args.length < 1)
    throw new RuntimeError("lu requires at least 1 argument");
  const outputForm = args.length >= 2 ? parseLuOutputForm(args[1]) : "matrix";
  if (outputForm === null)
    throw new RuntimeError("lu: outputForm must be 'matrix' or 'vector'");
  const A = args[0];
  if (isRuntimeNumber(A)) {
    const val = A as number;
    if (nargout <= 2) {
      const L = RTV.tensor(new FloatXArray([1]), [1, 1]);
      const U = RTV.tensor(new FloatXArray([val]), [1, 1]);
      if (nargout <= 1) return L;
      return [L, U];
    }
    const L = RTV.tensor(new FloatXArray([1]), [1, 1]);
    const U = RTV.tensor(new FloatXArray([val]), [1, 1]);
    if (outputForm === "vector") return [L, U, RTV.num(1)];
    return [L, U, RTV.tensor(new FloatXArray([1]), [1, 1])];
  }
  if (!isRuntimeTensor(A))
    throw new RuntimeError("lu: argument must be numeric");
  const [m, n] = tensorSize2D(A);
  const k = Math.min(m, n);
  const isComplex = A.imag !== undefined;
  let LU_re: FloatXArrayType,
    LU_im: FloatXArrayType | undefined,
    ipiv: Int32Array;
  if (isComplex) {
    const bridge = getEffectiveBridge("lu", "luComplex");
    if (!bridge?.luComplex)
      throw new RuntimeError("lu: complex LU requires the native LAPACK addon");
    const result = bridge.luComplex(toF64(A.data), toF64(A.imag!), m, n);
    LU_re = result.LURe;
    LU_im = result.LUIm;
    ipiv = result.ipiv;
  } else {
    const bridge = getEffectiveBridge("lu", "lu");
    if (!bridge?.lu) throw new RuntimeError("lu: LAPACK bridge not available");
    const result = bridge.lu(toF64(A.data), m, n);
    LU_re = result.LU;
    ipiv = result.ipiv;
  }
  const U_re = new FloatXArray(k * n);
  const U_im = isComplex ? new FloatXArray(k * n) : undefined;
  for (let j = 0; j < n; j++) {
    const imax = Math.min(j, k - 1);
    for (let i = 0; i <= imax; i++) {
      U_re[i + j * k] = LU_re[i + j * m];
      if (U_im && LU_im) U_im[i + j * k] = LU_im[i + j * m];
    }
  }
  if (nargout <= 1)
    return RTV.tensor(
      new FloatXArray(LU_re),
      [m, n],
      LU_im ? new FloatXArray(LU_im) : undefined
    );
  const perm = ipivToPermVector(ipiv, m);
  if (nargout === 2) {
    const L_unit_re = new FloatXArray(m * k);
    const L_unit_im = isComplex ? new FloatXArray(m * k) : undefined;
    for (let j = 0; j < k; j++) {
      L_unit_re[j + j * m] = 1;
      for (let i = j + 1; i < m; i++) {
        L_unit_re[i + j * m] = LU_re[i + j * m];
        if (L_unit_im && LU_im) L_unit_im[i + j * m] = LU_im[i + j * m];
      }
    }
    const L_re = new FloatXArray(m * k);
    const L_im = isComplex ? new FloatXArray(m * k) : undefined;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        L_re[perm[i] + j * m] = L_unit_re[i + j * m];
        if (L_im && L_unit_im) L_im[perm[i] + j * m] = L_unit_im[i + j * m];
      }
    }
    return [RTV.tensor(L_re, [m, k], L_im), RTV.tensor(U_re, [k, n], U_im)];
  }
  // nargout === 3
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
    for (let i = 0; i < m; i++) P_data[i] = perm[i] + 1;
    return [
      RTV.tensor(L_re, [m, k], L_im),
      RTV.tensor(U_re, [k, n], U_im),
      RTV.tensor(P_data, [m, 1]),
    ];
  }
  const P_data = new FloatXArray(m * m);
  for (let i = 0; i < m; i++) P_data[i + perm[i] * m] = 1;
  return [
    RTV.tensor(L_re, [m, k], L_im),
    RTV.tensor(U_re, [k, n], U_im),
    RTV.tensor(P_data, [m, m]),
  ];
}

function parseLuOutputForm(arg: RuntimeValue): "matrix" | "vector" | null {
  if (arg === undefined) return "matrix";
  const s = parseStringArgLower(arg);
  if (s === "vector") return "vector";
  if (s === "matrix") return "matrix";
  return null;
}

function ipivToPermVector(ipiv: Int32Array, m: number): Int32Array {
  const perm = new Int32Array(m);
  for (let i = 0; i < m; i++) perm[i] = i;
  for (let i = 0; i < ipiv.length; i++) {
    const j = ipiv[i] - 1;
    if (j !== i) {
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
  }
  return perm;
}

// ── eig ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "eig",
  resolve: (argTypes, nargout) => {
    if (
      nargout < 0 ||
      nargout > 3 ||
      argTypes.length < 1 ||
      argTypes.length > 3
    )
      return null;
    if (!isNumericJitType(argTypes[0])) return null;
    const c = tensorType(true);
    if (nargout <= 1)
      return { outputTypes: [c], apply: (args, n) => eigApply(args, n) };
    if (nargout === 2)
      return { outputTypes: [c, c], apply: (args, n) => eigApply(args, n) };
    return { outputTypes: [c, c, c], apply: (args, n) => eigApply(args, n) };
  },
});

function eigApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (args.length < 1)
    throw new RuntimeError("eig requires at least 1 argument");
  const A = args[0];
  const { balance, outputForm } = parseEigOptionsRuntime(args);
  if (isRuntimeNumber(A)) {
    const val = A;
    if (nargout <= 1) return RTV.num(val);
    const V = RTV.tensor(new FloatXArray([1]), [1, 1]);
    if (nargout === 2) {
      if (outputForm === "vector") return [V, RTV.num(val)];
      return [V, RTV.tensor(new FloatXArray([val]), [1, 1])];
    }
    const D =
      outputForm === "vector"
        ? RTV.num(val)
        : RTV.tensor(new FloatXArray([val]), [1, 1]);
    return [V, D, RTV.tensor(new FloatXArray([1]), [1, 1])];
  }
  if (!isRuntimeTensor(A))
    throw new RuntimeError("eig: argument must be numeric");
  const [m, n] = tensorSize2D(A);
  if (m !== n) throw new RuntimeError("eig: input must be a square matrix");
  const computeVL = nargout >= 3,
    computeVR = nargout >= 2;
  if (A.imag) {
    const bridge = getEffectiveBridge("eigComplex", "eigComplex");
    if (!bridge.eigComplex)
      throw new RuntimeError(
        "eig: complex eig requires the native LAPACK addon"
      );
    const result = bridge.eigComplex(
      toF64(A.data),
      toF64(A.imag),
      n,
      computeVL,
      computeVR
    );
    if (!result) throw new RuntimeError("eig: complex eig failed");
    const { wRe, wIm, VLRe, VLIm, VRRe, VRIm } = result;
    if (nargout <= 1) return maybeComplexTensor(wRe, [n, 1], wIm);
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
  const bridge = getEffectiveBridge("eig", "eig");
  if (!bridge.eig) throw new RuntimeError("eig: LAPACK bridge not available");
  const result = bridge.eig(toF64(A.data), n, computeVL, computeVR, balance);
  if (!result) throw new RuntimeError("eig: LAPACK eig failed");
  const { wr, wi, VL, VR } = result;
  const hasComplex = wi.some((v: number) => v !== 0);
  if (nargout <= 1) return maybeComplexTensor(wr, [n, 1], wi);
  const Vout =
    computeVR && VR
      ? buildEigenvectorMatrix(VR, wi, n, hasComplex)
      : RTV.tensor(new FloatXArray(n * n), [n, n]);
  const Dout =
    outputForm === "vector"
      ? maybeComplexTensor(wr, [n, 1], wi)
      : buildDiagMatrix(wr, wi, n);
  if (nargout === 2) return [Vout, Dout];
  const Wout =
    computeVL && VL
      ? buildEigenvectorMatrix(VL, wi, n, hasComplex)
      : RTV.tensor(new FloatXArray(n * n), [n, n]);
  return [Vout, Dout, Wout];
}

function parseEigOptionsRuntime(args: RuntimeValue[]): {
  balance: boolean;
  outputForm: "vector" | "matrix";
} {
  let balance = true,
    outputForm: "vector" | "matrix" = "matrix";
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

// ── chol ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "chol",
  resolve: (argTypes, nargout) => {
    if (
      nargout < 0 ||
      nargout > 3 ||
      argTypes.length < 1 ||
      argTypes.length > 3
    )
      return null;
    if (!isNumericJitType(argTypes[0])) return null;
    const t = tensorType();
    if (nargout <= 1)
      return { outputTypes: [t], apply: (args, n) => cholApply(args, n) };
    if (nargout === 2)
      return { outputTypes: [t, NUM], apply: (args, n) => cholApply(args, n) };
    return { outputTypes: [t, NUM, t], apply: (args, n) => cholApply(args, n) };
  },
});

function cholApply(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (args.length < 1)
    throw new RuntimeError("chol requires at least 1 argument");
  let A = args[0];
  const inputIsSparse = isRuntimeSparseMatrix(A);
  if (nargout >= 3 && !inputIsSparse)
    throw new RuntimeError("Third output only available for sparse matrices.");
  if (inputIsSparse)
    A = sparseToDense(
      A as import("../../runtime/types.js").RuntimeSparseMatrix
    );
  let triangle: "upper" | "lower" = "upper";
  let outputForm: "matrix" | "vector" = "matrix";
  if (args.length === 2) {
    const s = parseStringArgLower(args[1]);
    if (s === "upper" || s === "lower") triangle = s;
    else if (s === "matrix" || s === "vector") outputForm = s;
    else throw new RuntimeError("chol: invalid option");
  } else if (args.length === 3) {
    const tri = parseStringArgLower(args[1]);
    if (tri !== "upper" && tri !== "lower")
      throw new RuntimeError("chol: triangle must be 'upper' or 'lower'");
    triangle = tri;
    const form = parseStringArgLower(args[2]);
    if (form !== "matrix" && form !== "vector")
      throw new RuntimeError("chol: outputForm must be 'matrix' or 'vector'");
    outputForm = form as "matrix" | "vector";
  }
  const upper = triangle === "upper";
  if (isRuntimeNumber(A)) {
    const val = A as number;
    if (val <= 0) {
      if (nargout >= 2)
        return [RTV.tensor(new FloatXArray([0]), [1, 1]), RTV.num(1)];
      throw new RuntimeError("chol: Matrix must be positive definite.");
    }
    const r = Math.sqrt(val);
    if (nargout >= 2)
      return [RTV.tensor(new FloatXArray([r]), [1, 1]), RTV.num(0)];
    return RTV.tensor(new FloatXArray([r]), [1, 1]);
  }
  if (!isRuntimeTensor(A))
    throw new RuntimeError("chol: argument must be numeric");
  const [m, n] = tensorSize2D(A);
  if (m !== n) throw new RuntimeError("chol: Matrix must be square.");
  const isComplex = A.imag !== undefined;
  let R_re: FloatXArrayType,
    R_im: FloatXArrayType | undefined,
    info_val: number;
  if (isComplex) {
    const bridge = getEffectiveBridge("chol", "cholComplex");
    if (!bridge?.cholComplex)
      throw new RuntimeError(
        "chol: complex Cholesky requires the native LAPACK addon"
      );
    const result = bridge.cholComplex(toF64(A.data), toF64(A.imag!), n, upper);
    R_re = result.RRe;
    R_im = result.RIm;
    info_val = result.info;
  } else {
    const bridge = getEffectiveBridge("chol", "chol");
    if (!bridge?.chol)
      throw new RuntimeError("chol: LAPACK bridge not available");
    const result = bridge.chol(toF64(A.data), n, upper);
    R_re = result.R;
    info_val = result.info;
  }
  if (nargout >= 2 && info_val > 0) {
    const k = info_val - 1;
    if (nargout >= 3) {
      // 3-output partial result
      let partialR: RuntimeValue;
      if (upper) {
        const pr = new FloatXArray(k * n);
        const pi = isComplex ? new FloatXArray(k * n) : undefined;
        for (let j = 0; j < n; j++)
          for (let i = 0; i <= Math.min(j, k - 1); i++) {
            pr[i + j * k] = (R_re as FloatXArrayType)[i + j * n];
            if (pi && R_im)
              pi[i + j * k] = (R_im as FloatXArrayType)[i + j * n];
          }
        partialR = RTV.tensor(pr, [k, n], pi);
      } else {
        const pr = new FloatXArray(n * k);
        const pi = isComplex ? new FloatXArray(n * k) : undefined;
        for (let j = 0; j < k; j++)
          for (let i = j; i < n; i++) {
            pr[i + j * n] = (R_re as FloatXArrayType)[i + j * n];
            if (pi && R_im)
              pi[i + j * n] = (R_im as FloatXArrayType)[i + j * n];
          }
        partialR = RTV.tensor(pr, [n, k], pi);
      }
      const permOut =
        outputForm === "vector"
          ? RTV.tensor(
              new FloatXArray(Array.from({ length: n }, (_, i) => i + 1)),
              [n, 1]
            )
          : (() => {
              const P = new FloatXArray(n * n);
              for (let i = 0; i < n; i++) P[i + i * n] = 1;
              return RTV.tensor(P, [n, n]);
            })();
      return [partialR, RTV.num(info_val), permOut];
    }
    // 2-output partial result
    let partialR: RuntimeValue;
    if (upper) {
      const pr = new FloatXArray(k * k);
      const pi = isComplex ? new FloatXArray(k * k) : undefined;
      for (let j = 0; j < k; j++)
        for (let i = 0; i <= j; i++) {
          pr[i + j * k] = (R_re as FloatXArrayType)[i + j * n];
          if (pi && R_im) pi[i + j * k] = (R_im as FloatXArrayType)[i + j * n];
        }
      partialR = RTV.tensor(pr, [k, k], pi);
    } else {
      const pr = new FloatXArray(k * k);
      const pi = isComplex ? new FloatXArray(k * k) : undefined;
      for (let j = 0; j < k; j++)
        for (let i = j; i < k; i++) {
          pr[i + j * k] = (R_re as FloatXArrayType)[i + j * n];
          if (pi && R_im) pi[i + j * k] = (R_im as FloatXArrayType)[i + j * n];
        }
      partialR = RTV.tensor(pr, [k, k], pi);
    }
    return [partialR, RTV.num(info_val)];
  }
  if (nargout >= 2) {
    const R = RTV.tensor(
      new FloatXArray(R_re as ArrayLike<number>),
      [n, n],
      R_im ? new FloatXArray(R_im as ArrayLike<number>) : undefined
    );
    if (nargout >= 3) {
      const permOut =
        outputForm === "vector"
          ? RTV.tensor(
              new FloatXArray(Array.from({ length: n }, (_, i) => i + 1)),
              [n, 1]
            )
          : (() => {
              const P = new FloatXArray(n * n);
              for (let i = 0; i < n; i++) P[i + i * n] = 1;
              return RTV.tensor(P, [n, n]);
            })();
      return [R, RTV.num(0), permOut];
    }
    return [R, RTV.num(0)];
  }
  if (info_val > 0)
    throw new RuntimeError("chol: Matrix must be positive definite.");
  return RTV.tensor(
    new FloatXArray(R_re as ArrayLike<number>),
    [n, n],
    R_im ? new FloatXArray(R_im as ArrayLike<number>) : undefined
  );
}

// ── linsolve ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "linsolve",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (argTypes.length !== 2 || nargout > 1) return null;
        if (!isNumericJitType(argTypes[0]) || !isNumericJitType(argTypes[1]))
          return null;
        return [tensorType()];
      },
      apply: args => {
        if (args.length < 2)
          throw new RuntimeError("linsolve requires 2 arguments");
        const rawA = args[0],
          rawB = args[1];
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
          const ARe = A.data,
            AIm = A.imag ?? new FloatXArray(A.data.length);
          const BRe = B.data,
            BIm = B.imag ?? new FloatXArray(B.data.length);
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
  ],
});

// ── cond ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cond",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length < 1 || argTypes.length > 2)
          return null;
        if (!isNumericJitType(argTypes[0])) return null;
        return [NUM];
      },
      apply: args => {
        if (args.length < 1 || args.length > 2)
          throw new RuntimeError("cond requires 1 or 2 arguments");
        const A = args[0];
        if (isRuntimeNumber(A)) return RTV.num(A === 0 ? Infinity : 1);
        if (isRuntimeComplexNumber(A))
          return RTV.num(A.re === 0 && A.im === 0 ? Infinity : 1);
        if (!isRuntimeTensor(A))
          throw new RuntimeError("cond: argument must be numeric");
        let p: number | string = 2;
        if (args.length >= 2) {
          const pArg = args[1];
          if (isRuntimeString(pArg) || isRuntimeChar(pArg)) {
            const pStr = parseStringArgLower(pArg);
            if (pStr === "fro") p = "fro";
            else throw new RuntimeError("cond: string argument must be 'fro'");
          } else p = toNumber(pArg);
        }
        if (p === 2) {
          const svdIb = getIBuiltin("svd")!;
          const res = svdIb.resolve(args.map(inferJitType), 1);
          if (!res) throw new RuntimeError("cond: svd resolve failed");
          const sVec = res.apply([A], 1);
          if (!isRuntimeTensor(sVec as RuntimeValue))
            throw new RuntimeError("cond: unexpected svd result");
          const s = (sVec as RuntimeTensor).data;
          let sMax = -Infinity,
            sMin = Infinity;
          for (let i = 0; i < s.length; i++) {
            if (s[i] > sMax) sMax = s[i];
            if (s[i] < sMin) sMin = s[i];
          }
          if (sMin === 0) return RTV.num(Infinity);
          return RTV.num(sMax / sMin);
        }
        const [rows, cols] = tensorSize2D(A);
        if (rows !== cols)
          throw new RuntimeError(
            "cond: matrix must be square for non-2 condition number"
          );
        const normArg: RuntimeValue =
          p === "fro" ? RTV.string("fro") : RTV.num(p as number);
        const normA = toNumber(normApply([A, normArg]));
        const invIb = getIBuiltin("inv")!;
        const invRes = invIb.resolve(args.slice(0, 1).map(inferJitType), 1);
        if (!invRes) throw new RuntimeError("cond: inv resolve failed");
        const invA = invRes.apply([A], 1) as RuntimeValue;
        const normInvA = toNumber(normApply([invA, normArg]));
        return RTV.num(normA * normInvA);
      },
    },
  ],
});

// ── rank ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "rank",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length < 1 || argTypes.length > 2)
          return null;
        if (!isNumericJitType(argTypes[0])) return null;
        return [NUM];
      },
      apply: args => {
        if (args.length < 1 || args.length > 2)
          throw new RuntimeError("rank requires 1 or 2 arguments");
        const A = args[0];
        if (isRuntimeNumber(A)) return RTV.num(A === 0 ? 0 : 1);
        if (isRuntimeComplexNumber(A))
          return RTV.num(A.re === 0 && A.im === 0 ? 0 : 1);
        if (!isRuntimeTensor(A))
          throw new RuntimeError("rank: argument must be numeric");
        const [rows, cols] = tensorSize2D(A);
        const svdIb = getIBuiltin("svd")!;
        const res = svdIb.resolve([inferJitType(A)], 1);
        if (!res) throw new RuntimeError("rank: svd resolve failed");
        const sVec = res.apply([A], 1) as RuntimeValue;
        if (!isRuntimeTensor(sVec))
          throw new RuntimeError("rank: unexpected svd result");
        if (sVec.imag)
          throw new RuntimeError("rank: singular values must be real");
        const s = sVec.data;
        let tol: number;
        if (args.length >= 2) tol = toNumber(args[1]);
        else {
          let sMax = 0;
          for (let i = 0; i < s.length; i++) if (s[i] > sMax) sMax = s[i];
          tol = Math.max(rows, cols) * epsOf(sMax);
        }
        let k = 0;
        for (let i = 0; i < s.length; i++) if (s[i] > tol) k++;
        return RTV.num(k);
      },
    },
  ],
});

function epsOf(x: number): number {
  if (!isFinite(x) || x === 0) return Number.EPSILON;
  const ax = Math.abs(x);
  return Math.pow(2, Math.floor(Math.log2(ax)) - 52);
}

// ── pinv ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "pinv",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length < 1 || argTypes.length > 2)
          return null;
        if (!isNumericJitType(argTypes[0])) return null;
        const a = argTypes[0];
        if (a.kind === "number" || a.kind === "boolean") return [NUM];
        return [tensorType()];
      },
      apply: args => {
        if (args.length < 1 || args.length > 2)
          throw new RuntimeError("pinv requires 1 or 2 arguments");
        const A = args[0];
        if (isRuntimeNumber(A)) return RTV.num(A === 0 ? 0 : 1 / A);
        if (!isRuntimeTensor(A))
          throw new RuntimeError("pinv: argument must be numeric");
        const [m, n] = tensorSize2D(A);
        const k = Math.min(m, n);
        const bridge = getLapackBridge();
        if (!bridge || !bridge.svd) return pinvFallback(A.data, m, n);
        const svdResult = bridge.svd(toF64(A.data), m, n, true, true);
        if (!svdResult || !svdResult.U || !svdResult.V)
          throw new RuntimeError("pinv: SVD computation failed");
        const { U, S, V } = svdResult;
        const tol =
          args.length >= 2
            ? isRuntimeNumber(args[1])
              ? args[1]
              : 0
            : Math.max(m, n) * S[0] * 2.220446049250313e-16;
        const result = new FloatXArray(n * m);
        for (let i = 0; i < n; i++)
          for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let l = 0; l < k; l++)
              if (S[l] > tol) sum += V[l * n + i] * (1 / S[l]) * U[l * m + j];
            result[j * n + i] = sum;
          }
        return RTV.tensor(result, [n, m]);
      },
    },
  ],
});

function pinvFallback(
  data: FloatXArrayType,
  m: number,
  n: number
): ReturnType<typeof RTV.tensor> {
  let allZero = true;
  for (let i = 0; i < data.length; i++)
    if (data[i] !== 0) {
      allZero = false;
      break;
    }
  if (allZero) return RTV.tensor(new FloatXArray(n * m), [n, m]);
  const AT = new FloatXArray(n * m);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) AT[i * n + j] = data[j * m + i];
  if (m >= n) {
    const ATA = new FloatXArray(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < m; k++) sum += data[i * m + k] * data[j * m + k];
        ATA[j * n + i] = sum;
      }
    const augmented = new FloatXArray(n * (n + m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) augmented[j * n + i] = ATA[j * n + i];
      for (let j = 0; j < m; j++) augmented[(n + j) * n + i] = AT[j * n + i];
    }
    gaussJordanEliminate(augmented, n, n + m);
    const result = new FloatXArray(n * m);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++)
        result[j * n + i] = augmented[(n + j) * n + i];
    return RTV.tensor(result, [n, m]);
  }
  const AAT = new FloatXArray(m * m);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += data[k * m + i] * data[k * m + j];
      AAT[j * m + i] = sum;
    }
  const augmented = new FloatXArray(m * 2 * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) augmented[j * m + i] = AAT[j * m + i];
    augmented[(m + i) * m + i] = 1;
  }
  gaussJordanEliminate(augmented, m, 2 * m);
  const AATinv = new FloatXArray(m * m);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) AATinv[j * m + i] = augmented[(m + j) * m + i];
  const result = new FloatXArray(n * m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) sum += AT[k * n + i] * AATinv[j * m + k];
      result[j * n + i] = sum;
    }
  return RTV.tensor(result, [n, m]);
}

// ── kron ────────────────────────────────────────────────────────────────

defineBuiltin({
  name: "kron",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length !== 2) return null;
        if (!isNumericJitType(argTypes[0]) || !isNumericJitType(argTypes[1]))
          return null;
        return [tensorType()];
      },
      apply: args => {
        if (args.length !== 2)
          throw new RuntimeError("kron requires 2 arguments");
        const coerce = (v: RuntimeValue): RuntimeTensor => {
          if (isRuntimeNumber(v))
            return RTV.tensor(new FloatXArray([v]), [1, 1]);
          if (isRuntimeSparseMatrix(v)) return sparseToDense(v);
          if (isRuntimeTensor(v)) return v;
          throw new RuntimeError("kron: arguments must be numeric");
        };
        const A = coerce(args[0]);
        const B = coerce(args[1]);
        const [m, n] = tensorSize2D(A);
        const [p, q] = tensorSize2D(B);
        const rows = m * p,
          cols = n * q;
        const result = new FloatXArray(rows * cols);
        for (let ja = 0; ja < n; ja++)
          for (let ia = 0; ia < m; ia++) {
            const aVal = A.data[ia + ja * m];
            for (let jb = 0; jb < q; jb++)
              for (let ib = 0; ib < p; ib++) {
                const bVal = B.data[ib + jb * p];
                result[ia * p + ib + (ja * q + jb) * rows] = aVal * bVal;
              }
          }
        return RTV.tensor(result, [rows, cols]);
      },
    },
  ],
});

// ── blkdiag ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "blkdiag",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length === 0) return null;
        if (!argTypes.every(isNumericJitType)) return null;
        return [tensorType()];
      },
      apply: args => {
        if (args.length === 0)
          throw new RuntimeError("blkdiag requires at least 1 argument");
        const blocks = args.map(a => {
          if (isRuntimeNumber(a))
            return RTV.tensor(new FloatXArray([a]), [1, 1]);
          if (!isRuntimeTensor(a))
            throw new RuntimeError("blkdiag: arguments must be numeric");
          return a;
        });
        let totalRows = 0,
          totalCols = 0;
        const dims: [number, number][] = [];
        for (const block of blocks) {
          const [m, n] = tensorSize2D(block);
          dims.push([m, n]);
          totalRows += m;
          totalCols += n;
        }
        const result = new FloatXArray(totalRows * totalCols);
        let rowOffset = 0,
          colOffset = 0;
        for (let k = 0; k < blocks.length; k++) {
          const [m, n] = dims[k];
          const data = blocks[k].data;
          for (let j = 0; j < n; j++)
            for (let i = 0; i < m; i++)
              result[rowOffset + i + (colOffset + j) * totalRows] =
                data[i + j * m];
          rowOffset += m;
          colOffset += n;
        }
        return RTV.tensor(result, [totalRows, totalCols]);
      },
    },
  ],
});

// ── pagemtimes ──────────────────────────────────────────────────────────

defineBuiltin({
  name: "pagemtimes",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1) return null;
        if (argTypes.length !== 2 && argTypes.length !== 4) return null;
        const xIdx = 0,
          yIdx = argTypes.length === 4 ? 2 : 1;
        if (
          !isNumericJitType(argTypes[xIdx]) ||
          !isNumericJitType(argTypes[yIdx])
        )
          return null;
        return [tensorType()];
      },
      apply: args => {
        // Delegate to legacy apply — the full implementation is complex
        let X: RuntimeValue, Y: RuntimeValue;
        let transpX = "none",
          transpY = "none";
        if (args.length === 2) {
          X = args[0];
          Y = args[1];
        } else if (args.length === 4) {
          X = args[0];
          Y = args[2];
          if (
            !(isRuntimeString(args[1]) || isRuntimeChar(args[1])) ||
            !(isRuntimeString(args[3]) || isRuntimeChar(args[3]))
          )
            throw new RuntimeError(
              "pagemtimes: transpose options must be strings"
            );
          transpX = rstr(
            args[1] as
              | import("../../runtime/types.js").RuntimeString
              | import("../../runtime/types.js").RuntimeChar
          );
          transpY = rstr(
            args[3] as
              | import("../../runtime/types.js").RuntimeString
              | import("../../runtime/types.js").RuntimeChar
          );
        } else throw new RuntimeError("pagemtimes requires 2 or 4 arguments");

        const xT: RuntimeTensor = isRuntimeNumber(X)
          ? (RTV.tensor(
              new FloatXArray([X as number]),
              [1, 1]
            ) as RuntimeTensor)
          : (X as RuntimeTensor);
        const yT: RuntimeTensor = isRuntimeNumber(Y)
          ? (RTV.tensor(
              new FloatXArray([Y as number]),
              [1, 1]
            ) as RuntimeTensor)
          : (Y as RuntimeTensor);
        if (!isRuntimeTensor(xT) || !isRuntimeTensor(yT))
          throw new RuntimeError("pagemtimes: arguments must be numeric");

        const xShape = xT.shape.length < 2 ? [1, xT.shape[0] || 1] : xT.shape;
        const yShape = yT.shape.length < 2 ? [1, yT.shape[0] || 1] : yT.shape;
        let xRows = xShape[0],
          xCols = xShape[1];
        let yRows = yShape[0],
          yCols = yShape[1];
        if (transpX === "transpose" || transpX === "ctranspose")
          [xRows, xCols] = [xCols, xRows];
        if (transpY === "transpose" || transpY === "ctranspose")
          [yRows, yCols] = [yCols, yRows];
        if (xCols !== yRows)
          throw new RuntimeError(
            `pagemtimes: inner matrix dimensions must agree: ${xCols} vs ${yRows}`
          );

        const xExtra = xShape.slice(2),
          yExtra = yShape.slice(2);
        const maxExtraDims = Math.max(xExtra.length, yExtra.length);
        const outExtra: number[] = [];
        for (let d = 0; d < maxExtraDims; d++) {
          const xd = d < xExtra.length ? xExtra[d] : 1;
          const yd = d < yExtra.length ? yExtra[d] : 1;
          if (xd !== yd && xd !== 1 && yd !== 1)
            throw new RuntimeError(
              "pagemtimes: dimensions beyond first two must be compatible"
            );
          outExtra.push(Math.max(xd, yd));
        }
        const outShape = [xRows, yCols, ...outExtra];
        const pageSize = xRows * yCols;
        const xPageSize = xShape[0] * xShape[1],
          yPageSize = yShape[0] * yShape[1];
        const totalPages = outExtra.reduce((a, b) => a * b, 1);
        const result = new FloatXArray(pageSize * totalPages);
        const needTranspX = transpX === "transpose" || transpX === "ctranspose";
        const needTranspY = transpY === "transpose" || transpY === "ctranspose";
        const xTransBuf = needTranspX ? new FloatXArray(xPageSize) : null;
        const yTransBuf = needTranspY ? new FloatXArray(yPageSize) : null;
        const xExtraStrides: number[] = [],
          yExtraStrides: number[] = [];
        let xStride = 1,
          yStride = 1;
        for (let d = 0; d < maxExtraDims; d++) {
          xExtraStrides.push(xStride);
          yExtraStrides.push(yStride);
          xStride *= d < xExtra.length ? xExtra[d] : 1;
          yStride *= d < yExtra.length ? yExtra[d] : 1;
        }
        for (let p = 0; p < totalPages; p++) {
          let xPageIdx = 0,
            yPageIdx = 0,
            rem = p;
          for (let d = maxExtraDims - 1; d >= 0; d--) {
            const outD = outExtra[d],
              idx = rem % outD;
            rem = (rem - idx) / outD;
            const xd = d < xExtra.length ? xExtra[d] : 1;
            const yd = d < yExtra.length ? yExtra[d] : 1;
            xPageIdx += (xd === 1 ? 0 : idx) * xExtraStrides[d];
            yPageIdx += (yd === 1 ? 0 : idx) * yExtraStrides[d];
          }
          const xOff = xPageIdx * xPageSize,
            yOff = yPageIdx * yPageSize,
            outOff = p * pageSize;
          let aData = xT.data,
            aOff = xOff,
            aR = xShape[0],
            aC = xShape[1];
          let bData = yT.data,
            bOff = yOff,
            bR = yShape[0],
            bC = yShape[1];
          if (needTranspX && xTransBuf) {
            for (let j = 0; j < xShape[1]; j++)
              for (let i = 0; i < xShape[0]; i++)
                xTransBuf[j + i * xShape[1]] =
                  xT.data[xOff + i + j * xShape[0]];
            aData = xTransBuf;
            aOff = 0;
            aR = xRows;
            aC = xCols;
          }
          if (needTranspY && yTransBuf) {
            for (let j = 0; j < yShape[1]; j++)
              for (let i = 0; i < yShape[0]; i++)
                yTransBuf[j + i * yShape[1]] =
                  yT.data[yOff + i + j * yShape[0]];
            bData = yTransBuf;
            bOff = 0;
            bR = yRows;
            bC = yCols;
          }
          for (let j = 0; j < bC; j++)
            for (let i = 0; i < aR; i++) {
              let sum = 0;
              for (let kk = 0; kk < aC; kk++)
                sum += aData[aOff + i + kk * aR] * bData[bOff + kk + j * bR];
              result[outOff + i + j * aR] = sum;
            }
        }
        while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
          outShape.pop();
        return RTV.tensor(result, outShape);
      },
    },
  ],
});

// ── pagetranspose ───────────────────────────────────────────────────────

defineBuiltin({
  name: "pagetranspose",
  cases: [
    {
      match: (argTypes, nargout) => {
        if (nargout > 1 || argTypes.length !== 1) return null;
        if (argTypes[0].kind !== "tensor") return null;
        return [tensorType(argTypes[0].isComplex)];
      },
      apply: args => {
        if (args.length !== 1)
          throw new RuntimeError("pagetranspose requires exactly 1 argument");
        const X = args[0];
        if (!isRuntimeTensor(X))
          throw new RuntimeError(
            "pagetranspose: input must be a numeric array"
          );
        const xShape = X.shape;
        const rows = xShape[0],
          cols = xShape.length >= 2 ? xShape[1] : 1;
        const extraDims = xShape.slice(2);
        const totalPages = extraDims.reduce((a: number, b: number) => a * b, 1);
        const pageSize = rows * cols;
        const outData = new FloatXArray(X.data.length);
        const outShape = [cols, rows, ...extraDims];
        for (let p = 0; p < totalPages; p++) {
          const inOff = p * pageSize,
            outOff = p * pageSize;
          for (let j = 0; j < cols; j++)
            for (let i = 0; i < rows; i++)
              outData[outOff + j + i * cols] = X.data[inOff + i + j * rows];
        }
        while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
          outShape.pop();
        return RTV.tensor(outData, outShape);
      },
    },
  ],
});

// ── qz ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "qz",
  resolve: (argTypes, nargout) => {
    if (nargout !== 4 && nargout !== 6) return null;
    if (argTypes.length < 2 || argTypes.length > 3) return null;
    if (!isNumericJitType(argTypes[0]) || !isNumericJitType(argTypes[1]))
      return null;
    const c = tensorType(true);
    const types = nargout === 4 ? [c, c, c, c] : [c, c, c, c, c, c];
    return { outputTypes: types, apply: (args, n) => qzApply(args, n) };
  },
});

function qzApply(args: RuntimeValue[], nargout: number): RuntimeValue[] {
  if (args.length < 2)
    throw new RuntimeError("qz requires at least 2 arguments");
  let mode: "real" | "complex" = "complex";
  if (args.length >= 3) {
    const modeStr = parseStringArgLower(args[2]);
    if (modeStr === "complex") mode = "complex";
    else if (modeStr === "real") mode = "real";
    else throw new RuntimeError(`qz: unknown mode '${modeStr}'`);
  }
  const A = args[0],
    B = args[1];
  if (!isRuntimeTensor(A) || !isRuntimeTensor(B))
    throw new RuntimeError("qz: arguments must be numeric matrices");
  const [mA, nA] = tensorSize2D(A);
  const [mB, nB] = tensorSize2D(B);
  if (mA !== nA) throw new RuntimeError("qz: A must be a square matrix");
  if (mB !== nB) throw new RuntimeError("qz: B must be a square matrix");
  if (mA !== mB) throw new RuntimeError("qz: A and B must be the same size");
  const n = mA;
  const computeEigvecs = nargout >= 6;
  const isComplex = !!(A.imag || B.imag) || mode === "complex";
  if (isComplex) {
    const nn = n * n;
    const aRe = toF64(A.data),
      aIm = A.imag ? toF64(A.imag) : new Float64Array(nn);
    const bRe = toF64(B.data),
      bIm = B.imag ? toF64(B.imag) : new Float64Array(nn);
    const bridge = getEffectiveBridge("qzComplex", "qzComplex");
    if (!bridge.qzComplex)
      throw new RuntimeError(
        "qz: complex LAPACK bridge not available (requires native addon)"
      );
    const result = bridge.qzComplex(aRe, aIm, bRe, bIm, n, computeEigvecs);
    if (!result) throw new RuntimeError("qz: complex QZ failed");
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
    if (nargout === 4) return [AAout, BBout, Qout, Zout];
    const { VRe, VIm, WRe, WIm } = result;
    if (!VRe || !VIm || !WRe || !WIm)
      throw new RuntimeError("qz: failed to compute generalized eigenvectors");
    return [
      AAout,
      BBout,
      Qout,
      Zout,
      RTV.tensor(new FloatXArray(VRe), [n, n], new FloatXArray(VIm)),
      RTV.tensor(new FloatXArray(WRe), [n, n], new FloatXArray(WIm)),
    ];
  }
  const bridge = getEffectiveBridge("qz", "qz");
  if (!bridge.qz) throw new RuntimeError("qz: LAPACK bridge not available");
  const result = bridge.qz(toF64(A.data), toF64(B.data), n, computeEigvecs);
  if (!result) throw new RuntimeError("qz: real QZ failed");
  const AAout = RTV.tensor(new FloatXArray(result.AA), [n, n]);
  const BBout = RTV.tensor(new FloatXArray(result.BB), [n, n]);
  const Qout = RTV.tensor(new FloatXArray(result.Q), [n, n]);
  const Zout = RTV.tensor(new FloatXArray(result.Z), [n, n]);
  if (nargout === 4) return [AAout, BBout, Qout, Zout];
  const { alphai, V, W } = result;
  if (!V || !W)
    throw new RuntimeError("qz: failed to compute generalized eigenvectors");
  let hasComplex = false;
  for (let i = 0; i < n; i++)
    if (Math.abs(alphai[i]) > 0) {
      hasComplex = true;
      break;
    }
  return [
    AAout,
    BBout,
    Qout,
    Zout,
    buildEigenvectorMatrix(V, alphai, n, hasComplex),
    buildEigenvectorMatrix(W, alphai, n, hasComplex),
  ];
}

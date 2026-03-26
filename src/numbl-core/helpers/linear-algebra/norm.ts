/**
 * Vector/matrix norm builtin function
 */

import { RTV, toNumber, RuntimeError } from "../../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeComplexNumber,
  isRuntimeTensor,
  isRuntimeSparseMatrix,
  isRuntimeChar,
} from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { sparseToDense } from "../sparse-arithmetic.js";

function normImpl(args: import("../../runtime/types.js").RuntimeValue[]) {
  if (args.length < 1)
    throw new RuntimeError("norm requires at least 1 argument");
  const v = args[0];
  if (isRuntimeNumber(v)) return RTV.num(Math.abs(v));
  if (isRuntimeComplexNumber(v)) return RTV.num(Math.hypot(v.re, v.im));
  if (isRuntimeSparseMatrix(v))
    return normImpl([sparseToDense(v), ...args.slice(1)]);
  if (!isRuntimeTensor(v))
    throw new RuntimeError("norm: argument must be numeric");
  return normImplTensor(v, args);
}

function normImplTensor(
  v: import("../../runtime/types.js").RuntimeTensor,
  args: import("../../runtime/types.js").RuntimeValue[]
) {
  // Determine if vector or matrix
  const shape = v.shape;
  const rows = shape[0] || 1;
  const cols = shape.length >= 2 ? shape[1] : 1;
  const isVec =
    shape.length <= 2
      ? rows === 1 || cols === 1
      : v.data.length === Math.max(...shape); // N-d with all but one dim singleton
  // Parse the second argument: can be a number or a string ('inf', 'fro')
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
    // For vectors, 'fro' is the same as 2-norm
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
    let s = 0;
    for (let i = 0; i < v.data.length; i++) {
      const a = imag ? Math.hypot(v.data[i], imag[i]) : Math.abs(v.data[i]);
      s += Math.pow(a, vp);
    }
    return RTV.num(Math.pow(s, 1 / vp));
  }
  // Matrix norms
  if (p === "fro" || (typeof p === "number" && isNaN(p))) {
    // Frobenius norm
    let s = 0;
    for (let i = 0; i < v.data.length; i++) {
      const re = v.data[i];
      const im = imag ? imag[i] : 0;
      s += re * re + im * im;
    }
    return RTV.num(Math.sqrt(s));
  }
  if (p === 1) {
    // 1-norm: max column sum of absolute values
    let maxColSum = 0;
    for (let j = 0; j < cols; j++) {
      let colSum = 0;
      for (let i = 0; i < rows; i++) {
        const idx = j * rows + i; // column-major
        colSum += imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
      }
      maxColSum = Math.max(maxColSum, colSum);
    }
    return RTV.num(maxColSum);
  }
  if (p === Infinity) {
    // Inf-norm: max row sum of absolute values
    let maxRowSum = 0;
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        const idx = j * rows + i; // column-major
        rowSum += imag
          ? Math.hypot(v.data[idx], imag[idx])
          : Math.abs(v.data[idx]);
      }
      maxRowSum = Math.max(maxRowSum, rowSum);
    }
    return RTV.num(maxRowSum);
  }
  if (p === 2) {
    // 2-norm: largest singular value
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

export function registerNorm(): void {
  register(
    "norm",
    builtinSingle(args => normImpl(args), {
      outputType: { kind: "Number" },
    })
  );
}

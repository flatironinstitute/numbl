/**
 * gmres: restarted GMRES on the cores in helpers/gmres.ts, for real and
 * complex systems with matrix or function-handle operators and
 * preconditioners.
 */

import { type RuntimeValue, RTV, RuntimeError } from "../../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeFunction,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeComplexNumber,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { Runtime } from "../../runtime/runtime.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin } from "./builtin.js";
import { builtinError } from "./args.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import {
  gmresCore,
  gmresCoreComplex,
  luSolveInPlace,
  complexLuSolveInPlace,
  type MatvecFn,
  type PrecSolveFn,
  type ComplexMatvecFn,
  type ComplexPrecSolveFn,
  type ComplexVec,
} from "../../helpers/gmres.js";
import { dgetrf as _dgetrf } from "../../../ts-lapack/src/SRC/dgetrf.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { toF64 } from "../../helpers/check-helpers.js";

builtin({
  name: "gmres",
  nargout: 5,
  mfile: true,
  apply: (args, nargout, ctx) => _gmresImpl(ctx.rt, nargout, args),
});

// ── gmres implementation ──────────────────────────────────────────────────

function _gmresImpl(
  rt: Runtime,
  nargout: number,
  args: RuntimeValue[]
): RuntimeValue | RuntimeValue[] {
  if (args.length < 2)
    throw builtinError("MATLAB:gmres:NumInputs", "Not enough input arguments.");

  const Aarg = args[0];
  const bArg = args[1];

  // Extract b and determine n
  if (
    !isRuntimeTensor(bArg) &&
    !isRuntimeNumber(bArg) &&
    !isRuntimeComplexNumber(bArg)
  )
    throw new RuntimeError("gmres: b must be a numeric vector");
  let n: number;
  if (isRuntimeTensor(bArg)) {
    n = bArg.data.length;
  } else {
    n = 1;
  }

  // Parse optional arguments
  const restartArg = args.length >= 3 ? args[2] : null;
  const tolArg = args.length >= 4 ? args[3] : null;
  const maxitArg = args.length >= 5 ? args[4] : null;
  const M1arg = args.length >= 6 ? args[5] : null;
  const M2arg = args.length >= 7 ? args[6] : null;
  const x0Arg = args.length >= 8 ? args[7] : null;

  // Parse scalar options
  let restart = n;
  if (restartArg !== null && !_isEmpty(restartArg)) {
    restart = _toNum(restartArg, "gmres: restart");
  }
  if (restart <= 0 || restart > n) restart = n;
  const noRestart = restart === n;

  const tol =
    tolArg !== null && !_isEmpty(tolArg) ? _toNum(tolArg, "gmres: tol") : 1e-6;

  let maxit: number;
  if (maxitArg !== null && !_isEmpty(maxitArg)) {
    maxit = _toNum(maxitArg, "gmres: maxit");
  } else {
    maxit = noRestart ? 1 : Math.max(1, Math.min(Math.ceil(n / restart), 10));
  }
  if (noRestart && maxitArg === null) restart = Math.min(n, 10);

  // Detect whether any input is complex
  const isComplex =
    _isComplexArg(Aarg) ||
    _isComplexArg(bArg) ||
    (M1arg !== null && _isComplexArg(M1arg)) ||
    (M2arg !== null && _isComplexArg(M2arg)) ||
    (x0Arg !== null && _isComplexArg(x0Arg));

  let flag: number;
  let relres: number;
  let iter: [number, number];
  let resvec: Float64Array;
  let xTensor: RuntimeValue;

  if (isComplex) {
    // ── Complex path ──────────────────────────────────────────────────────
    const bCV = _toComplexVec(bArg, n);
    const x0CV =
      x0Arg !== null && !_isEmpty(x0Arg) ? _toComplexVec(x0Arg, n) : null;

    const allMatrices =
      _isMatrixArg(Aarg) && _isMatrixArg(M1arg) && _isMatrixArg(M2arg);

    if (allMatrices) {
      const { re: ARe, im: AIm } = _toComplexMatrix(Aarg, n * n);
      const M1cv = _extractComplexMatrix(M1arg, n * n);
      const M2cv = _extractComplexMatrix(M2arg, n * n);

      const r = getLapackBridge().gmresComplex(
        ARe,
        AIm,
        n,
        bCV.re,
        bCV.im,
        restart,
        tol,
        maxit,
        M1cv?.re ?? null,
        M1cv?.im ?? null,
        M2cv?.re ?? null,
        M2cv?.im ?? null,
        x0CV?.re ?? null,
        x0CV?.im ?? null
      );
      flag = r.flag;
      relres = r.relres;
      iter = [r.iter[0], r.iter[1]];
      resvec = r.resvec;
      xTensor = RTV.tensor(
        allocFloat64Array(r.xRe),
        [n, 1],
        allocFloat64Array(r.xIm)
      );
    } else {
      // Function handle or mixed: use gmresCoreComplex with JS callbacks
      const matvec = _makeComplexMatvec(rt, Aarg, n);
      const precSolve = _makeComplexPrecSolve(rt, M1arg, M2arg, n);
      const r = gmresCoreComplex(
        matvec,
        precSolve,
        bCV,
        n,
        restart,
        tol,
        maxit,
        x0CV
      );
      flag = r.flag;
      relres = r.relres;
      iter = r.iter;
      resvec = r.resvec;
      xTensor = RTV.tensor(
        allocFloat64Array(r.x.re),
        [n, 1],
        allocFloat64Array(r.x.im)
      );
    }
  } else {
    // ── Real path ─────────────────────────────────────────────────────────
    let bData: Float64Array;
    if (isRuntimeNumber(bArg)) {
      bData = allocFloat64Array([bArg as number]);
    } else {
      bData = toF64((bArg as RuntimeTensor).data);
    }
    let x0: Float64Array | null = null;
    if (x0Arg !== null && !_isEmpty(x0Arg)) {
      if (isRuntimeTensor(x0Arg)) x0 = toF64(x0Arg.data);
      else if (isRuntimeNumber(x0Arg))
        x0 = allocFloat64Array([x0Arg as number]);
    }

    let xResult: Float64Array;
    const Ais_matrix = _isMatrixArg(Aarg);
    const M1is_matrix = _isMatrixArg(M1arg);
    const M2is_matrix = _isMatrixArg(M2arg);

    if (Ais_matrix && M1is_matrix && M2is_matrix) {
      const Adata = _extractRealMatrix(Aarg)!;
      const M1data = _extractMatrix(M1arg);
      const M2data = _extractMatrix(M2arg);

      const result = getLapackBridge().gmres(
        Adata,
        n,
        bData,
        restart,
        tol,
        maxit,
        M1data,
        M2data,
        x0
      );
      xResult = result.x;
      flag = result.flag;
      relres = result.relres;
      iter = [result.iter[0], result.iter[1]];
      resvec = result.resvec;
    } else {
      const matvec: MatvecFn = _makeMatvec(rt, Aarg, n);
      const precSolve: PrecSolveFn | null = _makePrecSolve(rt, M1arg, M2arg, n);
      const r = gmresCore(matvec, precSolve, bData, n, restart, tol, maxit, x0);
      xResult = r.x;
      flag = r.flag;
      relres = r.relres;
      iter = r.iter;
      resvec = r.resvec;
    }
    xTensor = RTV.tensor(allocFloat64Array(xResult), [n, 1]);
  }

  // Print convergence message when flag output is not requested
  if (nargout <= 1) {
    if (flag === 0) {
      rt.output(
        `gmres converged at iteration ${iter[1]} to a solution with relative residual ${relres.toExponential(1)}.\n`
      );
    } else {
      rt.output(
        `gmres stopped at iteration ${iter[1]} without converging to the desired tolerance ${tol}\nbecause the maximum number of iterations was reached.\nThe iterate returned (number ${iter[1]}) has relative residual ${relres.toExponential(1)}.\n`
      );
    }
  }

  if (nargout <= 1) return xTensor;
  if (nargout === 2) return [xTensor, flag];
  if (nargout === 3) return [xTensor, flag, relres];
  if (nargout === 4) {
    const iterTensor = RTV.tensor(
      allocFloat64Array([iter[0], iter[1]]),
      [1, 2]
    );
    return [xTensor, flag, relres, iterTensor];
  }
  // nargout >= 5
  const iterTensor = RTV.tensor(allocFloat64Array([iter[0], iter[1]]), [1, 2]);
  const resvecTensor = RTV.tensor(allocFloat64Array(resvec), [
    resvec.length,
    1,
  ]);
  return [xTensor, flag, relres, iterTensor, resvecTensor];
}

// ── gmres helpers ─────────────────────────────────────────────────────────

function _isEmpty(v: RuntimeValue): boolean {
  if (isRuntimeTensor(v) && v.data.length === 0) return true;
  return false;
}

function _toNum(v: RuntimeValue, ctx: string): number {
  if (isRuntimeNumber(v)) return v as number;
  if (isRuntimeTensor(v) && v.data.length === 1) return v.data[0];
  throw new RuntimeError(`${ctx} must be a scalar`);
}

function _extractMatrix(arg: RuntimeValue | null): Float64Array | null {
  if (arg === null || _isEmpty(arg)) return null;
  if (isRuntimeNumber(arg)) return allocFloat64Array([arg as number]);
  if (isRuntimeSparseMatrix(arg)) {
    const dense = sparseToDense(arg);
    return toF64(dense.data);
  }
  if (isRuntimeTensor(arg)) return toF64(arg.data);
  return null;
}

/** Create a matvec callback. If A is a function handle, calls it via rt.index. */
function _makeMatvec(rt: Runtime, Aarg: RuntimeValue, n: number): MatvecFn {
  if (isRuntimeFunction(Aarg)) {
    return (x: Float64Array): Float64Array => {
      const xTensor = RTV.tensor(allocFloat64Array(x), [n, 1]);
      const resultRaw = rt.index(Aarg, [xTensor], 1);
      const rv = ensureRuntimeValue(resultRaw as RuntimeValue);
      if (isRuntimeNumber(rv)) return allocFloat64Array([rv as number]);
      if (isRuntimeTensor(rv)) return toF64(rv.data);
      throw new RuntimeError("gmres: A(x) must return a numeric vector");
    };
  }
  // Matrix case
  let Adata: Float64Array;
  if (isRuntimeNumber(Aarg)) {
    Adata = allocFloat64Array([Aarg as number]);
  } else if (isRuntimeSparseMatrix(Aarg)) {
    Adata = toF64(sparseToDense(Aarg).data);
  } else if (isRuntimeTensor(Aarg)) {
    Adata = toF64(Aarg.data);
  } else {
    throw new RuntimeError("gmres: A must be a matrix or function handle");
  }
  return (x: Float64Array): Float64Array => {
    const y = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += Adata[i + j * n] * x[j];
      y[i] = s;
    }
    return y;
  };
}

/** Create a preconditioner solve callback for M = M1*M2. */
function _makePrecSolve(
  rt: Runtime,
  M1arg: RuntimeValue | null,
  M2arg: RuntimeValue | null,
  n: number
): PrecSolveFn | null {
  const hasM1 = M1arg !== null && !_isEmpty(M1arg);
  const hasM2 = M2arg !== null && !_isEmpty(M2arg);
  if (!hasM1 && !hasM2) return null;

  // Build individual solve functions
  const solve1 = hasM1 ? _makeSinglePrecSolve(rt, M1arg!, n) : null;
  const solve2 = hasM2 ? _makeSinglePrecSolve(rt, M2arg!, n) : null;

  return (r: Float64Array): Float64Array => {
    let z = r;
    if (solve1) z = solve1(z);
    if (solve2) z = solve2(z);
    return z;
  };
}

function _makeSinglePrecSolve(
  rt: Runtime,
  Marg: RuntimeValue,
  n: number
): (r: Float64Array) => Float64Array {
  if (isRuntimeFunction(Marg)) {
    return (r: Float64Array): Float64Array => {
      const rTensor = RTV.tensor(allocFloat64Array(r), [n, 1]);
      const resultRaw = rt.index(Marg, [rTensor], 1);
      const rv = ensureRuntimeValue(resultRaw as RuntimeValue);
      if (isRuntimeNumber(rv)) return allocFloat64Array([rv as number]);
      if (isRuntimeTensor(rv)) return toF64(rv.data);
      throw new RuntimeError("gmres: M(x) must return a numeric vector");
    };
  }
  // Matrix: pre-factor with LU and apply solve
  let Mdata: Float64Array;
  if (isRuntimeNumber(Marg)) {
    Mdata = allocFloat64Array([Marg as number]);
  } else if (isRuntimeSparseMatrix(Marg)) {
    Mdata = toF64(sparseToDense(Marg).data);
  } else if (isRuntimeTensor(Marg)) {
    Mdata = toF64(Marg.data);
  } else {
    throw new RuntimeError(
      "gmres: preconditioner must be a matrix or function handle"
    );
  }
  // Pre-factor
  const LU = allocFloat64Array(Mdata);
  const ipiv = new Int32Array(n);
  const info = _dgetrf(n, n, LU, n, ipiv);
  if (info > 0)
    throw new RuntimeError("gmres: preconditioner matrix is singular");

  return (r: Float64Array): Float64Array => {
    const z = allocFloat64Array(r);
    luSolveInPlace(n, LU, ipiv, z);
    return z;
  };
}

// ── Complex gmres helpers ─────────────────────────────────────────────────

function _isComplexArg(v: RuntimeValue | null): boolean {
  if (v === null) return false;
  if (isRuntimeComplexNumber(v)) return true;
  if (isRuntimeTensor(v) && v.imag) return true;
  if (isRuntimeSparseMatrix(v) && v.pi) return true;
  return false;
}

function _isMatrixArg(v: RuntimeValue | null): boolean {
  if (v === null) return true; // null counts as "matrix" (absent)
  if (_isEmpty(v)) return true;
  return (
    isRuntimeTensor(v) ||
    isRuntimeNumber(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeSparseMatrix(v)
  );
}

function _extractRealMatrix(arg: RuntimeValue | null): Float64Array | null {
  if (arg === null) return null;
  if (isRuntimeNumber(arg)) return allocFloat64Array([arg as number]);
  if (isRuntimeSparseMatrix(arg)) return toF64(sparseToDense(arg).data);
  if (isRuntimeTensor(arg)) return toF64(arg.data);
  return null;
}

function _toComplexVec(v: RuntimeValue, n: number): ComplexVec {
  if (isRuntimeComplexNumber(v)) {
    return { re: allocFloat64Array([v.re]), im: allocFloat64Array([v.im]) };
  }
  if (isRuntimeTensor(v)) {
    return {
      re: allocFloat64Array(toF64(v.data)),
      im: v.imag ? allocFloat64Array(toF64(v.imag)) : allocFloat64Array(n),
    };
  }
  if (isRuntimeNumber(v)) {
    const re = allocFloat64Array(1);
    re[0] = v as number;
    return { re, im: allocFloat64Array(1) };
  }
  throw new RuntimeError("gmres: cannot convert argument to complex vector");
}

function _toComplexMatrix(
  v: RuntimeValue,
  len: number
): { re: Float64Array; im: Float64Array } {
  if (isRuntimeTensor(v)) {
    return {
      re: allocFloat64Array(toF64(v.data)),
      im: v.imag ? allocFloat64Array(toF64(v.imag)) : allocFloat64Array(len),
    };
  }
  if (isRuntimeSparseMatrix(v)) {
    const dense = sparseToDense(v);
    return {
      re: allocFloat64Array(toF64(dense.data)),
      im: dense.imag
        ? allocFloat64Array(toF64(dense.imag))
        : allocFloat64Array(len),
    };
  }
  if (isRuntimeNumber(v)) {
    const re = allocFloat64Array(1);
    re[0] = v as number;
    return { re, im: allocFloat64Array(1) };
  }
  throw new RuntimeError("gmres: cannot convert to complex matrix");
}

function _extractComplexMatrix(
  arg: RuntimeValue | null,
  len: number
): { re: Float64Array; im: Float64Array } | null {
  if (arg === null || _isEmpty(arg)) return null;
  return _toComplexMatrix(arg, len);
}

function _makeComplexMatvec(
  rt: Runtime,
  Aarg: RuntimeValue,
  n: number
): ComplexMatvecFn {
  if (isRuntimeFunction(Aarg)) {
    return (x: ComplexVec): ComplexVec => {
      const xTensor = RTV.tensor(
        allocFloat64Array(x.re),
        [n, 1],
        allocFloat64Array(x.im)
      );
      const resultRaw = rt.index(Aarg, [xTensor], 1);
      const rv = ensureRuntimeValue(resultRaw as RuntimeValue);
      if (isRuntimeTensor(rv)) {
        return {
          re: allocFloat64Array(toF64(rv.data)),
          im: rv.imag
            ? allocFloat64Array(toF64(rv.imag))
            : allocFloat64Array(n),
        };
      }
      if (isRuntimeComplexNumber(rv)) {
        return {
          re: allocFloat64Array([rv.re]),
          im: allocFloat64Array([rv.im]),
        };
      }
      if (isRuntimeNumber(rv)) {
        const re = allocFloat64Array(1);
        re[0] = rv as number;
        return { re, im: allocFloat64Array(1) };
      }
      throw new RuntimeError("gmres: A(x) must return a numeric vector");
    };
  }
  // Matrix case — complex matvec
  const { re: ARe, im: AIm } = _toComplexMatrix(Aarg, n * n);
  return (x: ComplexVec): ComplexVec => {
    const yRe = allocFloat64Array(n);
    const yIm = allocFloat64Array(n);
    for (let i = 0; i < n; i++) {
      let sR = 0,
        sI = 0;
      for (let j = 0; j < n; j++) {
        const aR = ARe[i + j * n],
          aI = AIm[i + j * n];
        sR += aR * x.re[j] - aI * x.im[j];
        sI += aR * x.im[j] + aI * x.re[j];
      }
      yRe[i] = sR;
      yIm[i] = sI;
    }
    return { re: yRe, im: yIm };
  };
}

function _makeComplexPrecSolve(
  rt: Runtime,
  M1arg: RuntimeValue | null,
  M2arg: RuntimeValue | null,
  n: number
): ComplexPrecSolveFn | null {
  const hasM1 = M1arg !== null && !_isEmpty(M1arg);
  const hasM2 = M2arg !== null && !_isEmpty(M2arg);
  if (!hasM1 && !hasM2) return null;

  const solve1 = hasM1 ? _makeSingleComplexPrecSolve(rt, M1arg!, n) : null;
  const solve2 = hasM2 ? _makeSingleComplexPrecSolve(rt, M2arg!, n) : null;

  return (r: ComplexVec): ComplexVec => {
    let z = r;
    if (solve1) z = solve1(z);
    if (solve2) z = solve2(z);
    return z;
  };
}

function _makeSingleComplexPrecSolve(
  rt: Runtime,
  Marg: RuntimeValue,
  n: number
): (r: ComplexVec) => ComplexVec {
  if (isRuntimeFunction(Marg)) {
    return (r: ComplexVec): ComplexVec => {
      const rTensor = RTV.tensor(
        allocFloat64Array(r.re),
        [n, 1],
        allocFloat64Array(r.im)
      );
      const resultRaw = rt.index(Marg, [rTensor], 1);
      const rv = ensureRuntimeValue(resultRaw as RuntimeValue);
      if (isRuntimeTensor(rv)) {
        return {
          re: allocFloat64Array(toF64(rv.data)),
          im: rv.imag
            ? allocFloat64Array(toF64(rv.imag))
            : allocFloat64Array(n),
        };
      }
      throw new RuntimeError("gmres: M(x) must return a numeric vector");
    };
  }
  // Matrix: pre-factor with complex LU
  const { re: MRe, im: MIm } = _toComplexMatrix(Marg, n * n);
  const { LURe, LUIm, ipiv } = getLapackBridge().luComplex(MRe, MIm, n, n);

  return (r: ComplexVec): ComplexVec => {
    const zRe = allocFloat64Array(r.re);
    const zIm = allocFloat64Array(r.im);
    complexLuSolveInPlace(n, LURe, LUIm, ipiv, zRe, zIm);
    return { re: zRe, im: zIm };
  };
}

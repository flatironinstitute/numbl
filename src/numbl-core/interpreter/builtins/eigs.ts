/**
 * eigs: a subset of eigenvalues and eigenvectors, computed with the dense
 * eig and then selected per k and sigma.
 */

import {
  type RuntimeValue,
  RTV,
  toNumber,
  toString,
  RuntimeError,
} from "../../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeFunction,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeStruct,
  isRuntimeSparseMatrix,
  isRuntimeComplexNumber,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import type { Runtime } from "../../runtime/runtime.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin } from "./builtin.js";
import { maybeComplexTensor } from "../../helpers/check-helpers.js";
import {
  selectEigsIndices,
  normalizeSigmaString,
  type SigmaSpec,
} from "../../helpers/eigs-select.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";

builtin({
  name: "eigs",
  nargin: [1, Infinity],
  nargout: 3,
  mfile: true,
  apply: (args, nargout, ctx) => _eigsImpl(ctx.rt, nargout, args),
});

// ── eigs implementation ───────────────────────────────────────────────
//
// `eigs` returns a subset of eigenvalues/eigenvectors. numbl has no
// iterative Krylov solver, so it piggybacks on the dense `eig`: compute
// every eigenvalue, then keep the requested `k` of them per `sigma`. This
// matches `eig(full(A))` (which the MATLAB docs themselves recommend for
// small dense matrices) while accepting the full `eigs` call surface.

/** Real/imag parts of a numeric runtime value as parallel arrays. */
function eigsReIm(v: RuntimeValue): { re: number[]; im: number[] } {
  if (isRuntimeNumber(v)) return { re: [v], im: [0] };
  if (isRuntimeComplexNumber(v)) return { re: [v.re], im: [v.im] };
  if (isRuntimeTensor(v)) {
    const re = Array.from(v.data);
    const im = v.imag ? Array.from(v.imag) : re.map(() => 0);
    return { re, im };
  }
  throw new RuntimeError("eigs: expected a numeric value");
}

/** Coerce a dispatch result (number or tensor) to a RuntimeTensor. */
function eigsAsTensor(v: unknown): RuntimeTensor {
  const rv = ensureRuntimeValue(v as RuntimeValue);
  if (isRuntimeTensor(rv)) return rv;
  if (isRuntimeNumber(rv))
    return RTV.tensor(allocFloat64Array([rv]), [1, 1]) as RuntimeTensor;
  if (isRuntimeComplexNumber(rv))
    return RTV.tensor(
      allocFloat64Array([rv.re]),
      [1, 1],
      allocFloat64Array([rv.im])
    ) as RuntimeTensor;
  throw new RuntimeError("eigs: expected a numeric matrix");
}

/** Square size of a matrix argument; 1 for a scalar. */
function eigsMatrixSize(A: RuntimeValue): number {
  if (isRuntimeNumber(A) || isRuntimeComplexNumber(A)) return 1;
  if (isRuntimeTensor(A)) {
    const rows = A.shape[0] ?? 1;
    const cols = A.shape[1] ?? 1;
    if (rows !== cols)
      throw new RuntimeError("eigs: input matrix must be square");
    return rows;
  }
  throw new RuntimeError("eigs: input must be a numeric matrix");
}

/** A 1×1/scalar numeric argument (a candidate for k or a numeric sigma). */
function eigsIsScalar(v: RuntimeValue): boolean {
  return (
    isRuntimeNumber(v) ||
    isRuntimeComplexNumber(v) ||
    (isRuntimeTensor(v) && v.data.length === 1)
  );
}

/** A matrix-shaped argument in the B position: a non-scalar tensor, or the
 *  empty matrix `[]` (which disambiguates to the standard problem). */
function eigsIsMatrixArg(v: RuntimeValue): boolean {
  if (!isRuntimeTensor(v)) return false;
  if (v.data.length === 0) return true;
  const rows = v.shape[0] ?? 1;
  const cols = v.shape[1] ?? 1;
  return rows > 1 || cols > 1;
}

function eigsToBool(v: RuntimeValue): boolean {
  if (isRuntimeLogical(v)) return v;
  if (isRuntimeNumber(v)) return v !== 0;
  if (isRuntimeTensor(v) && v.data.length >= 1) return v.data[0] !== 0;
  return false;
}

/** Parse a sigma argument into its canonical spec. */
function eigsParseSigma(v: RuntimeValue): SigmaSpec {
  if (isRuntimeString(v) || isRuntimeChar(v)) {
    const s = typeof v === "string" ? v : v.value;
    const kind = normalizeSigmaString(s);
    if (!kind) throw new RuntimeError(`eigs: unknown sigma option '${s}'`);
    return { kind };
  }
  let re: number;
  let im = 0;
  if (isRuntimeComplexNumber(v)) {
    re = v.re;
    im = v.im;
  } else if (isRuntimeNumber(v)) {
    re = v;
  } else if (isRuntimeTensor(v) && v.data.length >= 1) {
    re = v.data[0];
    im = v.imag ? v.imag[0] : 0;
  } else {
    throw new RuntimeError("eigs: invalid sigma argument");
  }
  // Numeric sigma of 0 is documented as equivalent to 'smallestabs'.
  if (re === 0 && im === 0) return { kind: "smallestabs" };
  return { kind: "scalar", re, im };
}

/** Reconstruct the operator matrix M from Afun by applying it to each
 *  identity column: M(:,j) = Afun(e_j). */
function eigsReconstructOperator(
  rt: Runtime,
  afun: RuntimeValue,
  n: number
): RuntimeValue {
  const re = allocFloat64Array(n * n);
  const im = allocFloat64Array(n * n);
  let anyImag = false;
  for (let j = 0; j < n; j++) {
    const e = allocFloat64Array(n);
    e[j] = 1;
    const col = ensureRuntimeValue(
      rt.index(afun, [RTV.tensor(e, [n, 1])], 1) as RuntimeValue
    );
    const { re: cre, im: cim } = eigsReIm(col);
    for (let i = 0; i < n; i++) {
      re[j * n + i] = cre[i] ?? 0;
      im[j * n + i] = cim[i] ?? 0;
      if (im[j * n + i] !== 0) anyImag = true;
    }
  }
  return anyImag ? RTV.tensor(re, [n, n], im) : RTV.tensor(re, [n, n]);
}

/** Recover the actual matrix A from the Afun operator M, inverting the
 *  shift/inverse implied by sigma so the same selection logic applies. */
function eigsRecoverAfunMatrix(
  rt: Runtime,
  M: RuntimeValue,
  sigma: SigmaSpec,
  B: RuntimeValue | null,
  n: number
): RuntimeValue {
  if (sigma.kind === "smallestabs") {
    // Afun(x) = A\x ⇒ M = inv(A) ⇒ A = inv(M).
    return eigsAsTensor(rt.dispatch("inv", 1, [M]));
  }
  if (sigma.kind === "scalar") {
    // Afun(x) = (A - sigma*S)\x ⇒ M = inv(A - sigma*S) ⇒ A = inv(M) + sigma*S,
    // where S is B (generalized) or I (standard).
    const base = eigsAsTensor(rt.dispatch("inv", 1, [M]));
    const shiftRe = allocFloat64Array(n * n);
    const shiftIm = allocFloat64Array(n * n);
    if (B && isRuntimeTensor(B)) {
      for (let i = 0; i < n * n; i++) {
        shiftRe[i] =
          (B.data[i] ?? 0) * sigma.re - (B.imag?.[i] ?? 0) * sigma.im;
        shiftIm[i] =
          (B.data[i] ?? 0) * sigma.im + (B.imag?.[i] ?? 0) * sigma.re;
      }
    } else {
      for (let d = 0; d < n; d++) {
        shiftRe[d * n + d] = sigma.re;
        shiftIm[d * n + d] = sigma.im;
      }
    }
    const shift = maybeComplexTensor(shiftRe, [n, n], shiftIm);
    return eigsAsTensor(rt.dispatch("plus", 1, [base, shift]));
  }
  // Default / 'largest*' text: Afun(x) = A*x ⇒ M = A.
  return M;
}

/** B(s,s) = R'·R reconstruction for the 'IsCholesky' option. */
function eigsReconstructCholesky(
  R: RuntimeValue,
  perm: number[] | null,
  n: number
): RuntimeValue {
  if (!isRuntimeTensor(R))
    throw new RuntimeError("eigs: Cholesky factor must be a matrix");
  const r = R.data;
  // RtR = R' * R (R is upper triangular n×n, column-major).
  const rtr = allocFloat64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let kk = 0; kk < n; kk++) s += r[i * n + kk] * r[j * n + kk];
      rtr[j * n + i] = s;
    }
  }
  if (!perm) return RTV.tensor(rtr, [n, n]);
  // B(perm,perm) = RtR ⇒ scatter back through the permutation (1-based).
  const b = allocFloat64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const pi = (perm[i] ?? i + 1) - 1;
      const pj = (perm[j] ?? j + 1) - 1;
      b[pj * n + pi] = rtr[j * n + i];
    }
  }
  return RTV.tensor(b, [n, n]);
}

function _eigsImpl(
  rt: Runtime,
  nargout: number,
  args: RuntimeValue[]
): RuntimeValue | RuntimeValue[] {
  const margs = [...args];

  // eigs computes on dense matrices internally; densify any sparse inputs
  // (e.g. a graph Laplacian) so the A / B positions are plain tensors.
  for (let i = 0; i < margs.length; i++) {
    const mi = margs[i];
    if (isRuntimeSparseMatrix(mi)) margs[i] = sparseToDense(mi);
  }

  // 1. Afun vs matrix input, and the matrix size n.
  let afun: RuntimeValue | null = null;
  let A: RuntimeValue | null = null;
  let n: number;
  let pos: number;
  if (isRuntimeFunction(margs[0])) {
    afun = margs[0];
    if (margs.length < 2 || !eigsIsScalar(margs[1]))
      throw new RuntimeError(
        "eigs: a function handle must be followed by the matrix size n"
      );
    n = Math.floor(toNumber(margs[1]));
    pos = 2;
  } else {
    A = margs[0];
    n = eigsMatrixSize(A);
    pos = 1;
  }

  // 2. Optional B (generalized problem); [] selects the standard problem.
  let B: RuntimeValue | null = null;
  if (pos < margs.length && eigsIsMatrixArg(margs[pos])) {
    const b = margs[pos];
    if (!(isRuntimeTensor(b) && b.data.length === 0)) B = b;
    pos++;
  }

  // 3. Optional k.
  let kReq: number | null = null;
  if (pos < margs.length && eigsIsScalar(margs[pos])) {
    kReq = Math.max(1, Math.floor(toNumber(margs[pos])));
    pos++;
  }

  // 4. Optional sigma — a scalar, or a recognized sigma keyword. A string
  //    that is not a sigma keyword is left for the options loop (it is the
  //    start of name-value pairs).
  let sigma: SigmaSpec = { kind: "largestabs" };
  if (pos < margs.length) {
    const a = margs[pos];
    if (isRuntimeString(a) || isRuntimeChar(a)) {
      const s = typeof a === "string" ? a : a.value;
      if (normalizeSigmaString(s)) {
        sigma = eigsParseSigma(a);
        pos++;
      }
    } else if (eigsIsScalar(a)) {
      sigma = eigsParseSigma(a);
      pos++;
    }
  }

  // 5. Options — opts struct or name-value pairs. Only the B-related options
  //    affect an exact dense computation; convergence/display options are
  //    accepted and ignored.
  let isCholesky = false;
  let cholPerm: number[] | null = null;
  const applyOption = (key: string, val: RuntimeValue) => {
    const lk = key.toLowerCase();
    if (lk === "ischolesky" || lk === "cholb") isCholesky = eigsToBool(val);
    else if (lk === "choleskypermutation" || lk === "permb") {
      if (isRuntimeTensor(val)) cholPerm = Array.from(val.data);
    }
  };
  while (pos < margs.length) {
    const a = margs[pos];
    if (isRuntimeStruct(a)) {
      for (const [key, val] of a.fields) applyOption(key, val);
      pos++;
    } else if (pos + 1 < margs.length) {
      applyOption(
        isRuntimeChar(margs[pos])
          ? (margs[pos] as { value: string }).value
          : String(toString(margs[pos])),
        margs[pos + 1]
      );
      pos += 2;
    } else {
      break;
    }
  }

  const k = kReq == null ? Math.min(6, n) : Math.min(kReq, n);

  // 6. Build the effective standard matrix whose eigenpairs we select from.
  let actualA: RuntimeValue;
  if (afun) {
    const M = eigsReconstructOperator(rt, afun, n);
    actualA = eigsRecoverAfunMatrix(rt, M, sigma, B, n);
  } else {
    actualA = A as RuntimeValue;
  }
  let effective = actualA;
  if (B) {
    const Bmat = isCholesky ? eigsReconstructCholesky(B, cholPerm, n) : B;
    // A*V = B*V*D ⇔ (B\A)*V = V*D.
    effective = eigsAsTensor(rt.dispatch("mldivide", 1, [Bmat, actualA]));
  }

  // 7. Dense eig, then select the subset.
  if (nargout <= 1) {
    const d = ensureRuntimeValue(rt.dispatch("eig", 1, [effective]));
    const { re, im } = eigsReIm(d);
    const order = selectEigsIndices(re, im, k, sigma);
    const outRe = allocFloat64Array(order.length);
    const outIm = allocFloat64Array(order.length);
    for (let c = 0; c < order.length; c++) {
      outRe[c] = re[order[c]];
      outIm[c] = im[order[c]];
    }
    return maybeComplexTensor(outRe, [order.length, 1], outIm);
  }

  const res = rt.dispatch("eig", 2, [effective]) as RuntimeValue[];
  const Vall = eigsAsTensor(res[0]);
  const Dall = eigsAsTensor(res[1]);
  const p = Dall.shape[0] ?? 1;
  const vrows = Vall.shape[0] ?? p;
  const re: number[] = [];
  const im: number[] = [];
  for (let j = 0; j < p; j++) {
    re.push(Dall.data[j * p + j]);
    im.push(Dall.imag ? Dall.imag[j * p + j] : 0);
  }
  const order = selectEigsIndices(re, im, k, sigma);
  const kk = order.length;

  const Vre = allocFloat64Array(vrows * kk);
  const Vim = allocFloat64Array(vrows * kk);
  const Dre = allocFloat64Array(kk * kk);
  const Dim = allocFloat64Array(kk * kk);
  for (let c = 0; c < kk; c++) {
    const s = order[c];
    for (let r = 0; r < vrows; r++) {
      Vre[c * vrows + r] = Vall.data[s * vrows + r];
      if (Vall.imag) Vim[c * vrows + r] = Vall.imag[s * vrows + r];
    }
    Dre[c * kk + c] = re[s];
    Dim[c * kk + c] = im[s];
  }
  const V = maybeComplexTensor(Vre, [vrows, kk], Vim);
  const D = maybeComplexTensor(Dre, [kk, kk], Dim);
  // Exact dense eig always "converges": flag 0.
  if (nargout === 2) return [V, D];
  return [V, D, RTV.num(0)];
}

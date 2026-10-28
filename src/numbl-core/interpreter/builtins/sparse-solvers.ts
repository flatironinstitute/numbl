/**
 * Sparse direct-solver builtins: fill-reducing orderings (symrcm, amd,
 * colamd), random sparse matrices (sprand, sprandn, sprandsym), and the
 * `decomposition` object that stores factors for repeated solves.
 *
 * The factorizations themselves live in helpers/sparse-direct.ts; chol, lu,
 * kron, inv and the operators `\` and `/` dispatch to them from
 * linear-algebra.ts and helpers/arithmetic.ts.
 */

import type { RuntimeValue, RuntimeSparseMatrix } from "../../runtime/types.js";
import {
  RuntimeClassInstance,
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RTV, RuntimeError, tensorSize2D } from "../../runtime/index.js";
import { toNumber } from "../../runtime/convert.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { parseStringArgLower, toF64 } from "../../helpers/check-helpers.js";
import { rngRandom, boxMullerRandom } from "../../helpers/prng.js";
import {
  sparseToDense,
  sparseTranspose,
} from "../../helpers/sparse-arithmetic.js";
import {
  denseToSparse,
  denseDataToSparse,
  minimumDegreeOrdering,
  reverseCuthillMcKee,
  columnOrdering,
  permVectorTensor,
  sparseChol,
  sparseLu,
  sparsePermute,
  sparseIsUpper,
  sparseIsLower,
  sparseIsSymmetric,
  sparseSolveUpper,
  sparseSolveLower,
  luSolve,
  luSolveTransposed,
  identityPerm,
  SINGULAR_WARNING,
  type LuResult,
} from "../../helpers/sparse-direct.js";
import { emitWarning } from "../../helpers/warn.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { builtin } from "./builtin.js";
import { builtinError, realScalarArg } from "./args.js";
import { errorFn, floatMatrixArg } from "./linalg-args.js";

type Sp = RuntimeSparseMatrix;

/** A double or logical matrix argument as CSC. */
function toSparseArg(v: RuntimeValue): Sp {
  const a = floatMatrixArg(
    v,
    errorFn(
      "MATLAB:math:mustBeFloatOrLogical",
      "Invalid data type. Argument must be double, single, or logical."
    )
  );
  return isRuntimeSparseMatrix(a) ? a : denseToSparse(a);
}

// ── Orderings ─────────────────────────────────────────────────────────────

builtin({
  name: "symrcm",
  help: {
    signatures: ["p = symrcm(A)"],
    description:
      "Reverse Cuthill-McKee ordering of the pattern of A + A'. A(p,p) tends to have a smaller bandwidth than A.",
  },
  nargin: 1,
  apply: args => permVectorTensor(reverseCuthillMcKee(toSparseArg(args[0]))),
});

builtin({
  name: "amd",
  help: {
    signatures: ["p = amd(A)"],
    description:
      "Minimum degree ordering of the pattern of A + A'. chol(A(p,p)) tends to be sparser than chol(A). numbl uses an exact minimum degree ordering, so p can differ from MATLAB's approximate minimum degree permutation.",
  },
  // amd(A, opts): the options are accepted and ignored.
  nargin: [1, 2],
  apply: args => permVectorTensor(minimumDegreeOrdering(toSparseArg(args[0]))),
});

builtin({
  name: "colamd",
  help: {
    signatures: ["p = colamd(A)"],
    description:
      "Column ordering for LU: minimum degree ordering of the pattern of A'*A. lu(A(:,p)) tends to produce sparser factors than lu(A).",
  },
  // colamd(A, knobs): the knobs are accepted and ignored.
  nargin: [1, 2],
  nargout: 2,
  apply: (args, nargout) => {
    if (nargout >= 2)
      throw new RuntimeError(
        "colamd: the statistics output [p, stats] = colamd(...) is not supported"
      );
    return permVectorTensor(columnOrdering(toSparseArg(args[0])));
  },
});

// ── Random sparse matrices ────────────────────────────────────────────────

/** k distinct integers in [0, total), sorted ascending. */
function sampleDistinct(total: number, k: number): Int32Array {
  if (k >= total) {
    const all = new Int32Array(total);
    for (let i = 0; i < total; i++) all[i] = i;
    return all;
  }
  let picked: Int32Array;
  if (k > total / 3) {
    // Partial Fisher-Yates over the full index range.
    const idx = new Int32Array(total);
    for (let i = 0; i < total; i++) idx[i] = i;
    for (let i = 0; i < k; i++) {
      const j = i + Math.floor(rngRandom() * (total - i));
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    picked = idx.slice(0, k);
  } else {
    const set = new Set<number>();
    while (set.size < k) set.add(Math.floor(rngRandom() * total));
    picked = Int32Array.from(set);
  }
  return picked.sort();
}

/** Random sparse matrix with a given pattern (linear indices) and values. */
function sparseFromLinearIndices(
  m: number,
  n: number,
  idx: Int32Array,
  gen: () => number
): Sp {
  const nnz = idx.length;
  const ir = new Int32Array(nnz);
  const jc = new Int32Array(n + 1);
  const pr = allocFloat64Array(nnz);
  for (let k = 0; k < nnz; k++) {
    const col = Math.floor(idx[k] / m);
    ir[k] = idx[k] - col * m;
    jc[col + 1]++;
    pr[k] = gen();
  }
  for (let c = 0; c < n; c++) jc[c + 1] += jc[c];
  return RTV.sparseMatrix(m, n, ir, jc, pr);
}

function registerSprand(name: string, gen: () => number): void {
  builtin({
    name,
    help: {
      signatures: [`R = ${name}(m, n, density)`, `R = ${name}(S)`],
      description:
        `Random m-by-n sparse matrix with approximately density*m*n ` +
        (name === "sprand" ? "uniformly" : "normally") +
        ` distributed nonzeros, or a matrix with the pattern of S and random values.`,
    },
    nargin: [1, 4],
    mfile: true,
    apply: args => {
      if (args.length === 1) {
        const S = toSparseArg(args[0]);
        const pr = allocFloat64Array(S.pr.length);
        for (let k = 0; k < pr.length; k++) pr[k] = gen();
        return RTV.sparseMatrix(
          S.m,
          S.n,
          new Int32Array(S.ir),
          new Int32Array(S.jc),
          pr
        );
      }
      if (args.length === 2)
        throw builtinError(`MATLAB:${name}:TwoInputs`, "Invalid syntax.");
      if (args.length > 3)
        throw new RuntimeError(
          `${name}: the reciprocal-condition form ${name}(m, n, density, rc) is not supported`
        );
      const m = Math.round(realScalarArg(args[0]));
      const n = Math.round(realScalarArg(args[1]));
      const density = realScalarArg(args[2]);
      if (!(density >= 0 && density <= 1))
        throw new RuntimeError(`${name}: density must be in [0, 1]`);
      const total = m * n;
      const k = Math.min(total, Math.round(density * total));
      return sparseFromLinearIndices(m, n, sampleDistinct(total, k), gen);
    },
  });
}

registerSprand("sprand", rngRandom);
registerSprand("sprandn", boxMullerRandom);

builtin({
  name: "sprandsym",
  help: {
    signatures: ["R = sprandsym(n, density)", "R = sprandsym(S)"],
    description:
      "Symmetric random n-by-n sparse matrix with approximately density*n*n normally distributed nonzeros, or a symmetric matrix with the pattern of the lower triangle of S.",
  },
  nargin: [1, 4],
  mfile: true,
  apply: args => {
    if (args.length > 2)
      throw new RuntimeError(
        "sprandsym: the condition-number forms sprandsym(n, density, rc, ...) are not supported"
      );
    let n: number;
    let lowerIdx: Int32Array;
    if (args.length === 1) {
      const S = toSparseArg(args[0]);
      if (S.m !== S.n) throw new RuntimeError("sprandsym: S must be square");
      n = S.n;
      const idx: number[] = [];
      for (let j = 0; j < n; j++)
        for (let k = S.jc[j]; k < S.jc[j + 1]; k++)
          if (S.ir[k] >= j) idx.push(j * n + S.ir[k]);
      lowerIdx = Int32Array.from(idx);
    } else {
      n = Math.round(realScalarArg(args[0]));
      const density = realScalarArg(args[1]);
      if (!(density >= 0 && density <= 1))
        throw new RuntimeError("sprandsym: density must be in [0, 1]");
      // Sample within the lower triangle (including the diagonal); each
      // off-diagonal entry is mirrored, so about density*n*n total.
      const tri = (n * (n + 1)) / 2;
      const k = Math.min(tri, Math.round((density * n * n) / 2));
      const triIdx = sampleDistinct(tri, k);
      const lin: number[] = [];
      for (let t = 0; t < triIdx.length; t++) {
        // Map triangle index -> (row >= col) in column-major order.
        let r = triIdx[t];
        let col = 0;
        while (r >= n - col) {
          r -= n - col;
          col++;
        }
        lin.push(col * n + col + r);
      }
      lowerIdx = Int32Array.from(lin).sort();
    }
    const lower = sparseFromLinearIndices(n, n, lowerIdx, boxMullerRandom);
    // Mirror: S = L + L' - diag(L).
    const Lt = sparseTranspose(lower);
    const dense = sparseToDense(lower).data;
    const denseT = sparseToDense(Lt).data;
    for (let i = 0; i < dense.length; i++) dense[i] += denseT[i];
    for (let i = 0; i < n; i++) dense[i + i * n] /= 2;
    return denseDataToSparse(dense, n, n);
  },
});

// ── decomposition ─────────────────────────────────────────────────────────
//
// dA = decomposition(A [, type] [, 'upper'|'lower'] [, Name, Value]) stores
// factors of A so that repeated `dA \ b` and `b / dA` reuse them. Factors
// are kept as sparse matrices in underscore-prefixed internal fields:
//   _kind 'lu'         : _L, _U, _p, _q        (P*A*Q = L*U)
//   _kind 'chol'       : _L, _p                (A(p,p) = L*L')
//   _kind 'triangular' : _T, _upper
// Dense inputs are factored with the dense LAPACK path (or its TypeScript
// fallback) and the factors are stored sparse, so the same solve code
// serves both.

const DEC_CLASS = "decomposition";

function isDecomposition(v: RuntimeValue): v is RuntimeClassInstance {
  return isRuntimeClassInstance(v) && v.className === DEC_CLASS;
}

function permFromField(d: RuntimeClassInstance, name: string): Int32Array {
  const t = d.fields.get(name);
  if (t === undefined || !isRuntimeTensor(t))
    throw new RuntimeError("decomposition: corrupt");
  const p = new Int32Array(t.data.length);
  for (let i = 0; i < p.length; i++) p[i] = t.data[i] - 1;
  return p;
}

function sparseField(d: RuntimeClassInstance, name: string): Sp {
  const v = d.fields.get(name);
  if (v === undefined || !isRuntimeSparseMatrix(v))
    throw new RuntimeError("decomposition: corrupt");
  return v;
}

function charField(d: RuntimeClassInstance, name: string): string {
  const v = d.fields.get(name);
  if (v === undefined) throw new RuntimeError("decomposition: corrupt");
  return parseStringArgLower(v);
}

/** min |d_ii| / max |d_ii| over the diagonal of a square sparse factor. */
function diagRatio(S: Sp): number {
  let lo = Infinity;
  let hi = 0;
  for (let j = 0; j < S.n; j++) {
    let d = 0;
    for (let k = S.jc[j]; k < S.jc[j + 1]; k++)
      if (S.ir[k] === j) {
        d = Math.abs(S.pr[k]);
        break;
      }
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  if (S.n === 0) return 1;
  return hi === 0 ? 0 : lo / hi;
}

/** Dense real LU via the LAPACK bridge, returned in the sparse LuResult form. */
function denseLuFactors(data: Float64Array, n: number): LuResult {
  const { LU, ipiv } = getLapackBridge().lu(toF64(data), n, n);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  for (let i = 0; i < ipiv.length; i++) {
    const j = ipiv[i] - 1;
    if (j !== i) {
      const t = perm[i];
      perm[i] = perm[j];
      perm[j] = t;
    }
  }
  const L = allocFloat64Array(n * n);
  const U = allocFloat64Array(n * n);
  let singular = false;
  for (let j = 0; j < n; j++) {
    L[j + j * n] = 1;
    for (let i = j + 1; i < n; i++) L[i + j * n] = LU[i + j * n];
    for (let i = 0; i <= j; i++) U[i + j * n] = LU[i + j * n];
    if (U[j + j * n] === 0) singular = true;
  }
  return {
    L: denseDataToSparse(L, n, n),
    U: denseDataToSparse(U, n, n),
    p: perm,
    q: identityPerm(n),
    singular,
  };
}

/** Dense real Cholesky via the LAPACK bridge; lower factor sparse, or null. */
function denseCholLower(data: Float64Array, n: number): Sp | null {
  const { R, info } = getLapackBridge().chol(toF64(data), n, false);
  if (info > 0) return null;
  return denseDataToSparse(R, n, n);
}

function denseIsTriangular(
  data: Float64Array,
  n: number,
  upper: boolean
): boolean {
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      if (upper ? i <= j : i >= j) continue;
      if (data[i + j * n] !== 0) return false;
    }
  return true;
}

builtin({
  name: "decomposition",
  help: {
    signatures: [
      "dA = decomposition(A)",
      "dA = decomposition(A, type)",
      "dA = decomposition(A, 'triangular', 'upper' | 'lower')",
      "x = dA \\ b",
      "x = b / dA",
    ],
    description:
      "Factor a square matrix once and reuse the factors: dA \\ b and b / dA solve with the stored factorization. type is 'auto' (default), 'chol', 'lu', or 'triangular'; other MATLAB types fall back to LU. Sparse matrices use the sparse Cholesky / LU with a fill-reducing ordering. Properties: MatrixSize, Type, Datatype, IsReal, IsConjugateTransposed, ScaleFactor, CheckCondition.",
  },
  mfile: true,
  apply: args => {
    if (args.length === 0)
      throw new RuntimeError(
        "decomposition: the empty form decomposition() is not supported"
      );
    return buildDecomposition(args);
  },
});

function buildDecomposition(args: RuntimeValue[]): RuntimeValue {
  const A = args[0];
  let type = "auto";
  let triangle: "upper" | "lower" | null = null;
  let checkCondition = true;
  let luTol = 1.0;
  let i = 1;
  const isStr = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);
  while (i < args.length) {
    const a = args[i];
    if (!isStr(a))
      throw new RuntimeError("decomposition: options must be strings");
    const s = parseStringArgLower(a);
    if (s === "upper" || s === "lower") {
      triangle = s;
      i++;
      continue;
    }
    if (
      s === "checkcondition" ||
      s === "ranktolerance" ||
      s === "banddensity" ||
      s === "ldlpivottolerance" ||
      s === "lupivottolerance"
    ) {
      if (i + 1 >= args.length)
        throw new RuntimeError(`decomposition: missing value for '${s}'`);
      const v = args[i + 1];
      if (s === "checkcondition") checkCondition = realScalarArg(v) !== 0;
      if (s === "lupivottolerance") luTol = realScalarArg(v);
      i += 2;
      continue;
    }
    if (i === 1) {
      type = s;
      i++;
      continue;
    }
    throw new RuntimeError(`decomposition: unknown option '${s}'`);
  }
  const known = [
    "auto",
    "chol",
    "lu",
    "triangular",
    "ldl",
    "qr",
    "banded",
    "diagonal",
    "permutedtriangular",
    "hessenberg",
    "cod",
  ];
  if (!known.includes(type))
    throw new RuntimeError(
      `decomposition: unknown decomposition type '${type}'`
    );

  // Normalize the input: real square, either sparse or dense column-major.
  let S: Sp | null = null;
  let dense: Float64Array | null = null;
  let n: number;
  if (isRuntimeSparseMatrix(A)) {
    if (A.pi)
      throw new RuntimeError(
        "decomposition: complex matrices are not supported"
      );
    if (A.m !== A.n)
      throw new RuntimeError(
        "decomposition: only square matrices are supported"
      );
    S = A;
    n = A.n;
  } else if (isRuntimeTensor(A) || isRuntimeNumber(A) || isRuntimeLogical(A)) {
    const t = isRuntimeTensor(A)
      ? A
      : RTV.tensor(allocFloat64Array([Number(A)]), [1, 1]);
    if (t.imag)
      throw new RuntimeError(
        "decomposition: complex matrices are not supported"
      );
    const [m, nn] = tensorSize2D(t);
    if (m !== nn)
      throw new RuntimeError(
        "decomposition: only square matrices are supported"
      );
    n = nn;
    dense = t.data;
  } else {
    throw new RuntimeError("decomposition: A must be a numeric matrix");
  }

  const fields = new Map<string, RuntimeValue>();
  // `rc` is a cheap reciprocal-condition estimate from the factor diagonal
  // (min |d| / max |d|); it drives isIllConditioned and the solve warning.
  const setCommon = (typeName: string, rc: number) => {
    fields.set("MatrixSize", RTV.tensor(allocFloat64Array([n, n]), [1, 2]));
    fields.set("Type", RTV.char(typeName));
    fields.set("Datatype", RTV.char("double"));
    fields.set("IsReal", RTV.logical(true));
    fields.set("IsConjugateTransposed", RTV.logical(false));
    fields.set("ScaleFactor", RTV.num(1));
    fields.set("CheckCondition", RTV.logical(checkCondition));
    fields.set("_rcond", RTV.num(rc));
  };

  const storeTriangular = (T: Sp, upper: boolean) => {
    fields.set("_kind", RTV.char("triangular"));
    fields.set("_T", T);
    fields.set("_upper", RTV.logical(upper));
    setCommon("triangular", diagRatio(T));
  };
  const storeChol = (L: Sp, p: Int32Array) => {
    fields.set("_kind", RTV.char("chol"));
    fields.set("_L", L);
    fields.set("_Lt", sparseTranspose(L));
    fields.set("_p", permVectorTensor(p));
    setCommon("chol", diagRatio(L) ** 2);
  };
  const storeLu = (lu: LuResult) => {
    fields.set("_kind", RTV.char("lu"));
    fields.set("_L", lu.L);
    fields.set("_U", lu.U);
    fields.set("_p", permVectorTensor(lu.p));
    fields.set("_q", permVectorTensor(lu.q));
    setCommon(type === "auto" || type === "lu" ? "lu" : type, diagRatio(lu.U));
  };

  // Triangular detection / request.
  const upperOk = S ? sparseIsUpper(S) : denseIsTriangular(dense!, n, true);
  const lowerOk = S ? sparseIsLower(S) : denseIsTriangular(dense!, n, false);
  if (type === "triangular" || (type === "auto" && (upperOk || lowerOk))) {
    let upper: boolean;
    if (triangle) upper = triangle === "upper";
    else if (upperOk) upper = true;
    else if (lowerOk) upper = false;
    else throw new RuntimeError("decomposition: matrix is not triangular");
    const T = S ?? denseDataToSparse(dense!, n, n);
    storeTriangular(T, upper);
    return new RuntimeClassInstance(DEC_CLASS, fields, false);
  }

  // Cholesky: requested, or auto with a symmetric sparse matrix (MATLAB's
  // auto type for dense symmetric input is 'lu').
  const symmetric = S !== null && sparseIsSymmetric(S);
  if (type === "chol" || (type === "auto" && symmetric)) {
    if (S) {
      const perm = minimumDegreeOrdering(S);
      const res = sparseChol(sparsePermute(S, perm, perm));
      if (res.ok) {
        storeChol(res.L, perm);
        return new RuntimeClassInstance(DEC_CLASS, fields, false);
      }
    } else {
      const L = denseCholLower(dense!, n);
      if (L) {
        storeChol(L, identityPerm(n));
        return new RuntimeClassInstance(DEC_CLASS, fields, false);
      }
    }
    if (type === "chol")
      throw new RuntimeError(
        "Cholesky decomposition requires the input matrix to be positive definite."
      );
  }

  // Everything else: LU with partial pivoting.
  const lu = S
    ? sparseLu(S, columnOrdering(S), luTol)
    : denseLuFactors(dense!, n);
  if (lu.singular) emitWarning(SINGULAR_WARNING);
  storeLu(lu);
  return new RuntimeClassInstance(DEC_CLASS, fields, false);
}

function decompositionRcond(d: RuntimeClassInstance): number {
  const rc = d.fields.get("_rcond");
  return typeof rc === "number" ? rc : 1;
}

builtin({
  name: "isIllConditioned",
  help: {
    signatures: ["tf = isIllConditioned(dA)"],
    description:
      "True if the matrix stored in a decomposition object is ill conditioned (reciprocal condition estimate below eps).",
  },
  nargin: 1,
  apply: args => {
    if (!isDecomposition(args[0]))
      throw new RuntimeError(
        "isIllConditioned: argument must be a decomposition object"
      );
    return RTV.logical(decompositionRcond(args[0]) < Number.EPSILON);
  },
});

/** Solve A * X = B (transposed = false) or A' * X = B with stored factors. */
function decompositionSolve(
  d: RuntimeClassInstance,
  B: Float64Array,
  n: number,
  nrhs: number,
  transposed: boolean
): Float64Array {
  if (d.fields.get("CheckCondition") === true) {
    const rc = decompositionRcond(d);
    if (rc < Number.EPSILON)
      emitWarning(
        "Matrix is close to singular or badly scaled. Results may be " +
          `inaccurate. RCOND =  ${rc.toExponential(6)}.`
      );
  }
  const kind = charField(d, "_kind");
  if (kind === "triangular") {
    let T = sparseField(d, "_T");
    let upper = d.fields.get("_upper") === true;
    if (transposed) {
      T = sparseTranspose(T);
      upper = !upper;
    }
    const r = upper
      ? sparseSolveUpper(T, B, nrhs)
      : sparseSolveLower(T, B, nrhs);
    if (r.singular) emitWarning(SINGULAR_WARNING);
    return r.x;
  }
  if (kind === "chol") {
    // A(p,p) = L L'  =>  x(p) = L' \ (L \ b(p)); symmetric, so A' = A.
    const L = sparseField(d, "_L");
    const p = permFromField(d, "_p");
    const bp = allocFloat64Array(B.length);
    for (let c = 0; c < nrhs; c++)
      for (let k = 0; k < n; k++) bp[c * n + k] = B[c * n + p[k]];
    const y = sparseSolveLower(L, bp, nrhs).x;
    const z = sparseSolveUpper(sparseField(d, "_Lt"), y, nrhs).x;
    const x = allocFloat64Array(B.length);
    for (let c = 0; c < nrhs; c++)
      for (let k = 0; k < n; k++) x[c * n + p[k]] = z[c * n + k];
    return x;
  }
  const lu: LuResult = {
    L: sparseField(d, "_L"),
    U: sparseField(d, "_U"),
    p: permFromField(d, "_p"),
    q: permFromField(d, "_q"),
    singular: false,
  };
  return transposed ? luSolveTransposed(lu, B, nrhs) : luSolve(lu, B, nrhs);
}

/** Real dense column-major data and 2-D size of a right-hand side. */
function rhsData(v: RuntimeValue): {
  data: Float64Array;
  m: number;
  n: number;
} {
  if (isRuntimeSparseMatrix(v)) {
    if (v.pi)
      throw new RuntimeError(
        "decomposition: complex operands are not supported"
      );
    return { data: sparseToDense(v).data, m: v.m, n: v.n };
  }
  if (isRuntimeTensor(v)) {
    if (v.imag)
      throw new RuntimeError(
        "decomposition: complex operands are not supported"
      );
    const [m, n] = tensorSize2D(v);
    return { data: v.data, m, n };
  }
  if (isRuntimeNumber(v) || isRuntimeLogical(v))
    return { data: allocFloat64Array([toNumber(v)]), m: 1, n: 1 };
  throw new RuntimeError("decomposition: operand must be numeric");
}

/**
 * Hook for the mldivide / mrdivide operator builtins: `dA \ B` and `B / dA`
 * with a decomposition object. Returns undefined when neither operand is a
 * decomposition so the ordinary operator runs.
 */
export function tryDecompositionDivide(
  name: "mldivide" | "mrdivide",
  a: RuntimeValue,
  b: RuntimeValue
): RuntimeValue | undefined {
  if (name === "mldivide") {
    if (!isDecomposition(a)) {
      if (isDecomposition(b))
        throw new RuntimeError(
          "decomposition: a decomposition object can only be the left operand of \\"
        );
      return undefined;
    }
    const n = toNumber(permSize(a));
    const rhs = rhsData(b);
    if (rhs.m !== n)
      throw new RuntimeError(
        `mldivide (\\): A is ${n}×${n} but B has ${rhs.m} rows`
      );
    const x = decompositionSolve(a, rhs.data, n, rhs.n, false);
    return RTV.tensor(x, [n, rhs.n]);
  }
  if (!isDecomposition(b)) {
    if (isDecomposition(a))
      throw new RuntimeError(
        "decomposition: a decomposition object can only be the right operand of /"
      );
    return undefined;
  }
  // B / A = (A' \ B')'.
  const n = toNumber(permSize(b));
  const rhs = rhsData(a);
  if (rhs.n !== n)
    throw new RuntimeError(
      `mrdivide (/): A is ${n}×${n} but B has ${rhs.n} columns`
    );
  const nrhs = rhs.m;
  const Bt = allocFloat64Array(n * nrhs);
  for (let c = 0; c < nrhs; c++)
    for (let k = 0; k < n; k++) Bt[c * n + k] = rhs.data[c + k * nrhs];
  const xt = decompositionSolve(b, Bt, n, nrhs, true);
  const x = allocFloat64Array(n * nrhs);
  for (let c = 0; c < nrhs; c++)
    for (let k = 0; k < n; k++) x[c + k * nrhs] = xt[c * n + k];
  return RTV.tensor(x, [nrhs, n]);
}

function permSize(d: RuntimeClassInstance): RuntimeValue {
  const ms = d.fields.get("MatrixSize");
  if (ms === undefined || !isRuntimeTensor(ms))
    throw new RuntimeError("decomposition: corrupt");
  return ms.data[0];
}

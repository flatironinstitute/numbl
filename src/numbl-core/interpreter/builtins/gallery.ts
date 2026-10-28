/**
 * gallery — families of test matrices.
 *
 * Currently implements the 'tridiag' family (sparse tridiagonal matrices),
 * which is the variant exercised by library code such as M-M.E.S.S.
 */

import { isRuntimeString, isRuntimeChar } from "../../runtime/types.js";
import type { RuntimeValue, RuntimeSparseMatrix } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { parseStringArgLower } from "../../helpers/check-helpers.js";
import { builtin } from "./builtin.js";
import { realDenseArg } from "./math-args.js";
import { allocFloat64Array } from "../../runtime/alloc.js";

function toNumArray(v: RuntimeValue, what: string): number[] {
  const err = () =>
    new RuntimeError(`gallery: complex ${what} is not supported`);
  const a = realDenseArg(v, err);
  return typeof a === "number" ? [a] : Array.from(a.data);
}

function scalar(v: RuntimeValue, what: string): number {
  const a = toNumArray(v, what);
  if (a.length !== 1)
    throw new RuntimeError(`gallery: ${what} must be a scalar`);
  return a[0];
}

/**
 * Build the n-by-n sparse tridiagonal matrix with the given subdiagonal
 * (length n-1), diagonal (length n), and superdiagonal (length n-1).
 * Explicit zeros are dropped to match MATLAB's sparse storage.
 */
function buildTridiag(
  sub: number[],
  diag: number[],
  sup: number[]
): RuntimeSparseMatrix {
  const n = diag.length;
  if (sub.length !== n - 1 || sup.length !== n - 1) {
    throw new RuntimeError(
      "gallery: tridiag sub/superdiagonal must have length one less than the diagonal"
    );
  }
  // Build CSC column by column. Within a column the rows c-1 < c < c+1 are
  // already in increasing order, as required by the CSC format.
  const ir: number[] = [];
  const pr: number[] = [];
  const jc = new Int32Array(n + 1);
  for (let c = 0; c < n; c++) {
    jc[c] = ir.length;
    if (c >= 1 && sup[c - 1] !== 0) {
      ir.push(c - 1);
      pr.push(sup[c - 1]);
    }
    if (diag[c] !== 0) {
      ir.push(c);
      pr.push(diag[c]);
    }
    if (c <= n - 2 && sub[c] !== 0) {
      ir.push(c + 1);
      pr.push(sub[c]);
    }
  }
  jc[n] = ir.length;
  return RTV.sparseMatrix(n, n, new Int32Array(ir), jc, allocFloat64Array(pr));
}

/** Toeplitz tridiagonal of order n: scalar sub c, diagonal d, super e. */
function buildTridiagToeplitz(
  n: number,
  c: number,
  d: number,
  e: number
): RuntimeSparseMatrix {
  const off = Math.max(0, n - 1);
  return buildTridiag(
    new Array(off).fill(c),
    new Array(n).fill(d),
    new Array(off).fill(e)
  );
}

function galleryTridiag(rest: RuntimeValue[]): RuntimeValue {
  // gallery('tridiag', n) == gallery('tridiag', n, -1, 2, -1)
  if (rest.length === 1) {
    return buildTridiagToeplitz(Math.round(scalar(rest[0], "n")), -1, 2, -1);
  }
  // gallery('tridiag', n, c, d, e) — n scalar, c/d/e scalars (Toeplitz)
  if (rest.length === 4) {
    return buildTridiagToeplitz(
      Math.round(scalar(rest[0], "n")),
      scalar(rest[1], "c"),
      scalar(rest[2], "d"),
      scalar(rest[3], "e")
    );
  }
  // gallery('tridiag', x, y, z) — x/z vectors (length n-1), y diagonal (length n)
  if (rest.length === 3) {
    const sub = toNumArray(rest[0], "subdiagonal");
    const diag = toNumArray(rest[1], "diagonal");
    const sup = toNumArray(rest[2], "superdiagonal");
    // All-scalar form yields a 1x1 matrix [y] (Toeplitz of order 1).
    if (sub.length === 1 && diag.length === 1 && sup.length === 1) {
      return buildTridiag([], diag, []);
    }
    return buildTridiag(sub, diag, sup);
  }
  throw new RuntimeError("gallery: tridiag expects (n), (n,c,d,e), or (x,y,z)");
}

builtin({
  name: "gallery",
  help: {
    signatures: [
      "A = gallery('tridiag', n)",
      "A = gallery('tridiag', n, c, d, e)",
      "A = gallery('tridiag', x, y, z)",
    ],
    description:
      "Test matrices. gallery('tridiag', ...) returns a sparse tridiagonal " +
      "matrix: gallery('tridiag', n) is the order-n matrix with -1 on the " +
      "sub/superdiagonals and 2 on the diagonal (the negative second-difference " +
      "matrix); gallery('tridiag', n, c, d, e) is the order-n Toeplitz " +
      "tridiagonal with scalar sub c, diagonal d, super e; " +
      "gallery('tridiag', x, y, z) uses vectors x (subdiagonal), y (diagonal), " +
      "z (superdiagonal), where x and z have length one less than y.",
  },
  nargin: [1, Infinity],
  mfile: true,
  apply: args => {
    const name = parseStringArgLower(args[0]);
    // Drop an optional trailing classname argument ('single'/'double').
    let rest = args.slice(1);
    if (rest.length >= 1) {
      const last = rest[rest.length - 1];
      if (isRuntimeString(last) || isRuntimeChar(last)) {
        const cn = parseStringArgLower(last);
        if (cn === "single" || cn === "double") rest = rest.slice(0, -1);
      }
    }
    switch (name) {
      case "tridiag":
        return galleryTridiag(rest);
      default:
        throw new RuntimeError(
          `gallery: matrix family '${name}' is not supported`
        );
    }
  },
});

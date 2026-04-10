/**
 * Core GMRES (Generalized Minimum Residual) algorithm.
 *
 * Callback-based interface so the same code handles both dense matrices
 * (dispatched through the LAPACK bridge) and function-handle arguments
 * (dispatched through the runtime).
 */

export interface GmresResult {
  x: Float64Array;
  flag: number; // 0=converged, 1=max iterations
  relres: number; // norm(M\(b-A*x))/norm(M\b)
  iter: [number, number]; // [outer, inner]
  resvec: Float64Array; // preconditioned residual norms at each inner iteration
}

export type MatvecFn = (x: Float64Array) => Float64Array;
export type PrecSolveFn = (r: Float64Array) => Float64Array;

/**
 * Restarted GMRES with left preconditioning and Givens rotations.
 *
 * @param matvec    Computes A*x
 * @param precSolve Computes M\x (or null for no preconditioning)
 * @param b         Right-hand side (length n)
 * @param n         System dimension
 * @param restart   Inner iteration count (use n for no restart)
 * @param tol       Convergence tolerance on relative preconditioned residual
 * @param maxit     Maximum number of outer iterations
 * @param x0        Initial guess (or null for zeros)
 */
export function gmresCore(
  matvec: MatvecFn,
  precSolve: PrecSolveFn | null,
  b: Float64Array,
  n: number,
  restart: number,
  tol: number,
  maxit: number,
  x0: Float64Array | null
): GmresResult {
  // Initial guess
  const x = x0 ? new Float64Array(x0) : new Float64Array(n);

  if (restart <= 0 || restart > n) restart = n;

  // Compute initial preconditioned residual: r = M\(b - A*x)
  let r = sub(b, matvec(x), n);
  if (precSolve) r = precSolve(r);
  let beta = nrm2(r, n);

  // Preconditioned RHS norm for relative residual
  let normMb: number;
  if (precSolve) {
    normMb = nrm2(precSolve(new Float64Array(b)), n);
  } else {
    normMb = nrm2(b, n);
  }
  if (normMb === 0) normMb = 1;

  const resvecList: number[] = [beta];

  // Already converged?
  if (beta / normMb <= tol) {
    return {
      x,
      flag: 0,
      relres: beta / normMb,
      iter: [0, 0],
      resvec: new Float64Array(resvecList),
    };
  }

  let flag = 1;
  let outerIter = 0;
  let innerIter = 0;

  for (let outer = 1; outer <= maxit; outer++) {
    outerIter = outer;

    // Arnoldi vectors V[n × (restart+1)]
    const V = new Float64Array(n * (restart + 1));
    // V(:,0) = r / beta
    for (let i = 0; i < n; i++) V[i] = r[i] / beta;

    // Upper Hessenberg matrix H[(restart+1) × restart]
    const H = new Float64Array((restart + 1) * restart);
    const cs = new Float64Array(restart);
    const sn = new Float64Array(restart);

    // RHS of the least-squares problem: g = beta * e1
    const g = new Float64Array(restart + 1);
    g[0] = beta;

    let converged = false;

    for (let j = 0; j < restart; j++) {
      innerIter = j + 1;

      // w = M\(A * V(:,j))
      let w = matvec(V.subarray(j * n, (j + 1) * n));
      if (precSolve) w = precSolve(w);

      // Modified Gram-Schmidt
      const ldh = restart + 1;
      for (let i = 0; i <= j; i++) {
        const viOff = i * n;
        let hij = 0;
        for (let k = 0; k < n; k++) hij += w[k] * V[viOff + k];
        H[i + j * ldh] = hij;
        for (let k = 0; k < n; k++) w[k] -= hij * V[viOff + k];
      }

      const wnorm = nrm2(w, n);
      H[j + 1 + j * ldh] = wnorm;

      if (wnorm > 1e-300) {
        const vOff = (j + 1) * n;
        for (let k = 0; k < n; k++) V[vOff + k] = w[k] / wnorm;
      }

      // Apply previous Givens rotations to column j
      for (let i = 0; i < j; i++) {
        const hi = H[i + j * ldh];
        const hi1 = H[i + 1 + j * ldh];
        H[i + j * ldh] = cs[i] * hi + sn[i] * hi1;
        H[i + 1 + j * ldh] = -sn[i] * hi + cs[i] * hi1;
      }

      // Compute new Givens rotation
      const [c, s] = givensRotation(H[j + j * ldh], H[j + 1 + j * ldh]);
      cs[j] = c;
      sn[j] = s;

      // Apply to H and g
      H[j + j * ldh] = c * H[j + j * ldh] + s * H[j + 1 + j * ldh];
      H[j + 1 + j * ldh] = 0;
      const gj = g[j];
      g[j] = c * gj;
      g[j + 1] = -s * gj;

      const residNorm = Math.abs(g[j + 1]);
      resvecList.push(residNorm);

      if (residNorm / normMb <= tol) {
        // Converged — solve H*y = g, update x
        const y = backSolve(H, g, j + 1, ldh);
        for (let k = 0; k < n; k++) {
          for (let l = 0; l <= j; l++) x[k] += V[k + l * n] * y[l];
        }
        flag = 0;
        converged = true;
        break;
      }
    }

    if (converged) break;

    // End of restart cycle — solve and update x0
    const y = backSolve(H, g, restart, restart + 1);
    for (let k = 0; k < n; k++) {
      for (let l = 0; l < restart; l++) x[k] += V[k + l * n] * y[l];
    }

    // Recompute preconditioned residual
    r = sub(b, matvec(x), n);
    if (precSolve) r = precSolve(r);
    beta = nrm2(r, n);

    if (beta / normMb <= tol) {
      flag = 0;
      innerIter = 0;
      break;
    }
  }

  // Final relres (recompute for accuracy if converged)
  let relres: number;
  if (flag === 0) {
    let finalR = sub(b, matvec(x), n);
    if (precSolve) finalR = precSolve(finalR);
    relres = nrm2(finalR, n) / normMb;
  } else {
    relres = beta / normMb;
  }

  return {
    x,
    flag,
    relres,
    iter: [outerIter, innerIter],
    resvec: new Float64Array(resvecList),
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function nrm2(a: Float64Array, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function sub(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[i] = a[i] - b[i];
  return r;
}

function givensRotation(a: number, b: number): [number, number] {
  if (b === 0) return [1, 0];
  if (Math.abs(b) > Math.abs(a)) {
    const t = a / b;
    const s = 1 / Math.sqrt(1 + t * t);
    return [s * t, s];
  }
  const t = b / a;
  const c = 1 / Math.sqrt(1 + t * t);
  return [c, c * t];
}

/** Solve upper triangular H[0:m, 0:m] * y = g[0:m] via back substitution. */
function backSolve(
  H: Float64Array,
  g: Float64Array,
  m: number,
  ldh: number
): Float64Array {
  const y = new Float64Array(m);
  for (let i = 0; i < m; i++) y[i] = g[i];
  for (let i = m - 1; i >= 0; i--) {
    for (let j = i + 1; j < m; j++) y[i] -= H[i + j * ldh] * y[j];
    y[i] /= H[i + i * ldh];
  }
  return y;
}

// ══════════════════════════════════════════════════════════════════════════════
// Complex GMRES
// ══════════════════════════════════════════════════════════════════════════════

export interface ComplexVec {
  re: Float64Array;
  im: Float64Array;
}

export interface GmresComplexResult {
  x: ComplexVec;
  flag: number;
  relres: number;
  iter: [number, number];
  resvec: Float64Array;
}

export type ComplexMatvecFn = (x: ComplexVec) => ComplexVec;
export type ComplexPrecSolveFn = (r: ComplexVec) => ComplexVec;

/** Restarted GMRES for complex systems. */
export function gmresCoreComplex(
  matvec: ComplexMatvecFn,
  precSolve: ComplexPrecSolveFn | null,
  b: ComplexVec,
  n: number,
  restart: number,
  tol: number,
  maxit: number,
  x0: ComplexVec | null
): GmresComplexResult {
  const xRe = x0 ? new Float64Array(x0.re) : new Float64Array(n);
  const xIm = x0 ? new Float64Array(x0.im) : new Float64Array(n);

  if (restart <= 0 || restart > n) restart = n;

  // r = M\(b - A*x)
  const ax = matvec({ re: xRe, im: xIm });
  let rRe = new Float64Array(n);
  let rIm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    rRe[i] = b.re[i] - ax.re[i];
    rIm[i] = b.im[i] - ax.im[i];
  }
  if (precSolve) {
    const pr = precSolve({ re: rRe, im: rIm });
    rRe = new Float64Array(pr.re);
    rIm = new Float64Array(pr.im);
  }
  let beta = cnrm2(rRe, rIm, n);

  // Preconditioned RHS norm
  let normMb: number;
  if (precSolve) {
    const mb = precSolve({
      re: new Float64Array(b.re),
      im: new Float64Array(b.im),
    });
    normMb = cnrm2(mb.re, mb.im, n);
  } else {
    normMb = cnrm2(b.re, b.im, n);
  }
  if (normMb === 0) normMb = 1;

  const resvecList: number[] = [beta];

  if (beta / normMb <= tol) {
    return {
      x: { re: xRe, im: xIm },
      flag: 0,
      relres: beta / normMb,
      iter: [0, 0],
      resvec: new Float64Array(resvecList),
    };
  }

  let flag = 1;
  let outerIter = 0;
  let innerIter = 0;

  for (let outer = 1; outer <= maxit; outer++) {
    outerIter = outer;

    // Arnoldi vectors (split re/im)
    const VRe = new Float64Array(n * (restart + 1));
    const VIm = new Float64Array(n * (restart + 1));
    for (let i = 0; i < n; i++) {
      VRe[i] = rRe[i] / beta;
      VIm[i] = rIm[i] / beta;
    }

    // Complex Hessenberg H, Givens (c real, s complex), g complex
    const ldh = restart + 1;
    const HRe = new Float64Array(ldh * restart);
    const HIm = new Float64Array(ldh * restart);
    const cs = new Float64Array(restart); // c is real
    const snRe = new Float64Array(restart);
    const snIm = new Float64Array(restart);
    const gRe = new Float64Array(restart + 1);
    const gIm = new Float64Array(restart + 1);
    gRe[0] = beta;

    let converged = false;

    for (let j = 0; j < restart; j++) {
      innerIter = j + 1;

      // w = M\(A * V(:,j))
      const vjOff = j * n;
      let wRe: Float64Array, wIm: Float64Array;
      const mv = matvec({
        re: VRe.subarray(vjOff, vjOff + n),
        im: VIm.subarray(vjOff, vjOff + n),
      });
      wRe = mv.re;
      wIm = mv.im;
      if (precSolve) {
        const pw = precSolve({ re: wRe, im: wIm });
        wRe = pw.re;
        wIm = pw.im;
      }
      // Ensure we own the arrays (matvec/precSolve may return views)
      if (wRe.length !== n) wRe = new Float64Array(wRe);
      if (wIm.length !== n) wIm = new Float64Array(wIm);

      // Modified Gram-Schmidt (conjugate dot product)
      for (let i = 0; i <= j; i++) {
        const viOff = i * n;
        let dRe = 0,
          dIm = 0;
        for (let k = 0; k < n; k++) {
          // conj(V) * w
          dRe += VRe[viOff + k] * wRe[k] + VIm[viOff + k] * wIm[k];
          dIm += VRe[viOff + k] * wIm[k] - VIm[viOff + k] * wRe[k];
        }
        HRe[i + j * ldh] = dRe;
        HIm[i + j * ldh] = dIm;
        for (let k = 0; k < n; k++) {
          wRe[k] -= dRe * VRe[viOff + k] - dIm * VIm[viOff + k];
          wIm[k] -= dRe * VIm[viOff + k] + dIm * VRe[viOff + k];
        }
      }

      const wnorm = cnrm2(wRe, wIm, n);
      HRe[j + 1 + j * ldh] = wnorm;
      // HIm[j+1 + j*ldh] = 0  (already zero)

      if (wnorm > 1e-300) {
        const vOff = (j + 1) * n;
        for (let k = 0; k < n; k++) {
          VRe[vOff + k] = wRe[k] / wnorm;
          VIm[vOff + k] = wIm[k] / wnorm;
        }
      }

      // Apply previous Givens rotations to column j of H
      for (let i = 0; i < j; i++) {
        const c = cs[i],
          sR = snRe[i],
          sI = snIm[i];
        const hiR = HRe[i + j * ldh],
          hiI = HIm[i + j * ldh];
        const hi1R = HRe[i + 1 + j * ldh],
          hi1I = HIm[i + 1 + j * ldh];
        // temp = c*H[i] + s*H[i+1]
        HRe[i + j * ldh] = c * hiR + (sR * hi1R - sI * hi1I);
        HIm[i + j * ldh] = c * hiI + (sR * hi1I + sI * hi1R);
        // H[i+1] = -conj(s)*H[i] + c*H[i+1]
        HRe[i + 1 + j * ldh] = -(sR * hiR + sI * hiI) + c * hi1R;
        HIm[i + 1 + j * ldh] = -(-sI * hiR + sR * hiI) + c * hi1I;
      }

      // New Givens rotation: [c s; -conj(s) c] * [H[j]; H[j+1]] = [r; 0]
      const aR = HRe[j + j * ldh],
        aI = HIm[j + j * ldh];
      const bR2 = HRe[j + 1 + j * ldh],
        bI2 = HIm[j + 1 + j * ldh];
      const {
        c,
        sRe: sR,
        sIm: sI,
        rRe: rrR,
        rIm: rrI,
      } = complexGivens(aR, aI, bR2, bI2);
      cs[j] = c;
      snRe[j] = sR;
      snIm[j] = sI;

      HRe[j + j * ldh] = rrR;
      HIm[j + j * ldh] = rrI;
      HRe[j + 1 + j * ldh] = 0;
      HIm[j + 1 + j * ldh] = 0;

      // Apply to g: temp = c*g[j] + s*g[j+1]; g[j+1] = -conj(s)*g[j] + c*g[j+1]
      const gjR = gRe[j],
        gjI = gIm[j];
      const gj1R = gRe[j + 1],
        gj1I = gIm[j + 1];
      gRe[j] = c * gjR + (sR * gj1R - sI * gj1I);
      gIm[j] = c * gjI + (sR * gj1I + sI * gj1R);
      gRe[j + 1] = -(sR * gjR + sI * gjI) + c * gj1R;
      gIm[j + 1] = -(-sI * gjR + sR * gjI) + c * gj1I;

      const residNorm = Math.sqrt(
        gRe[j + 1] * gRe[j + 1] + gIm[j + 1] * gIm[j + 1]
      );
      resvecList.push(residNorm);

      if (residNorm / normMb <= tol) {
        const { yRe, yIm } = complexBackSolve(HRe, HIm, gRe, gIm, j + 1, ldh);
        for (let k = 0; k < n; k++) {
          for (let l = 0; l <= j; l++) {
            xRe[k] += VRe[k + l * n] * yRe[l] - VIm[k + l * n] * yIm[l];
            xIm[k] += VRe[k + l * n] * yIm[l] + VIm[k + l * n] * yRe[l];
          }
        }
        flag = 0;
        converged = true;
        break;
      }
    }

    if (converged) break;

    // Restart: solve and update
    const { yRe, yIm } = complexBackSolve(HRe, HIm, gRe, gIm, restart, ldh);
    for (let k = 0; k < n; k++) {
      for (let l = 0; l < restart; l++) {
        xRe[k] += VRe[k + l * n] * yRe[l] - VIm[k + l * n] * yIm[l];
        xIm[k] += VRe[k + l * n] * yIm[l] + VIm[k + l * n] * yRe[l];
      }
    }

    // Recompute residual
    const ax2 = matvec({ re: xRe, im: xIm });
    rRe = new Float64Array(n);
    rIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      rRe[i] = b.re[i] - ax2.re[i];
      rIm[i] = b.im[i] - ax2.im[i];
    }
    if (precSolve) {
      const pr = precSolve({ re: rRe, im: rIm });
      rRe = new Float64Array(pr.re);
      rIm = new Float64Array(pr.im);
    }
    beta = cnrm2(rRe, rIm, n);

    if (beta / normMb <= tol) {
      flag = 0;
      innerIter = 0;
      break;
    }
  }

  let relres: number;
  if (flag === 0) {
    const ax3 = matvec({ re: xRe, im: xIm });
    let fRe = new Float64Array(n),
      fIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      fRe[i] = b.re[i] - ax3.re[i];
      fIm[i] = b.im[i] - ax3.im[i];
    }
    if (precSolve) {
      const pr = precSolve({ re: fRe, im: fIm });
      fRe = new Float64Array(pr.re);
      fIm = new Float64Array(pr.im);
    }
    relres = cnrm2(fRe, fIm, n) / normMb;
  } else {
    relres = beta / normMb;
  }

  return {
    x: { re: xRe, im: xIm },
    flag,
    relres,
    iter: [outerIter, innerIter],
    resvec: new Float64Array(resvecList),
  };
}

// ── Complex helpers ─────────────────────────────────────────────────────────

function cnrm2(re: Float64Array, im: Float64Array, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += re[i] * re[i] + im[i] * im[i];
  return Math.sqrt(s);
}

/** Complex Givens rotation: [c s; -conj(s) c] * [a; b] = [r; 0], c real >= 0. */
function complexGivens(
  aRe: number,
  aIm: number,
  bRe: number,
  bIm: number
): { c: number; sRe: number; sIm: number; rRe: number; rIm: number } {
  const absB = Math.sqrt(bRe * bRe + bIm * bIm);
  if (absB === 0) return { c: 1, sRe: 0, sIm: 0, rRe: aRe, rIm: aIm };
  const absA = Math.sqrt(aRe * aRe + aIm * aIm);
  if (absA === 0)
    return { c: 0, sRe: bRe / absB, sIm: -bIm / absB, rRe: absB, rIm: 0 };
  const norm = Math.sqrt(absA * absA + absB * absB);
  const c = absA / norm;
  // alpha = a / |a|
  const alpRe = aRe / absA,
    alpIm = aIm / absA;
  // s = alpha * conj(b) / norm
  const sRe = (alpRe * bRe + alpIm * bIm) / norm;
  const sIm = (alpIm * bRe - alpRe * bIm) / norm;
  return { c, sRe, sIm, rRe: alpRe * norm, rIm: alpIm * norm };
}

/** Solve complex upper triangular system H * y = g. */
function complexBackSolve(
  HRe: Float64Array,
  HIm: Float64Array,
  gRe: Float64Array,
  gIm: Float64Array,
  m: number,
  ldh: number
): { yRe: Float64Array; yIm: Float64Array } {
  const yRe = new Float64Array(m);
  const yIm = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    yRe[i] = gRe[i];
    yIm[i] = gIm[i];
  }
  for (let i = m - 1; i >= 0; i--) {
    for (let j = i + 1; j < m; j++) {
      const hR = HRe[i + j * ldh],
        hI = HIm[i + j * ldh];
      yRe[i] -= hR * yRe[j] - hI * yIm[j];
      yIm[i] -= hR * yIm[j] + hI * yRe[j];
    }
    const dR = HRe[i + i * ldh],
      dI = HIm[i + i * ldh];
    const dAbs2 = dR * dR + dI * dI;
    const tmpR = yRe[i],
      tmpI = yIm[i];
    yRe[i] = (tmpR * dR + tmpI * dI) / dAbs2;
    yIm[i] = (tmpI * dR - tmpR * dI) / dAbs2;
  }
  return { yRe, yIm };
}

/** Complex LU solve in-place (split re/im). LU and ipiv from zgetrf-style factorization. */
export function complexLuSolveInPlace(
  n: number,
  LURe: Float64Array,
  LUIm: Float64Array,
  ipiv: Int32Array,
  rhsRe: Float64Array,
  rhsIm: Float64Array
): void {
  // Row permutations
  for (let i = 0; i < n; i++) {
    const pi = ipiv[i] - 1;
    if (pi !== i) {
      let tmp = rhsRe[i];
      rhsRe[i] = rhsRe[pi];
      rhsRe[pi] = tmp;
      tmp = rhsIm[i];
      rhsIm[i] = rhsIm[pi];
      rhsIm[pi] = tmp;
    }
  }
  // Forward substitution (L * y = Pb, unit lower triangular)
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const lR = LURe[i + j * n],
        lI = LUIm[i + j * n];
      rhsRe[i] -= lR * rhsRe[j] - lI * rhsIm[j];
      rhsIm[i] -= lR * rhsIm[j] + lI * rhsRe[j];
    }
  }
  // Back substitution (U * x = y)
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      const uR = LURe[i + j * n],
        uI = LUIm[i + j * n];
      rhsRe[i] -= uR * rhsRe[j] - uI * rhsIm[j];
      rhsIm[i] -= uR * rhsIm[j] + uI * rhsRe[j];
    }
    const dR = LURe[i + i * n],
      dI = LUIm[i + i * n];
    const dAbs2 = dR * dR + dI * dI;
    const tmpR = rhsRe[i],
      tmpI = rhsIm[i];
    rhsRe[i] = (tmpR * dR + tmpI * dI) / dAbs2;
    rhsIm[i] = (tmpI * dR - tmpR * dI) / dAbs2;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Shared utilities
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Apply row permutation + forward/back substitution to solve a pre-factored
 * LU system in-place.  LU and ipiv come from dgetrf (1-based pivots).
 */
export function luSolveInPlace(
  n: number,
  LU: Float64Array,
  ipiv: Int32Array,
  rhs: Float64Array
): void {
  // Row permutations
  for (let i = 0; i < n; i++) {
    const pi = ipiv[i] - 1;
    if (pi !== i) {
      const tmp = rhs[i];
      rhs[i] = rhs[pi];
      rhs[pi] = tmp;
    }
  }
  // Forward substitution (L * y = Pb, unit lower triangular)
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) rhs[i] -= LU[i + j * n] * rhs[j];
  }
  // Back substitution (U * x = y)
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) rhs[i] -= LU[i + j * n] * rhs[j];
    rhs[i] /= LU[i + i * n];
  }
}

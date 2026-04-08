/**
 * Adaptive Gauss-Kronrod (7-15) quadrature — a minimal MATLAB-compatible
 * quadgk implementation sufficient for smooth integrands on a finite real
 * interval.
 *
 * Not supported (yet):
 *  - complex / contour integration
 *  - Waypoints
 *  - infinite limits
 *
 * The integrand `fun` must accept a vector of quadrature nodes and return
 * a vector of the same length (MATLAB's quadgk contract).
 */

// Gauss-Kronrod 7-15 nodes on [-1, 1], ordered ascending.
// Odd indices (1,3,5,7,9,11,13) are the 7 Gauss-Legendre nodes,
// even indices are the additional Kronrod nodes.
const XK: readonly number[] = [
  -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
  -0.7415311855993945, -0.586087235467691, -0.4058451513773972,
  -0.2077849550078985, 0.0, 0.2077849550078985, 0.4058451513773972,
  0.586087235467691, 0.7415311855993945, 0.8648644233597691, 0.9491079123427585,
  0.9914553711208126,
];

// Kronrod weights for the 15-node rule (aligned with XK).
const WK: readonly number[] = [
  0.0229353220105292, 0.0630920926299786, 0.1047900103222502,
  0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
  0.2044329400752989, 0.2094821410847278, 0.2044329400752989,
  0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
  0.1047900103222502, 0.0630920926299786, 0.0229353220105292,
];

// Gauss weights for the 7-node rule, aligned with the Gauss nodes at
// positions 1, 3, 5, 7, 9, 11, 13 in XK.
const WG: readonly number[] = [
  0.1294849661688697, 0.2797053914892767, 0.3818300505051189,
  0.4179591836734694, 0.3818300505051189, 0.2797053914892767,
  0.1294849661688697,
];

/** Return the 15 Kronrod nodes of [lo, hi]. */
export function kronrodNodes(lo: number, hi: number): number[] {
  const m = (lo + hi) / 2;
  const h = (hi - lo) / 2;
  const out = new Array<number>(15);
  for (let i = 0; i < 15; i++) out[i] = m + h * XK[i];
  return out;
}

/** G7-K15 estimate and error for a segment given 15 integrand samples. */
function segmentEstimate(
  lo: number,
  hi: number,
  fv: ArrayLike<number>
): { K: number; err: number } {
  const h = (hi - lo) / 2;
  let K = 0;
  let G = 0;
  for (let i = 0; i < 15; i++) K += WK[i] * fv[i];
  for (let i = 0; i < 7; i++) G += WG[i] * fv[2 * i + 1];
  K *= h;
  G *= h;
  // MATLAB's quadgk uses |K - G|^1.5 * something, but |K-G| is the
  // standard GK error estimate and works well for smooth integrands.
  return { K, err: Math.abs(K - G) };
}

export interface QuadgkOptions {
  relTol?: number;
  absTol?: number;
  maxIntervalCount?: number;
}

export interface QuadgkResult {
  value: number;
  errbnd: number;
  intervalsUsed: number;
}

/**
 * Adaptive Gauss-Kronrod 7-15 quadrature of `integrand` over `[a, b]`.
 *
 * `integrand(pts)` receives a 15-element array of nodes and must return a
 * 15-element array of function values.
 */
export function quadgkAdaptive(
  integrand: (pts: number[]) => number[],
  a: number,
  b: number,
  opts: QuadgkOptions = {}
): QuadgkResult {
  const relTol = opts.relTol ?? 1e-6;
  const absTol = opts.absTol ?? 1e-10;
  const maxIntervals = opts.maxIntervalCount ?? 650;

  if (a === b) return { value: 0, errbnd: 0, intervalsUsed: 0 };
  const sign = a < b ? 1 : -1;
  const lo0 = Math.min(a, b);
  const hi0 = Math.max(a, b);

  const segmentOn = (lo: number, hi: number): { K: number; err: number } => {
    const pts = kronrodNodes(lo, hi);
    const fv = integrand(pts);
    if (fv.length !== 15) {
      throw new Error(
        `quadgk: integrand must return 15 values for 15 nodes, got ${fv.length}`
      );
    }
    return segmentEstimate(lo, hi, fv);
  };

  type Segment = { lo: number; hi: number; K: number; err: number };
  const initial = segmentOn(lo0, hi0);
  let totalK = initial.K;
  let totalErr = initial.err;

  // Priority list: keep sorted so the worst segment is at the end for
  // cheap pop().  N is small in practice for smooth integrands.
  const worklist: Segment[] = [{ lo: lo0, hi: hi0, ...initial }];

  const converged = (): boolean =>
    totalErr <= Math.max(absTol, relTol * Math.abs(totalK));

  let iters = 0;
  while (!converged() && worklist.length < maxIntervals && iters < 10000) {
    iters++;
    // Find segment with largest error.  Linear scan is fine — list stays
    // small because subdivision targets the worst segment.
    let worstIdx = 0;
    for (let i = 1; i < worklist.length; i++) {
      if (worklist[i].err > worklist[worstIdx].err) worstIdx = i;
    }
    const worst = worklist[worstIdx];
    if (worst.hi - worst.lo <= 1e-15 * (hi0 - lo0)) break; // cannot subdivide further

    const mid = (worst.lo + worst.hi) / 2;
    const s1 = segmentOn(worst.lo, mid);
    const s2 = segmentOn(mid, worst.hi);
    totalK += s1.K + s2.K - worst.K;
    totalErr += s1.err + s2.err - worst.err;

    worklist[worstIdx] = { lo: worst.lo, hi: mid, ...s1 };
    worklist.push({ lo: mid, hi: worst.hi, ...s2 });
  }

  return {
    value: sign * totalK,
    errbnd: totalErr,
    intervalsUsed: worklist.length,
  };
}

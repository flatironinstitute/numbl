/**
 * DOP853 �� Explicit Runge-Kutta method of order 8.
 *
 * Algorithm and coefficients follow the scipy DOP853 implementation.
 * See: https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/rk.py
 * See: https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/dop853_coefficients.py
 *
 * Pure numerical code -- no runtime types.
 */

/* eslint-disable no-loss-of-precision */

import {
  evalStepAt,
  type OdeOptions,
  type OdeResult,
  type StepData,
} from "./ode-rk.js";

// ── Step-size control constants ─────────────────────���───────────────

const SAFETY = 0.9;
const MIN_FACTOR = 0.2;
const MAX_FACTOR = 10;

// ── DOP853 coefficients ─────────────────���───────────────────────────

const N_STAGES = 12;
const N_STAGES_EXTENDED = 16;
const INTERPOLATOR_POWER = 7;

/** Nodes c_i (length 16). */
const C: number[] = [
  0.0, 0.526001519587677318785587544488e-1, 0.789002279381515978178381316732e-1,
  0.11835034190722739672675719751, 0.28164965809277260327324280249,
  0.333333333333333333333333333333, 0.25, 0.307692307692307692307692307692,
  0.651282051282051282051282051282, 0.6, 0.857142857142857142857142857142, 1.0,
  1.0, 0.1, 0.2, 0.777777777777777777777777777778,
];

/**
 * RK coefficients A_{s,j} stored as sparse rows.
 * A[s] = [[col, val], ...] for non-zero entries.
 */
const A_SPARSE: [number, number][][] = [
  [], // row 0
  [[0, 5.26001519587677318785587544488e-2]], // row 1
  [
    [0, 1.97250569845378994544595329183e-2],
    [1, 5.91751709536136983633785987549e-2],
  ],
  [
    [0, 2.95875854768068491816892993775e-2],
    [2, 8.87627564304205475450678981324e-2],
  ],
  [
    [0, 2.41365134159266685502369798665e-1],
    [2, -8.84549479328286085344864962717e-1],
    [3, 9.24834003261792003115737966543e-1],
  ],
  [
    [0, 3.7037037037037037037037037037e-2],
    [3, 1.70828608729473871279604482173e-1],
    [4, 1.25467687566822425016691814123e-1],
  ],
  [
    [0, 3.7109375e-2],
    [3, 1.70252211019544039314978060272e-1],
    [4, 6.02165389804559606850219397283e-2],
    [5, -1.7578125e-2],
  ],
  [
    [0, 3.70920001185047927108779319836e-2],
    [3, 1.70383925712239993810214054705e-1],
    [4, 1.07262030446373284651809199168e-1],
    [5, -1.53194377486244017527936158236e-2],
    [6, 8.27378916381402288758473766002e-3],
  ],
  [
    [0, 6.24110958716075717114429577812e-1],
    [3, -3.36089262944694129406857109825],
    [4, -8.68219346841726006818189891453e-1],
    [5, 2.75920996994467083049415600797e1],
    [6, 2.01540675504778934086186788979e1],
    [7, -4.34898841810699588477366255144e1],
  ],
  [
    [0, 4.77662536438264365890433908527e-1],
    [3, -2.48811461997166764192642586468],
    [4, -5.90290826836842996371446475743e-1],
    [5, 2.12300514481811942347288949897e1],
    [6, 1.52792336328824235832596922938e1],
    [7, -3.32882109689848629194453265587e1],
    [8, -2.03312017085086261358222928593e-2],
  ],
  [
    [0, -9.3714243008598732571704021658e-1],
    [3, 5.18637242884406370830023853209],
    [4, 1.09143734899672957818500254654],
    [5, -8.14978701074692612513997267357],
    [6, -1.85200656599969598641566180701e1],
    [7, 2.27394870993505042818970056734e1],
    [8, 2.49360555267965238987089396762],
    [9, -3.0467644718982195003823669022],
  ],
  [
    [0, 2.27331014751653820792359768449],
    [3, -1.05344954667372501984066689879e1],
    [4, -2.00087205822486249909675718444],
    [5, -1.79589318631187989172765950534e1],
    [6, 2.79488845294199600508499808837e1],
    [7, -2.85899827713502369474065508674],
    [8, -8.87285693353062954433549289258],
    [9, 1.23605671757943030647266201528e1],
    [10, 6.43392746015763530355970484046e-1],
  ],
  // row 12 = B weights
  [
    [0, 5.42937341165687622380535766363e-2],
    [5, 4.45031289275240888144113950566],
    [6, 1.89151789931450038304281599044],
    [7, -5.8012039600105847814672114227],
    [8, 3.1116436695781989440891606237e-1],
    [9, -1.52160949662516078556178806805e-1],
    [10, 2.01365400804030348374776537501e-1],
    [11, 4.47106157277725905176885569043e-2],
  ],
  // row 13 (extra for dense output)
  [
    [0, 5.61675022830479523392909219681e-2],
    [6, 2.53500210216624811088794765333e-1],
    [7, -2.46239037470802489917441475441e-1],
    [8, -1.24191423263816360469010140626e-1],
    [9, 1.5329179827876569731206322685e-1],
    [10, 8.20105229563468988491666602057e-3],
    [11, 7.56789766054569976138603589584e-3],
    [12, -8.298e-3],
  ],
  // row 14
  [
    [0, 3.18346481635021405060768473261e-2],
    [5, 2.83009096723667755288322961402e-2],
    [6, 5.35419883074385676223797384372e-2],
    [7, -5.49237485713909884646569340306e-2],
    [10, -1.08347328697249322858509316994e-4],
    [11, 3.82571090835658412954920192323e-4],
    [12, -3.40465008687404560802977114492e-4],
    [13, 1.41312443674632500278074618366e-1],
  ],
  // row 15
  [
    [0, -4.28896301583791923408573538692e-1],
    [5, -4.69762141536116384314449447206],
    [6, 7.68342119606259904184240953878],
    [7, 4.06898981839711007970213554331],
    [8, 3.56727187455281109270669543021e-1],
    [12, -1.39902416515901462129418009734e-3],
    [13, 2.9475147891527723389556272149],
    [14, -9.15095847217987001081870187138],
  ],
];

/** B weights = A[12, :12]. Extracted for clarity. */
const B: number[] = (() => {
  const b = new Array<number>(N_STAGES).fill(0);
  for (const [col, val] of A_SPARSE[12]) b[col] = val;
  return b;
})();

/** Error estimator E3 (length N_STAGES + 1 = 13). */
const E3: number[] = (() => {
  const e = B.slice();
  e.push(0); // K[12] entry
  e[0] -= 0.244094488188976377952755905512;
  e[8] -= 0.733846688281611857341361741547;
  e[11] -= 0.220588235294117647058823529412e-1;
  return e;
})();

/** Error estimator E5 (length N_STAGES + 1 = 13). */
const E5: number[] = (() => {
  const e = new Array<number>(N_STAGES + 1).fill(0);
  e[0] = 0.1312004499419488073250102996e-1;
  e[5] = -0.1225156446376204440720569753e1;
  e[6] = -0.4957589496572501915214079952;
  e[7] = 0.1664377182454986536961530415e1;
  e[8] = -0.350328848749973681688648729;
  e[9] = 0.3341791187130174790297318841;
  e[10] = 0.8192320648511571246570742613e-1;
  e[11] = -0.2235530786388629525884427845e-1;
  return e;
})();

/** Dense output D matrix (4 x 16). D[r][c] for rows 0..3. */
const D: number[][] = [
  (() => {
    const r = new Array<number>(N_STAGES_EXTENDED).fill(0);
    r[0] = -0.84289382761090128651353491142e1;
    r[5] = 0.5667149535193777696253178359;
    r[6] = -0.30689499459498916912797304727e1;
    r[7] = 0.2384667656512069828772814968e1;
    r[8] = 0.21170345824450282767155149946e1;
    r[9] = -0.8713915837779729920678990749;
    r[10] = 0.2240437430260788275854177165e1;
    r[11] = 0.6315787787694688181557024929;
    r[12] = -0.889903364513333108206981174e-1;
    r[13] = 0.18148505520854727256656404962e2;
    r[14] = -0.91946323924783554000451984436e1;
    r[15] = -0.44360363875948939664310572e1;
    return r;
  })(),
  (() => {
    const r = new Array<number>(N_STAGES_EXTENDED).fill(0);
    r[0] = 0.10427508642579134603413151009e2;
    r[5] = 0.24228349177525818288430175319e3;
    r[6] = 0.16520045171727028198505394887e3;
    r[7] = -0.37454675472269020279518312152e3;
    r[8] = -0.22113666853125306036270938578e2;
    r[9] = 0.77334326684722638389603898808e1;
    r[10] = -0.30674084731089398182061213626e2;
    r[11] = -0.93321305264302278729567221706e1;
    r[12] = 0.15697238121770843886131091075e2;
    r[13] = -0.31139403219565177677282850411e2;
    r[14] = -0.93529243588444783865713862664e1;
    r[15] = 0.3581684148639408375246589854e2;
    return r;
  })(),
  (() => {
    const r = new Array<number>(N_STAGES_EXTENDED).fill(0);
    r[0] = 0.19985053242002433820987653617e2;
    r[5] = -0.38703730874935176555105901742e3;
    r[6] = -0.18917813819516756882830838328e3;
    r[7] = 0.52780815920542364900561016686e3;
    r[8] = -0.11573902539959630126141871134e2;
    r[9] = 0.68812326946963000169666922661e1;
    r[10] = -0.1000605096691083840318386098e1;
    r[11] = 0.7777137798053443209286926574;
    r[12] = -0.27782057523535084065932004339e1;
    r[13] = -0.60196695231264120758267380846e2;
    r[14] = 0.84320405506677161018159903784e2;
    r[15] = 0.1199229113618278932803513003e2;
    return r;
  })(),
  (() => {
    const r = new Array<number>(N_STAGES_EXTENDED).fill(0);
    r[0] = -0.25693933462703749003312586129e2;
    r[5] = -0.15418974869023643374053993627e3;
    r[6] = -0.23152937917604549567536039109e3;
    r[7] = 0.3576391179106141237828534991e3;
    r[8] = 0.93405324183624310003907691704e2;
    r[9] = -0.37458323136451633156875139351e2;
    r[10] = 0.10409964950896230045147246184e3;
    r[11] = 0.29840293426660503123344363579e2;
    r[12] = -0.43533456590011143754432175058e2;
    r[13] = 0.963245539591882829483949506e2;
    r[14] = -0.39177261675615439165231486172e2;
    r[15] = -0.14972683625798562581422125276e3;
    return r;
  })(),
];

// ── Solver ────────────────���─────────────────────────────────────────

/**
 * Solve y' = f(t,y) using the DOP853 method.
 *
 * Step control follows scipy DOP853._step_impl.
 * Error estimation uses dual E3/E5 estimators.
 * Dense output uses extra stages and 7th-order polynomial.
 */
export function solveDOP853(
  f: (t: number, y: number[]) => number[],
  tspan: number[],
  y0: number[],
  opts: Partial<OdeOptions>
): OdeResult {
  const t0 = tspan[0];
  const tf = tspan[tspan.length - 1];
  const direction = tf >= t0 ? 1 : -1;
  const neq = y0.length;
  const nK = N_STAGES + 1; // 13 entries for main step

  const relTol = opts.relTol ?? 1e-3;
  const absTol = opts.absTol ?? 1e-6;
  const maxStep = opts.maxStep ?? Infinity;
  const errorExponent = -1 / (7 + 1); // error_estimator_order = 7

  // K storage: N_STAGES_EXTENDED x neq (16 rows for dense output stages)
  const K: number[][] = new Array(N_STAGES_EXTENDED);
  for (let s = 0; s < N_STAGES_EXTENDED; s++) K[s] = new Array(neq);

  let t = t0;
  let y = y0.slice();

  // Initial derivative
  const f0 = f(t, y);
  copyInto(K[0], f0);

  // Initial step size (same algorithm as RK, with errorOrder = 7)
  let hAbs: number;
  if (opts.initialStep != null && opts.initialStep > 0) {
    hAbs = opts.initialStep;
  } else {
    hAbs = selectInitialStep(f, t0, y0, tf, f0, direction, 7, relTol, absTol);
  }

  // Output arrays
  const tOut: number[] = [t0];
  const yOut: number[][] = [y0.slice()];
  const steps: StepData[] = [];
  const te: number[] = [];
  const ye: number[][] = [];
  const ie: number[] = [];

  // Events
  let prevEventValues: number[] | null = null;
  if (opts.events) {
    prevEventValues = opts.events(t, y)[0];
  }

  let terminated = false;

  while (direction * (tf - t) > 0 && !terminated) {
    const minStep = 10 * Number.EPSILON * Math.abs(t);
    if (hAbs > maxStep) hAbs = maxStep;
    else if (hAbs < minStep) hAbs = minStep;

    let stepAccepted = false;
    let stepRejected = false;

    while (!stepAccepted) {
      if (hAbs < minStep) {
        throw new Error(
          `dop853: step size too small at t = ${t}. The problem may be stiff.`
        );
      }

      let h = hAbs * direction;
      let tNew = t + h;
      if (direction * (tNew - tf) > 0) tNew = tf;
      h = tNew - t;
      hAbs = Math.abs(h);

      // ── Compute 12 stages (rk_step) ─────────────────────────
      // K[0] already set
      for (let s = 1; s < N_STAGES; s++) {
        const ts = t + C[s] * h;
        const ys = new Array<number>(neq);
        for (let i = 0; i < neq; i++) {
          let dy = 0;
          for (const [col, val] of A_SPARSE[s]) {
            dy += val * K[col][i];
          }
          ys[i] = y[i] + dy * h;
        }
        copyInto(K[s], f(ts, ys));
      }

      // y_new = y + h * B @ K[:N_STAGES]
      const yNew = new Array<number>(neq);
      for (let i = 0; i < neq; i++) {
        let sum = 0;
        for (let s = 0; s < N_STAGES; s++) {
          sum += B[s] * K[s][i];
        }
        yNew[i] = y[i] + h * sum;
      }

      // K[12] = f(tNew, yNew)
      copyInto(K[N_STAGES], f(tNew, yNew));

      // ── Error estimation (scipy DOP853._estimate_error_norm) ─
      const scale = new Array<number>(neq);
      for (let i = 0; i < neq; i++) {
        scale[i] =
          absTol + relTol * Math.max(Math.abs(y[i]), Math.abs(yNew[i]));
      }

      let err5NormSq = 0;
      let err3NormSq = 0;
      for (let i = 0; i < neq; i++) {
        let e5 = 0;
        let e3 = 0;
        for (let s = 0; s < nK; s++) {
          e5 += E5[s] * K[s][i];
          e3 += E3[s] * K[s][i];
        }
        e5 /= scale[i];
        e3 /= scale[i];
        err5NormSq += e5 * e5;
        err3NormSq += e3 * e3;
      }

      let errNorm: number;
      if (err5NormSq === 0 && err3NormSq === 0) {
        errNorm = 0;
      } else {
        const denom = err5NormSq + 0.01 * err3NormSq;
        errNorm = (Math.abs(h) * err5NormSq) / Math.sqrt(denom * neq);
      }

      // ── Step acceptance ──────────────────────────────────────
      if (errNorm < 1) {
        let factor: number;
        if (errNorm === 0) {
          factor = MAX_FACTOR;
        } else {
          factor = Math.min(
            MAX_FACTOR,
            SAFETY * Math.pow(errNorm, errorExponent)
          );
        }
        if (stepRejected) factor = Math.min(1, factor);
        hAbs *= factor;
        stepAccepted = true;

        // ── Dense output: compute extra stages K[13..15] ─────
        for (let s = N_STAGES + 1; s < N_STAGES_EXTENDED; s++) {
          const ts = t + C[s] * h;
          const ys = new Array<number>(neq);
          for (let i = 0; i < neq; i++) {
            let dy = 0;
            for (const [col, val] of A_SPARSE[s]) {
              dy += val * K[col][i];
            }
            ys[i] = y[i] + dy * h;
          }
          copyInto(K[s], f(ts, ys));
        }

        // Build F[0..6] (scipy: Dop853DenseOutput)
        const fOld = K[0];
        const fNew = K[N_STAGES]; // f(tNew, yNew)
        const F: number[][] = new Array(INTERPOLATOR_POWER);
        for (let p = 0; p < INTERPOLATOR_POWER; p++) {
          F[p] = new Array(neq);
        }

        for (let i = 0; i < neq; i++) {
          const deltaY = yNew[i] - y[i];
          F[0][i] = deltaY;
          F[1][i] = h * fOld[i] - deltaY;
          F[2][i] = 2 * deltaY - h * (fNew[i] + fOld[i]);
        }
        // F[3..6] = h * D @ K
        for (let r = 0; r < 4; r++) {
          const dRow = D[r];
          for (let i = 0; i < neq; i++) {
            let sum = 0;
            for (let s = 0; s < N_STAGES_EXTENDED; s++) {
              sum += dRow[s] * K[s][i];
            }
            F[3 + r][i] = h * sum;
          }
        }

        const stepData: StepData = {
          tOld: t,
          tNew,
          h,
          yOld: y.slice(),
          yNew: yNew.slice(),
          F,
        };
        steps.push(stepData);

        // Event detection (same logic as RK)
        if (opts.events && prevEventValues) {
          const [newVals, isterminal, eventDir] = opts.events(tNew, yNew);
          for (let ei = 0; ei < newVals.length; ei++) {
            const oldV = prevEventValues[ei];
            const newV = newVals[ei];
            const crossed =
              (eventDir[ei] === 0 && oldV * newV < 0) ||
              (eventDir[ei] === 1 && oldV < 0 && newV >= 0) ||
              (eventDir[ei] === -1 && oldV > 0 && newV <= 0);
            if (crossed) {
              const [tE, yE] = bisectEvent(ei, stepData, opts.events);
              te.push(tE);
              ye.push(yE);
              ie.push(ei + 1);
              if (isterminal[ei]) {
                tOut.push(tE);
                yOut.push(yE);
                terminated = true;
                break;
              }
            }
          }
          prevEventValues = newVals;
        }

        if (!terminated) {
          tOut.push(tNew);
          yOut.push(yNew.slice());
        }

        t = tNew;
        y = yNew;
        copyInto(K[0], K[N_STAGES]); // FSAL
      } else {
        hAbs *= Math.max(MIN_FACTOR, SAFETY * Math.pow(errNorm, errorExponent));
        stepRejected = true;
      }
    }
  }

  return { t: tOut, y: yOut, te, ye, ie, steps };
}

// ── Helpers ────────────��────────────────────────────────────────────

function copyInto(dst: number[], src: number[]): void {
  for (let i = 0; i < src.length; i++) dst[i] = src[i];
}

/** Initial step size estimation (same algorithm as RK). */
function selectInitialStep(
  fun: (t: number, y: number[]) => number[],
  t0: number,
  y0: number[],
  tBound: number,
  f0: number[],
  direction: number,
  errorOrder: number,
  rtol: number,
  atol: number
): number {
  const n = y0.length;
  const span = Math.abs(tBound - t0);

  let d0 = 0;
  let d1 = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + rtol * Math.abs(y0[i]);
    d0 += (y0[i] / sc) ** 2;
    d1 += (f0[i] / sc) ** 2;
  }
  d0 = Math.sqrt(d0 / n);
  d1 = Math.sqrt(d1 / n);

  let h0: number;
  if (d0 < 1e-5 || d1 < 1e-5) {
    h0 = 1e-6;
  } else {
    h0 = (0.01 * d0) / d1;
  }

  const y1 = new Array<number>(n);
  for (let i = 0; i < n; i++) y1[i] = y0[i] + h0 * direction * f0[i];
  const f1 = fun(t0 + h0 * direction, y1);

  let d2 = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + rtol * Math.abs(y0[i]);
    d2 += ((f1[i] - f0[i]) / sc) ** 2;
  }
  d2 = Math.sqrt(d2 / n) / h0;

  let h1: number;
  if (Math.max(d1, d2) <= 1e-15) {
    h1 = Math.max(1e-6, h0 * 1e-3);
  } else {
    h1 = Math.pow(0.01 / Math.max(d1, d2), 1 / (errorOrder + 1));
  }

  return Math.min(100 * h0, h1, span);
}

/** Bisect to find event time within a step. */
function bisectEvent(
  eventIdx: number,
  step: StepData,
  events: (t: number, y: number[]) => [number[], boolean[], number[]]
): [number, number[]] {
  const { tOld, h } = step;
  let lo = 0;
  let hi = 1;

  for (let iter = 0; iter < 50; iter++) {
    const mid = (lo + hi) / 2;
    const yMid = evalStepAt(step, mid);
    const tMid = tOld + mid * h;
    const [vals] = events(tMid, yMid);

    const yLo = evalStepAt(step, lo);
    const vLo = events(tOld + lo * h, yLo)[0][eventIdx];

    if (vLo * vals[eventIdx] <= 0) {
      hi = mid;
    } else {
      lo = mid;
    }

    if (Math.abs(hi - lo) * Math.abs(h) < 1e-12) break;
  }

  const xFinal = (lo + hi) / 2;
  return [tOld + xFinal * h, evalStepAt(step, xFinal)];
}

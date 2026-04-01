/**
 * Adaptive explicit Runge-Kutta ODE solvers.
 *
 * Algorithm and coefficients follow the scipy RK45 implementation.
 * See: https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/rk.py
 *
 * Pure numerical code — no runtime types.
 */

// ── Step-size control constants (scipy: module-level) ───────────────

const SAFETY = 0.9;
const MIN_FACTOR = 0.2;
const MAX_FACTOR = 10;

// ── Tableau ─────────────────────────────────────────────────────────

export interface RKTableau {
  name: string;
  order: number;
  errorOrder: number;
  nStages: number;
  /** Nodes c_i, length nStages */
  C: number[];
  /** RK coefficients a_{s,j}, nStages rows, lower-triangular */
  A: number[][];
  /** Higher-order weights, length nStages */
  B: number[];
  /** Error coefficients E = bhat - b, length nStages + 1 */
  E: number[];
  /** Dense output polynomial coefficients, (nStages+1) x nInterp */
  P: number[][];
  /** Number of interpolation polynomial columns */
  nInterp: number;
}

/**
 * Dormand-Prince 5(4) pair — used by ode45.
 *
 * Error controlled assuming 4th-order accuracy; steps taken with
 * 5th-order formula (local extrapolation). Quartic interpolation
 * polynomial from Shampine [2].
 */
export const dormandPrince45: RKTableau = {
  name: "ode45",
  order: 5,
  errorOrder: 4,
  nStages: 6,
  C: [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1],
  A: [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  ],
  B: [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
  E: [
    -71 / 57600,
    0,
    71 / 16695,
    -71 / 1920,
    17253 / 339200,
    -22 / 525,
    1 / 40,
  ],
  P: [
    [
      1,
      -8048581381 / 2820520608,
      8663915743 / 2820520608,
      -12715105075 / 11282082432,
    ],
    [0, 0, 0, 0],
    [
      0,
      131558114200 / 32700410799,
      -68118460800 / 10900136933,
      87487479700 / 32700410799,
    ],
    [
      0,
      -1754552775 / 470086768,
      14199869525 / 1410260304,
      -10690763975 / 1880347072,
    ],
    [
      0,
      127303824393 / 49829197408,
      -318862633887 / 49829197408,
      701980252875 / 199316789632,
    ],
    [
      0,
      -282668133 / 205662961,
      2019193451 / 616988883,
      -1453857185 / 822651844,
    ],
    [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
  ],
  nInterp: 4,
};

/**
 * Bogacki-Shampine 3(2) pair — used by ode23.
 *
 * Error controlled assuming 2nd-order accuracy; steps taken with
 * 3rd-order formula (local extrapolation). Cubic Hermite polynomial
 * for the dense output.
 */
export const bogackiShampine23: RKTableau = {
  name: "ode23",
  order: 3,
  errorOrder: 2,
  nStages: 3,
  C: [0, 1 / 2, 3 / 4],
  A: [[], [1 / 2], [0, 3 / 4]],
  B: [2 / 9, 1 / 3, 4 / 9],
  E: [5 / 72, -1 / 12, -1 / 9, 1 / 8],
  P: [
    [1, -4 / 3, 5 / 9],
    [0, 1, -2 / 3],
    [0, 4 / 3, -8 / 9],
    [0, -1, 1],
  ],
  nInterp: 3,
};

// ── Dense output ────────────────────────────────────────────────────

export interface StepData {
  tOld: number;
  tNew: number;
  h: number;
  yOld: number[];
  yNew: number[];
  /** Q = K^T P, dimensions neq x nInterp (RK45, RK23) */
  Q?: number[][];
  /** F interpolation vectors, nPower x neq (DOP853) */
  F?: number[][];
}

/**
 * Evaluate the dense output polynomial at fractional position x in [0,1].
 *
 * Matches scipy RkDenseOutput._call_impl:
 *   y = y_old + h * Q @ [x, x^2, ..., x^nInterp]
 */
export function denseOutputEval(
  yOld: number[],
  Q: number[][],
  h: number,
  x: number
): number[] {
  const neq = yOld.length;
  const nInterp = Q[0].length;
  const result = new Array<number>(neq);

  for (let i = 0; i < neq; i++) {
    // Horner: Q[i][0] + x*(Q[i][1] + x*(Q[i][2] + x*Q[i][3]))
    let val = Q[i][nInterp - 1];
    for (let j = nInterp - 2; j >= 0; j--) {
      val = val * x + Q[i][j];
    }
    result[i] = yOld[i] + h * x * val;
  }
  return result;
}

/**
 * Evaluate dense output for a step at fractional position x in [0,1].
 * Dispatches between Q-based (RK45/RK23) and F-based (DOP853) formats.
 */
export function evalStepAt(step: StepData, x: number): number[] {
  if (step.Q) return denseOutputEval(step.yOld, step.Q, step.h, x);
  if (step.F) return dop853DenseEval(step.yOld, step.F, x);
  throw new Error("Step has no interpolation data");
}

/**
 * DOP853 dense output evaluation.
 * Matches scipy Dop853DenseOutput._call_impl:
 *   Nested polynomial in x and (1-x) over F[0..nPower-1].
 */
function dop853DenseEval(yOld: number[], F: number[][], x: number): number[] {
  const neq = yOld.length;
  const nPower = F.length;
  const result = new Array<number>(neq).fill(0);
  const x1m = 1 - x;

  // Evaluate reversed: F[nPower-1], F[nPower-2], ..., F[0]
  // Even index (in reversed order) multiplies by x, odd by (1-x)
  for (let i = 0; i < nPower; i++) {
    const fi = F[nPower - 1 - i];
    for (let j = 0; j < neq; j++) result[j] += fi[j];
    if (i % 2 === 0) {
      for (let j = 0; j < neq; j++) result[j] *= x;
    } else {
      for (let j = 0; j < neq; j++) result[j] *= x1m;
    }
  }
  for (let j = 0; j < neq; j++) result[j] += yOld[j];
  return result;
}

// ── Solver options & result ─────────────────────────────────────────

export interface OdeOptions {
  relTol: number;
  absTol: number;
  maxStep: number;
  initialStep: number;
  events?: (t: number, y: number[]) => [number[], boolean[], number[]];
}

export interface OdeResult {
  t: number[];
  y: number[][];
  te: number[];
  ye: number[][];
  ie: number[];
  steps: StepData[];
}

// ── Core solver ─────────────────────────────────────────────────────

/**
 * Solve y' = f(t,y) using an adaptive embedded Runge-Kutta method.
 *
 * Step control follows scipy RungeKutta._step_impl exactly.
 */
export function solveRK(
  tableau: RKTableau,
  f: (t: number, y: number[]) => number[],
  tspan: number[],
  y0: number[],
  opts: Partial<OdeOptions>
): OdeResult {
  const t0 = tspan[0];
  const tf = tspan[tspan.length - 1];
  const direction = tf >= t0 ? 1 : -1;
  const neq = y0.length;
  const { nStages, nInterp } = tableau;
  const nK = nStages + 1; // K has nStages+1 rows (last = f_new)

  const relTol = opts.relTol ?? 1e-3;
  const absTol = opts.absTol ?? 1e-6;
  const maxStep = opts.maxStep ?? Infinity;
  const errorExponent = -1 / (tableau.errorOrder + 1);

  // K storage: nK x neq (stages stored in rows)
  const K: number[][] = new Array(nK);
  for (let s = 0; s < nK; s++) K[s] = new Array(neq);

  let t = t0;
  let y = y0.slice();

  // Initial derivative
  const f0 = f(t, y);
  copyInto(K[0], f0);

  // Initial step size (scipy: select_initial_step)
  let hAbs: number;
  if (opts.initialStep != null && opts.initialStep > 0) {
    hAbs = opts.initialStep;
  } else {
    hAbs = selectInitialStep(
      f,
      t0,
      y0,
      tf,
      f0,
      direction,
      tableau.errorOrder,
      relTol,
      absTol
    );
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

  // ── Main integration loop (one outer iteration = one scipy _step_impl call)
  while (direction * (tf - t) > 0 && !terminated) {
    // scipy: min_step = 10 * |nextafter(t, direction*inf) - t|
    const minStep = 10 * Number.EPSILON * Math.abs(t);

    // scipy: clamp h_abs to [min_step, max_step]
    if (hAbs > maxStep) hAbs = maxStep;
    else if (hAbs < minStep) hAbs = minStep;

    let stepAccepted = false;
    let stepRejected = false;

    // ── Inner retry loop (scipy: while not step_accepted)
    while (!stepAccepted) {
      if (hAbs < minStep) {
        throw new Error(
          `${tableau.name}: step size too small at t = ${t}. ` +
            `The problem may be stiff.`
        );
      }

      let h = hAbs * direction;
      let tNew = t + h;

      // Don't overshoot t_bound
      if (direction * (tNew - tf) > 0) {
        tNew = tf;
      }
      h = tNew - t;
      hAbs = Math.abs(h);

      // ── rk_step (scipy: rk_step function) ──────────────────────
      // K[0] already set
      for (let s = 1; s < nStages; s++) {
        const ts = t + tableau.C[s] * h;
        const ys = new Array<number>(neq);
        for (let i = 0; i < neq; i++) {
          let dy = 0;
          const aRow = tableau.A[s];
          for (let j = 0; j < s; j++) {
            dy += aRow[j] * K[j][i];
          }
          ys[i] = y[i] + dy * h;
        }
        copyInto(K[s], f(ts, ys));
      }

      // y_new = y + h * B @ K[:nStages]
      const yNew = new Array<number>(neq);
      for (let i = 0; i < neq; i++) {
        let sum = 0;
        for (let s = 0; s < nStages; s++) {
          sum += tableau.B[s] * K[s][i];
        }
        yNew[i] = y[i] + h * sum;
      }

      // f_new = f(t_new, y_new); K[nStages] = f_new
      const fNew = f(tNew, yNew);
      copyInto(K[nStages], fNew);

      // ── Error estimation (scipy: _estimate_error_norm) ─────────
      // error = h * E @ K, scale = atol + max(|y|,|y_new|) * rtol
      let errNormSq = 0;
      for (let i = 0; i < neq; i++) {
        let errI = 0;
        for (let s = 0; s < nK; s++) {
          errI += tableau.E[s] * K[s][i];
        }
        errI *= h;
        const sc =
          absTol + relTol * Math.max(Math.abs(y[i]), Math.abs(yNew[i]));
        errNormSq += (errI / sc) ** 2;
      }
      const errNorm = Math.sqrt(errNormSq / neq);

      // ── Step acceptance (scipy: _step_impl) ────────────────────
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
        if (stepRejected) {
          factor = Math.min(1, factor);
        }
        hAbs *= factor;
        stepAccepted = true;

        // Dense output: Q = K^T P
        const Q = computeQ(K, tableau.P, neq, nK, nInterp);

        const stepData: StepData = {
          tOld: t,
          tNew,
          h,
          yOld: y.slice(),
          yNew: yNew.slice(),
          Q,
        };
        steps.push(stepData);

        // Event detection
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
              ie.push(ei + 1); // 1-based
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

        // Advance state; FSAL: K[0] = f_new for next step
        t = tNew;
        y = yNew;
        copyInto(K[0], fNew);
      } else {
        // Step rejected
        hAbs *= Math.max(MIN_FACTOR, SAFETY * Math.pow(errNorm, errorExponent));
        stepRejected = true;
      }
    }
  }

  return { t: tOut, y: yOut, te, ye, ie, steps };
}

// ── Interpolation at user-requested time points ─────────────────────

export function interpolateAtPoints(
  result: OdeResult,
  tRequested: number[]
): { t: number[]; y: number[][] } {
  const { steps } = result;
  const nSteps = steps.length;
  const tOut: number[] = [];
  const yOut: number[][] = [];

  for (const tReq of tRequested) {
    // Find the step interval containing tReq
    let idx = 0;
    for (let s = 0; s < nSteps; s++) {
      const lo = Math.min(steps[s].tOld, steps[s].tNew);
      const hi = Math.max(steps[s].tOld, steps[s].tNew);
      if (tReq >= lo - 1e-14 && tReq <= hi + 1e-14) {
        idx = s;
        break;
      }
      idx = s; // fallback to last
    }
    if (idx >= nSteps) idx = nSteps - 1;

    const step = steps[idx];
    if (Math.abs(step.h) < 1e-300) {
      tOut.push(tReq);
      yOut.push(step.yOld.slice());
      continue;
    }

    const x = Math.max(0, Math.min(1, (tReq - step.tOld) / step.h));
    tOut.push(tReq);
    yOut.push(evalStepAt(step, x));
  }

  return { t: tOut, y: yOut };
}

// ── Helpers ─────────────────────────────────────────────────────────

function copyInto(dst: number[], src: number[]): void {
  for (let i = 0; i < src.length; i++) dst[i] = src[i];
}

/** Compute Q = K^T @ P, returning neq x nInterp matrix. */
function computeQ(
  K: number[][],
  P: number[][],
  neq: number,
  nK: number,
  nInterp: number
): number[][] {
  const Q: number[][] = new Array(neq);
  for (let i = 0; i < neq; i++) {
    Q[i] = new Array(nInterp);
    for (let j = 0; j < nInterp; j++) {
      let sum = 0;
      for (let s = 0; s < nK; s++) {
        sum += K[s][i] * P[s][j];
      }
      Q[i][j] = sum;
    }
  }
  return Q;
}

/**
 * Estimate initial step size.
 * Matches scipy common.select_initial_step.
 */
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

  // scale = atol + |y0| * rtol
  // d0 = rms_norm(y0 / scale), d1 = rms_norm(f0 / scale)
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

  // Explicit Euler step to estimate second derivative
  const y1 = new Array<number>(n);
  for (let i = 0; i < n; i++) y1[i] = y0[i] + h0 * direction * f0[i];
  const f1 = fun(t0 + h0 * direction, y1);

  // d2 = rms_norm((f1 - f0) / scale) / h0
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

/** Bisect to find the event time within a step using dense output. */
function bisectEvent(
  eventIdx: number,
  step: StepData,
  events: (t: number, y: number[]) => [number[], boolean[], number[]]
): [number, number[]] {
  const { tOld, h } = step;
  let lo = 0; // fractional position in [0, 1]
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

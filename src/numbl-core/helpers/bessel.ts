/**
 * Bessel function implementations (pure numeric, no runtime dependencies).
 *
 * Provides besselj, bessely, besseli, besselk for real-valued arguments.
 *
 * The rational polynomial approximations for J0, J1, Y0, Y1 are derived from
 * the Cephes Math Library Release 2.8 (June 2000) by Stephen L. Moshier.
 * Original Cephes code: Copyright 1984-2000 by Stephen L. Moshier.
 * Used under the BSD license with permission of the author.
 *
 * The underlying minimax polynomial coefficients are based on the work of
 * W.J. Cody, "Algorithm 715: SPECFUN", ACM Trans. Math. Software 19(1),
 * pp. 22-32, 1993.
 */

/* eslint-disable no-loss-of-precision */
// The Cephes coefficients below are written with ~18 significant digits
// (matching the published values). JavaScript rounds to the nearest
// representable double, which is the desired behavior.

// Lanczos approximation for gamma function (also used by gamma/factorial in math.ts)
export function lanczosGamma(x: number): number {
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    return Math.PI / (Math.sin(Math.PI * x) * lanczosGamma(1 - x));
  }

  x -= 1;
  let a = coef[0];
  for (let i = 1; i < coef.length; i++) {
    a += coef[i] / (x + i);
  }

  const t = x + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
}

// ── besselj: Bessel function of the first kind ─────────────────────────
// Integer orders 0,1 use Cephes rational polynomial approximations (full
// double precision). Higher integer orders use forward recurrence from
// J0/J1. Non-integer orders fall back to the power series.

export function besselj(nu: number, x: number): number {
  if (x === 0) {
    return nu === 0 ? 1 : 0;
  }
  if (x < 0 && nu !== Math.floor(nu)) {
    return NaN;
  }
  // For negative x with integer order: J_n(-x) = (-1)^n * J_n(x)
  if (x < 0) {
    const n = Math.round(nu);
    return (n % 2 === 0 ? 1 : -1) * besselj(nu, -x);
  }
  // For negative order with integer nu: J_{-n}(x) = (-1)^n * J_n(x)
  if (nu < 0 && nu === Math.floor(nu)) {
    const n = Math.round(-nu);
    return (n % 2 === 0 ? 1 : -1) * besselj(-nu, x);
  }
  // Integer orders: use Cephes J0/J1 + recurrence
  if (nu === Math.floor(nu) && nu >= 0) {
    const n = Math.round(nu);
    if (n === 0) return _cephes_j0(x);
    if (n === 1) return _cephes_j1(x);
    // Forward recurrence is unstable when x < n; use series for those cases
    if (x < n) return _besseljSeries(n, x);
    let jm1 = _cephes_j0(x);
    let j = _cephes_j1(x);
    for (let k = 1; k < n; k++) {
      const jnext = ((2 * k) / x) * j - jm1;
      jm1 = j;
      j = jnext;
    }
    return j;
  }
  // Non-integer orders: series for small x, asymptotic for large x
  if (x <= 25 + Math.abs(nu) / 2) {
    return _besseljSeries(nu, x);
  }
  return _hankelBesselj(nu, x);
}

// ── Cephes J0: rational polynomial approximation ──────────────────────
// Coefficients from the Cephes Math Library by Stephen L. Moshier.
// Copyright 1984-2000 Stephen L. Moshier. BSD license.
// Peak error ~4e-16 over [0, ∞).

function _cephes_j0(x: number): number {
  if (x < 0) x = -x; // J0 is even
  if (x <= 5.0) {
    const z = x * x;
    // J0(x) = (z - DR1)*(z - DR2) * P(z)/Q(z)
    const DR1 = 5.78318596294678452118;
    const DR2 = 30.4712623436620863991;
    const RP = [
      -4.79443220978201773821e9, 1.95617491946556577543e12,
      -2.49248344360967716204e14, 9.70862251047306323952e15,
    ];
    const RQ = [
      1.0, 4.99563147152651017219e2, 1.73785401676374683123e5,
      4.84409658339962045305e7, 1.11855537045356834862e10,
      2.11277520115489217587e12, 3.10518229857422583814e14,
      3.18121955943204943306e16, 1.71086294081043136091e18,
    ];
    const p = _polyeval(RP, z);
    const q = _polyeval(RQ, z);
    return (z - DR1) * (z - DR2) * (p / q);
  }
  // x > 5: asymptotic form
  return _j0_large(x);
}

function _j0_large(x: number): number {
  // J0(x) = sqrt(2/(pi*x)) * [P0(x)*cos(x0) - Q0(x)*sin(x0)]
  // where x0 = x - pi/4
  const PP = [
    7.96936729297347051624e-4, 8.28352392107440799803e-2,
    1.23953371646414299388, 5.4472500305876877509, 8.74716500199817011941,
    5.30324038235394892183, 9.99999999999999997821e-1,
  ];
  const PQ = [
    9.24408810558863637013e-4, 8.56288474354474431428e-2,
    1.25352743901058953537, 5.47097740330417105182, 8.76190883237069594232,
    5.30605288235394617618, 1.00000000000000000218,
  ];
  const QP = [
    -1.13663838898469149931e-2, -1.28252718670509318512,
    -1.95539544257735972385e1, -9.32060152123768231369e1,
    -1.77681167980488050595e2, -1.47077505154951170175e2,
    -5.1410532676659933022e1, -6.05014350600728481186,
  ];
  const QQ = [
    1.0, 6.43178256118178023184e1, 8.56430025976980587198e2,
    3.88240183605401609683e3, 7.24046774195652478189e3,
    5.93072701187316984827e3, 2.06209331660327847417e3,
    2.42005740240291393179e2,
  ];
  const w = 5.0 / x;
  const z = w * w;
  const p = _polyeval(PP, z) / _polyeval(PQ, z);
  const q = _polyeval(QP, z) / _polyeval(QQ, z);
  const xn = x - Math.PI / 4;
  return (
    Math.sqrt(2 / (Math.PI * x)) * (p * Math.cos(xn) - w * q * Math.sin(xn))
  );
}

// ── Cephes J1: rational polynomial approximation ──────────────────────

function _cephes_j1(x: number): number {
  const sign = x < 0 ? -1 : 1;
  if (x < 0) x = -x; // J1 is odd
  if (x <= 5.0) {
    const z = x * x;
    const Z1 = 1.46819706421238932572e1;
    const Z2 = 4.92184563216946036703e1;
    const RP = [
      -8.99971225705559398224e8, 4.52228297998194034323e11,
      -7.27494245221818276015e13, 3.68295732863852883286e15,
    ];
    const RQ = [
      1.0, 6.20836478118054335476e2, 2.56987256757748830383e5,
      8.35146791431949253037e7, 2.21511595479792499675e10,
      4.74914122079991414898e12, 7.84369607876235854894e14,
      8.95222336184627338078e16, 5.32278620332680085395e18,
    ];
    const p = _polyeval(RP, z);
    const q = _polyeval(RQ, z);
    return sign * x * (z - Z1) * (z - Z2) * (p / q);
  }
  return sign * _j1_large(x);
}

function _j1_large(x: number): number {
  // J1(x) = sqrt(2/(pi*x)) * [P1(x)*cos(x1) - Q1(x)*sin(x1)]
  // where x1 = x - 3*pi/4
  const PP = [
    7.62125616208173112003e-4, 7.31397056940917570436e-2,
    1.12719608129684925192, 5.11207951146807644818, 8.42404590141772420927,
    5.21451598682361821619, 1.00000000000000000254,
  ];
  const PQ = [
    5.71323128072548699714e-4, 6.88455908754495404082e-2,
    1.10514232634061696926, 5.07386386128601488557, 8.39985554327604159757,
    5.20982848682361821619, 9.99999999999999997461e-1,
  ];
  const QP = [
    5.10862594750176621635e-2, 4.9821387295123344942, 7.58238284132545283818e1,
    3.667796093601507778e2, 7.10856304998926107277e2, 5.97489612400613639965e2,
    2.11688757100572135698e2, 2.52070205858023719784e1,
  ];
  const QQ = [
    1.0, 7.42373277035675149943e1, 1.05644886038262816351e3,
    4.98641058337653607651e3, 9.56231892404756170795e3, 7.9970416044735068365e3,
    2.826192785176390966e3, 3.36093607810698293419e2,
  ];
  const w = 5.0 / x;
  const z = w * w;
  const p = _polyeval(PP, z) / _polyeval(PQ, z);
  const q = _polyeval(QP, z) / _polyeval(QQ, z);
  const xn = x - (3 * Math.PI) / 4;
  return (
    Math.sqrt(2 / (Math.PI * x)) * (p * Math.cos(xn) - w * q * Math.sin(xn))
  );
}

/** Evaluate polynomial (Horner, descending powers): c[0]*x^n + c[1]*x^(n-1) + ... + c[n] */
function _polyeval(coeffs: number[], x: number): number {
  let r = coeffs[0];
  for (let i = 1; i < coeffs.length; i++) {
    r = r * x + coeffs[i];
  }
  return r;
}

/** Power series for non-integer order besselj (fallback) */
function _besseljSeries(nu: number, x: number): number {
  const halfX = x / 2;
  let term = Math.pow(halfX, nu) / lanczosGamma(nu + 1);
  let sum = term;
  const x2over4 = -(halfX * halfX);
  for (let k = 1; k <= 300; k++) {
    term *= x2over4 / (k * (k + nu));
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * 1e-16) break;
  }
  return sum;
}

/** Hankel asymptotic for non-integer order besselj (fallback) */
function _hankelBesselj(nu: number, x: number): number {
  const mu = 4 * nu * nu;
  const chi = x - (nu / 2 + 0.25) * Math.PI;
  let P = 1,
    Q = 0,
    termP = 1,
    termQ = 1;
  for (let k = 0; k < 30; k++) {
    if (k > 0) {
      termP *=
        (-(mu - (4 * k - 3) * (4 * k - 3)) * (mu - (4 * k - 1) * (4 * k - 1))) /
        ((2 * k - 1) * (2 * k) * 64 * x * x);
      P += termP;
    }
    termQ =
      (k === 0 ? 1 : termQ) *
      (k === 0
        ? (mu - 1) / (8 * x)
        : (-(mu - (4 * k - 1) * (4 * k - 1)) *
            (mu - (4 * k + 1) * (4 * k + 1))) /
          (2 * k * (2 * k + 1) * 64 * x * x));
    if (k === 0) {
      Q = (mu - 1) / (8 * x);
    } else {
      Q += termQ;
    }
    if (Math.abs(termP) + Math.abs(termQ) < 1e-16) break;
  }
  return Math.sqrt(2 / (Math.PI * x)) * (P * Math.cos(chi) - Q * Math.sin(chi));
}

// ── bessely: Bessel function of the second kind ────────────────────────
// Integer orders 0,1 use Cephes rational polynomial approximations.
// Higher integer orders use forward recurrence from Y0/Y1.

export function bessely(nu: number, x: number): number {
  if (x <= 0) return NaN;
  if (nu === Math.floor(nu)) {
    return _besselyInteger(Math.round(nu), x);
  }
  // Y_nu = (J_nu * cos(nu*pi) - J_{-nu}) / sin(nu*pi)
  const sinPi = Math.sin(nu * Math.PI);
  return (besselj(nu, x) * Math.cos(nu * Math.PI) - besselj(-nu, x)) / sinPi;
}

function _besselyInteger(n: number, x: number): number {
  if (n < 0) {
    return (n % 2 === 0 ? 1 : -1) * _besselyInteger(-n, x);
  }
  if (n === 0) return _cephes_y0(x);
  if (n === 1) return _cephes_y1(x);
  // Forward recurrence: Y_{n+1}(x) = (2n/x) * Y_n(x) - Y_{n-1}(x)
  let ym1 = _cephes_y0(x);
  let y = _cephes_y1(x);
  for (let k = 1; k < n; k++) {
    const ynext = ((2 * k) / x) * y - ym1;
    ym1 = y;
    y = ynext;
  }
  return y;
}

// ── Cephes Y0: rational polynomial approximation ──────────────────────

function _cephes_y0(x: number): number {
  if (x <= 5.0) {
    // Y0(x) = P(x^2)/Q(x^2) + (2/pi)*ln(x)*J0(x)
    const YP = [
      1.55924367855235737965e4, -1.46639295903971606143e7,
      5.43526477051876500413e9, -9.82136065717911466409e11,
      8.75906394395366999549e13, -3.46628303384729719441e15,
      4.42733268572569800351e16, -1.84950800436986690637e16,
    ];
    const YQ = [
      1.0, 1.04128353664259848412e3, 6.26107330137134956842e5,
      2.68919633393814121987e8, 8.64002487103935000337e10,
      2.02979612750105546709e13, 3.17157752842975028269e15,
      2.50596256172653059228e17,
    ];
    const z = x * x;
    const p = _polyeval(YP, z);
    const q = _polyeval(YQ, z);
    return p / q + (2 / Math.PI) * Math.log(x) * _cephes_j0(x);
  }
  // x > 5: asymptotic form (same P/Q polynomials as J0 large)
  return _y0_large(x);
}

function _y0_large(x: number): number {
  // Y0(x) = sqrt(2/(pi*x)) * [P0(x)*sin(x0) + Q0(x)*cos(x0)]
  const PP = [
    7.96936729297347051624e-4, 8.28352392107440799803e-2,
    1.23953371646414299388, 5.4472500305876877509, 8.74716500199817011941,
    5.30324038235394892183, 9.99999999999999997821e-1,
  ];
  const PQ = [
    9.24408810558863637013e-4, 8.56288474354474431428e-2,
    1.25352743901058953537, 5.47097740330417105182, 8.76190883237069594232,
    5.30605288235394617618, 1.00000000000000000218,
  ];
  const QP = [
    -1.13663838898469149931e-2, -1.28252718670509318512,
    -1.95539544257735972385e1, -9.32060152123768231369e1,
    -1.77681167980488050595e2, -1.47077505154951170175e2,
    -5.1410532676659933022e1, -6.05014350600728481186,
  ];
  const QQ = [
    1.0, 6.43178256118178023184e1, 8.56430025976980587198e2,
    3.88240183605401609683e3, 7.24046774195652478189e3,
    5.93072701187316984827e3, 2.06209331660327847417e3,
    2.42005740240291393179e2,
  ];
  const w = 5.0 / x;
  const z = w * w;
  const p = _polyeval(PP, z) / _polyeval(PQ, z);
  const q = _polyeval(QP, z) / _polyeval(QQ, z);
  const xn = x - Math.PI / 4;
  return (
    Math.sqrt(2 / (Math.PI * x)) * (p * Math.sin(xn) + w * q * Math.cos(xn))
  );
}

// ── Cephes Y1: rational polynomial approximation ──────────────────────

function _cephes_y1(x: number): number {
  if (x <= 5.0) {
    // Y1(x) = x*P(x^2)/Q(x^2) + (2/pi)*(J1(x)*ln(x) - 1/x)
    const YP = [
      1.2632047479017802644e9, -6.47355876379160291031e11,
      1.14509511541823727583e14, -8.12770255501325109621e15,
      2.02439475713594898196e17, -7.78877196265950026825e17,
    ];
    const YQ = [
      1.0, 5.94301592346128195359e2, 2.35564092943068577943e5,
      7.3481194445972170566e7, 1.87601316108706159478e10,
      3.88231277496238566008e12, 6.20557727146953693363e14,
      6.87141087355300489866e16, 3.97270608116560655612e18,
    ];
    const z = x * x;
    const p = _polyeval(YP, z);
    const q = _polyeval(YQ, z);
    return x * (p / q) + (2 / Math.PI) * (_cephes_j1(x) * Math.log(x) - 1 / x);
  }
  return _y1_large(x);
}

function _y1_large(x: number): number {
  // Y1(x) = sqrt(2/(pi*x)) * [P1(x)*sin(x1) + Q1(x)*cos(x1)]
  const PP = [
    7.62125616208173112003e-4, 7.31397056940917570436e-2,
    1.12719608129684925192, 5.11207951146807644818, 8.42404590141772420927,
    5.21451598682361821619, 1.00000000000000000254,
  ];
  const PQ = [
    5.71323128072548699714e-4, 6.88455908754495404082e-2,
    1.10514232634061696926, 5.07386386128601488557, 8.39985554327604159757,
    5.20982848682361821619, 9.99999999999999997461e-1,
  ];
  const QP = [
    5.10862594750176621635e-2, 4.9821387295123344942, 7.58238284132545283818e1,
    3.667796093601507778e2, 7.10856304998926107277e2, 5.97489612400613639965e2,
    2.11688757100572135698e2, 2.52070205858023719784e1,
  ];
  const QQ = [
    1.0, 7.42373277035675149943e1, 1.05644886038262816351e3,
    4.98641058337653607651e3, 9.56231892404756170795e3, 7.9970416044735068365e3,
    2.826192785176390966e3, 3.36093607810698293419e2,
  ];
  const w = 5.0 / x;
  const z = w * w;
  const p = _polyeval(PP, z) / _polyeval(PQ, z);
  const q = _polyeval(QP, z) / _polyeval(QQ, z);
  const xn = x - (3 * Math.PI) / 4;
  return (
    Math.sqrt(2 / (Math.PI * x)) * (p * Math.sin(xn) + w * q * Math.cos(xn))
  );
}

// ── besseli: Modified Bessel function of the first kind ────────────────

export function besseli(nu: number, x: number): number {
  if (x === 0) {
    return nu === 0 ? 1 : 0;
  }
  if (x < 0 && nu !== Math.floor(nu)) {
    return NaN;
  }
  if (x < 0) {
    const n = Math.round(nu);
    return (n % 2 === 0 ? 1 : -1) * besseli(nu, -x);
  }
  if (nu < 0 && nu === Math.floor(nu)) {
    return besseli(-nu, x); // I_{-n} = I_n for integer n
  }
  const halfX = x / 2;
  let term = Math.pow(halfX, nu) / lanczosGamma(nu + 1);
  let sum = term;
  const x2over4 = halfX * halfX;
  for (let k = 1; k <= 300; k++) {
    term *= x2over4 / (k * (k + nu));
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * 1e-16) break;
  }
  return sum;
}

// ── besselk: Modified Bessel function of the second kind ───────────────

export function besselk(nu: number, x: number): number {
  if (x <= 0) return NaN;
  if (nu < 0) nu = -nu; // K_{-nu} = K_nu
  if (nu === Math.floor(nu)) {
    return besselkInteger(Math.round(nu), x);
  }
  // K_nu = (pi/2) * (I_{-nu} - I_nu) / sin(nu*pi)
  const sinPi = Math.sin(nu * Math.PI);
  return ((Math.PI / 2) * (besseli(-nu, x) - besseli(nu, x))) / sinPi;
}

function besselkInteger(n: number, x: number): number {
  if (n === 0) return besselk0(x);
  if (n === 1) return besselk1(x);
  // Forward recurrence: K_{n+1}(x) = (2n/x) * K_n(x) + K_{n-1}(x)
  let km1 = besselk0(x);
  let k = besselk1(x);
  for (let i = 1; i < n; i++) {
    const knext = ((2 * i) / x) * k + km1;
    km1 = k;
    k = knext;
  }
  return k;
}

function besselk0(x: number): number {
  const euler = 0.5772156649015329;
  const i0 = besseli(0, x);
  const logTerm = -(Math.log(x / 2) + euler) * i0;
  const halfX = x / 2;
  const x2over4 = halfX * halfX;
  let term = 1;
  let hk = 0;
  let sum = 0;
  for (let k = 1; k <= 300; k++) {
    hk += 1 / k;
    term *= x2over4 / (k * k);
    sum += hk * term;
    if (Math.abs(term * hk) < Math.abs(sum) * 1e-16) break;
  }
  return logTerm + sum;
}

function besselk1(x: number): number {
  // A&S 9.6.11: K_1(x) = (1/x) + (x/2)*ln(x/2)*S1 - (x/4)*S2
  // where S1 = sum (x^2/4)^k / (k!(k+1)!), S2 = sum (psi(k+1)+psi(k+2)) * (x^2/4)^k / (k!(k+1)!)
  const euler = 0.5772156649015329;
  const halfX = x / 2;
  const x2over4 = halfX * halfX;

  let term = 1;
  let psi1 = -euler; // psi(1) = -gamma
  let psi2 = -euler + 1; // psi(2) = -gamma + 1
  let S1 = term;
  let S2 = (psi1 + psi2) * term;
  for (let k = 1; k <= 300; k++) {
    term *= x2over4 / (k * (k + 1));
    psi1 += 1 / k; // psi(k+1)
    psi2 += 1 / (k + 1); // psi(k+2)
    S1 += term;
    S2 += (psi1 + psi2) * term;
    if (Math.abs(term) < Math.abs(S1) * 1e-16) break;
  }
  return 1 / x + halfX * Math.log(halfX) * S1 - (halfX / 2) * S2;
}

// ── Airy functions via Bessel function relations ────────────────────────
// Reference: DLMF 9.6

/**
 * Airy function of the first kind, Ai(x).
 */
export function airyAi(x: number): number {
  if (x === 0) {
    return 1 / (Math.pow(3, 2 / 3) * lanczosGamma(2 / 3));
  }
  if (x > 0) {
    const zeta = (2 / 3) * Math.pow(x, 1.5);
    return (1 / Math.PI) * Math.sqrt(x / 3) * besselk(1 / 3, zeta);
  }
  // x < 0
  const t = -x;
  const zeta = (2 / 3) * Math.pow(t, 1.5);
  return (Math.sqrt(t) / 3) * (besselj(1 / 3, zeta) + besselj(-1 / 3, zeta));
}

/**
 * Derivative of Airy function of the first kind, Ai'(x).
 */
export function airyAiPrime(x: number): number {
  if (x === 0) {
    return -1 / (Math.pow(3, 1 / 3) * lanczosGamma(1 / 3));
  }
  if (x > 0) {
    const zeta = (2 / 3) * Math.pow(x, 1.5);
    return -(x / (Math.PI * Math.sqrt(3))) * besselk(2 / 3, zeta);
  }
  // x < 0
  const t = -x;
  const zeta = (2 / 3) * Math.pow(t, 1.5);
  return (t / 3) * (besselj(2 / 3, zeta) - besselj(-2 / 3, zeta));
}

/**
 * Airy function of the second kind, Bi(x).
 */
export function airyBi(x: number): number {
  if (x === 0) {
    return 1 / (Math.pow(3, 1 / 6) * lanczosGamma(2 / 3));
  }
  if (x > 0) {
    const zeta = (2 / 3) * Math.pow(x, 1.5);
    return Math.sqrt(x / 3) * (besseli(1 / 3, zeta) + besseli(-1 / 3, zeta));
  }
  // x < 0
  const t = -x;
  const zeta = (2 / 3) * Math.pow(t, 1.5);
  return Math.sqrt(t / 3) * (besselj(-1 / 3, zeta) - besselj(1 / 3, zeta));
}

/**
 * Derivative of Airy function of the second kind, Bi'(x).
 */
export function airyBiPrime(x: number): number {
  if (x === 0) {
    return Math.pow(3, 1 / 6) / lanczosGamma(1 / 3);
  }
  if (x > 0) {
    const zeta = (2 / 3) * Math.pow(x, 1.5);
    return (x / Math.sqrt(3)) * (besseli(2 / 3, zeta) + besseli(-2 / 3, zeta));
  }
  // x < 0
  const t = -x;
  const zeta = (2 / 3) * Math.pow(t, 1.5);
  return (t / Math.sqrt(3)) * (besselj(2 / 3, zeta) + besselj(-2 / 3, zeta));
}

// ── Complex Airy functions via Maclaurin series ─────────────────────────
// The Airy ODE y'' = z*y gives the recurrence:
//   a_{n+2} = a_{n-1} / ((n+2)(n+1)) for n >= 1, a_2 = 0
// Non-zero coefficients fall on indices 0,3,6,... and 1,4,7,...
// We compute two basis series:
//   f(z) = sum_{k=0}^inf alpha_k * z^{3k}   (with f(0)=1)
//   g(z) = z * sum_{k=0}^inf beta_k * z^{3k} (with g'(0)=1)
// Then Ai(z) = Ai(0)*f(z) + Ai'(0)*g(z), etc.

interface CR {
  re: number;
  im: number;
}

/**
 * Compute all four complex Airy functions at once: Ai(z), Ai'(z), Bi(z), Bi'(z).
 */
export function airyAllComplex(
  zr: number,
  zi: number
): { ai: CR; aip: CR; bi: CR; bip: CR } {
  // z^3
  const z3r = zr * zr * zr - 3 * zr * zi * zi;
  const z3i = 3 * zr * zr * zi - zi * zi * zi;

  // Accumulate series terms
  let tfr = 1,
    tfi = 0; // term for f: alpha_k * z^{3k}
  let tgr = 1,
    tgi = 0; // term for g: beta_k * z^{3k}
  let sfr = 1,
    sfi = 0; // sum_f
  let sgr = 1,
    sgi = 0; // sum_g
  let sfpr = 0,
    sfpi = 0; // sum of 3k * alpha_k * z^{3k} (for f')
  let sgpr = 1,
    sgpi = 0; // sum of (3k+1) * beta_k * z^{3k} (for g')

  for (let k = 1; k <= 100; k++) {
    const df = 3 * k * (3 * k - 1);
    const dg = (3 * k + 1) * (3 * k);

    // term_f *= z3 / df
    const nfr = (tfr * z3r - tfi * z3i) / df;
    const nfi = (tfr * z3i + tfi * z3r) / df;
    tfr = nfr;
    tfi = nfi;

    // term_g *= z3 / dg
    const ngr = (tgr * z3r - tgi * z3i) / dg;
    const ngi = (tgr * z3i + tgi * z3r) / dg;
    tgr = ngr;
    tgi = ngi;

    sfr += tfr;
    sfi += tfi;
    sgr += tgr;
    sgi += tgi;

    const m3k = 3 * k;
    sfpr += m3k * tfr;
    sfpi += m3k * tfi;
    sgpr += (m3k + 1) * tgr;
    sgpi += (m3k + 1) * tgi;

    const mag = Math.abs(tfr) + Math.abs(tfi) + Math.abs(tgr) + Math.abs(tgi);
    const ref =
      Math.abs(sfr) + Math.abs(sfi) + Math.abs(sgr) + Math.abs(sgi) + 1e-300;
    if (mag < 1e-16 * ref) break;
  }

  // f(z) = sum_f
  // g(z) = z * sum_g
  const gr = zr * sgr - zi * sgi;
  const gi = zr * sgi + zi * sgr;

  // f'(z) = sum_fp / z  (0 at z=0)
  let fpr: number, fpi: number;
  if (zr === 0 && zi === 0) {
    fpr = 0;
    fpi = 0;
  } else {
    const zabs2 = zr * zr + zi * zi;
    fpr = (sfpr * zr + sfpi * zi) / zabs2;
    fpi = (sfpi * zr - sfpr * zi) / zabs2;
  }

  // g'(z) = sum_gp
  const gpr = sgpr;
  const gpi = sgpi;

  // Constants at z=0
  const ai0 = airyAi(0);
  const aip0 = airyAiPrime(0);
  const bi0 = airyBi(0);
  const bip0 = airyBiPrime(0);

  return {
    ai: { re: ai0 * sfr + aip0 * gr, im: ai0 * sfi + aip0 * gi },
    aip: { re: ai0 * fpr + aip0 * gpr, im: ai0 * fpi + aip0 * gpi },
    bi: { re: bi0 * sfr + bip0 * gr, im: bi0 * sfi + bip0 * gi },
    bip: { re: bi0 * fpr + bip0 * gpr, im: bi0 * fpi + bip0 * gpi },
  };
}

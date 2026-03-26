/**
 * Bessel function implementations (pure numeric, no runtime dependencies).
 *
 * Provides besselj, bessely, besseli, besselk for real-valued arguments.
 */

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
  // Use series expansion for small x or small order
  if (x <= 25 + Math.abs(nu) / 2) {
    return besseljSeries(nu, x);
  }
  // Asymptotic expansion for large x
  return hankelBesselj(nu, x);
}

function besseljSeries(nu: number, x: number): number {
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

/**
 * Hankel asymptotic expansion for J_nu(x) for large x.
 */
function hankelBesselj(nu: number, x: number): number {
  const mu = 4 * nu * nu;
  const chi = x - (nu / 2 + 0.25) * Math.PI;
  let P = 1;
  let Q = 0;
  let termP = 1;
  let termQ = 1;
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

export function bessely(nu: number, x: number): number {
  if (x <= 0) return NaN;
  if (nu === Math.floor(nu)) {
    return besselyInteger(Math.round(nu), x);
  }
  // Y_nu = (J_nu * cos(nu*pi) - J_{-nu}) / sin(nu*pi)
  const sinPi = Math.sin(nu * Math.PI);
  return (besselj(nu, x) * Math.cos(nu * Math.PI) - besselj(-nu, x)) / sinPi;
}

function besselyInteger(n: number, x: number): number {
  if (n < 0) {
    return (n % 2 === 0 ? 1 : -1) * besselyInteger(-n, x);
  }
  if (n === 0) return bessely0(x);
  if (n === 1) return bessely1(x);
  // Forward recurrence: Y_{n+1}(x) = (2n/x) * Y_n(x) - Y_{n-1}(x)
  let ym1 = bessely0(x);
  let y = bessely1(x);
  for (let k = 1; k < n; k++) {
    const ynext = ((2 * k) / x) * y - ym1;
    ym1 = y;
    y = ynext;
  }
  return y;
}

function bessely0(x: number): number {
  if (x <= 25) {
    // Y_0(x) = (2/pi) * [(ln(x/2) + gamma) * J_0(x)
    //          + sum_{k=1}^inf (-1)^{k+1} * H_k * (x/2)^{2k} / (k!)^2 ]
    const euler = 0.5772156649015329;
    const j0 = besselj(0, x);
    const halfX = x / 2;
    const x2over4 = halfX * halfX;
    let term = 1;
    let hk = 0;
    let sum = 0;
    for (let k = 1; k <= 300; k++) {
      hk += 1 / k;
      term *= x2over4 / (k * k);
      sum += (k % 2 === 1 ? 1 : -1) * hk * term;
      if (Math.abs(term * hk) < Math.abs(sum) * 1e-16) break;
    }
    return (2 / Math.PI) * ((Math.log(halfX) + euler) * j0 + sum);
  }
  return hankelBessely(0, x);
}

function bessely1(x: number): number {
  if (x <= 25) {
    // Y_1(x) = (2/pi) * [(ln(x/2) + gamma) * J_1(x)]
    //          - (2/(pi*x))
    //          - (1/pi) * sum_{k=0}^inf (-1)^k * (H_k + H_{k+1}) * (x/2)^{2k+1} / (k!(k+1)!)
    const euler = 0.5772156649015329;
    const j1 = besselj(1, x);
    const halfX = x / 2;
    const x2over4 = halfX * halfX;
    let term = halfX;
    let hk = 0;
    let sum = 0;
    for (let k = 0; k <= 300; k++) {
      if (k > 0) {
        term *= x2over4 / (k * (k + 1));
        hk += 1 / k;
      }
      const hk1 = hk + 1 / (k + 1);
      sum += (k % 2 === 0 ? -1 : 1) * (hk + hk1) * term;
      if (k > 0 && Math.abs(term * (hk + hk1)) < Math.abs(sum) * 1e-16) break;
    }
    return (
      (2 / Math.PI) * ((Math.log(halfX) + euler) * j1 - 1 / x) +
      (1 / Math.PI) * sum
    );
  }
  return hankelBessely(1, x);
}

function hankelBessely(nu: number, x: number): number {
  const mu = 4 * nu * nu;
  const chi = x - (nu / 2 + 0.25) * Math.PI;
  let P = 1;
  let Q = 0;
  let termP = 1;
  let termQ = 1;
  for (let k = 0; k < 30; k++) {
    if (k > 0) {
      termP *=
        (-(mu - (4 * k - 3) * (4 * k - 3)) * (mu - (4 * k - 1) * (4 * k - 1))) /
        ((2 * k - 1) * (2 * k) * 64 * x * x);
      P += termP;
    }
    if (k === 0) {
      Q = (mu - 1) / (8 * x);
      termQ = Q;
    } else {
      termQ *=
        (-(mu - (4 * k - 1) * (4 * k - 1)) * (mu - (4 * k + 1) * (4 * k + 1))) /
        (2 * k * (2 * k + 1) * 64 * x * x);
      Q += termQ;
    }
    if (Math.abs(termP) + Math.abs(termQ) < 1e-16) break;
  }
  return Math.sqrt(2 / (Math.PI * x)) * (P * Math.sin(chi) + Q * Math.cos(chi));
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

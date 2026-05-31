// JS sibling of `cscalar.h`. JS has no native complex type, so each
// helper takes and returns `{re, im}` objects. Mirrors the C
// signatures so emitJs renders calls structurally identical to the
// C path (just with JS objects instead of `double _Complex`).
//
// Division (`mtoc2_cdiv`) lives in `cdiv.js` because it needs
// Smith's algorithm with signed-zero detection to match numbl
// byte-for-byte.

export function mtoc2_cmake(re, im) {
  return { re, im };
}
export function mtoc2_creal(z) {
  return z.re;
}
export function mtoc2_cimag(z) {
  return z.im;
}
export function mtoc2_cadd(a, b) {
  return { re: a.re + b.re, im: a.im + b.im };
}
export function mtoc2_csub(a, b) {
  return { re: a.re - b.re, im: a.im - b.im };
}
export function mtoc2_cmul(a, b) {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
export function mtoc2_cneg(z) {
  return { re: -z.re, im: -z.im };
}
export function mtoc2_cconj(z) {
  return { re: z.re, im: -z.im };
}
export function mtoc2_cabs(z) {
  return Math.hypot(z.re, z.im);
}
export function mtoc2_cangle(z) {
  return Math.atan2(z.im, z.re);
}
export function mtoc2_cnonzero(z) {
  return z.re !== 0 || z.im !== 0;
}
export function mtoc2_ceq(a, b) {
  return a.re === b.re && a.im === b.im;
}
export function mtoc2_cne(a, b) {
  return a.re !== b.re || a.im !== b.im;
}

// `cpow(a, b) = exp(b * log(a))`. Matches numbl / C99 `cpow` for
// finite inputs; special zero / pole cases follow the same path
// the C side picks (exp(NaN) = NaN, etc.).
function clog(z) {
  // Normalize a -0 real part to +0 so log(-0) (a real -0) gives
  // -Inf + 0i, not -Inf + πi — MATLAB / C99 treat -0 as 0, not as an
  // approach-from-below negative. atan2's -0 second arg only changes the
  // result when the first arg is also 0, so this affects nothing else.
  const reAngle = z.re === 0 ? 0 : z.re;
  return {
    re: Math.log(Math.hypot(z.re, z.im)),
    im: Math.atan2(z.im, reAngle),
  };
}
function cexp(z) {
  const m = Math.exp(z.re);
  // Real-valued input (im === 0): the imaginary lane is m*sin(0) = 0, but
  // when m overflows to Inf that becomes Inf*0 = NaN. Return a clean 0
  // (matches the interpreter's real exp and C99 cexp).
  if (z.im === 0) return { re: m, im: 0 };
  return { re: m * Math.cos(z.im), im: m * Math.sin(z.im) };
}
export function mtoc2_cpow(a, b) {
  // Zero base: match the interpreter's complexPow (helpers/arithmetic.ts)
  // — 0^0 = 1, a positive real part of the exponent gives 0 (regardless
  // of the imaginary part), and anything else gives Inf+0i. The old
  // guard required b.im===0 for the zero case and otherwise returned
  // NaN+NaNi.
  if (a.re === 0 && a.im === 0) {
    if (b.re === 0 && b.im === 0) return { re: 1, im: 0 };
    if (b.re > 0) return { re: 0, im: 0 };
    return { re: Infinity, im: 0 };
  }
  return cexp(mtoc2_cmul(b, clog(a)));
}

// Unary math.
export function mtoc2_csqrt(z) {
  // Pure-real input: compute directly. Smith's formula below suffers
  // catastrophic cancellation/overflow when z.im === 0 — e.g.
  // sqrt(-Inf) → sqrt((Inf + -Inf)/2) = sqrt(NaN) = NaN, and
  // sqrt(-1e308) → sqrt((1e308 - -1e308)/2) overflows to Inf instead of
  // 1e154. Mirror the interpreter's complexSqrt (math.ts) and C's libm
  // csqrt, both of which special-case the real axis.
  if (z.im === 0) {
    if (z.re >= 0) return { re: Math.sqrt(z.re), im: 0 };
    return { re: 0, im: Math.sqrt(-z.re) };
  }
  // Smith's stable formula for hypot * sign.
  const r = Math.hypot(z.re, z.im);
  if (r === 0) return { re: 0, im: 0 };
  const re = Math.sqrt((r + z.re) / 2);
  const im = (z.im >= 0 ? 1 : -1) * Math.sqrt((r - z.re) / 2);
  return { re, im };
}
export function mtoc2_cexp(z) {
  return cexp(z);
}
export function mtoc2_clog(z) {
  return clog(z);
}
export function mtoc2_clog2(z) {
  const l = clog(z);
  const lg2 = Math.log(2);
  return { re: l.re / lg2, im: l.im / lg2 };
}
export function mtoc2_clog10(z) {
  const l = clog(z);
  const lg10 = Math.log(10);
  return { re: l.re / lg10, im: l.im / lg10 };
}
export function mtoc2_csin(z) {
  // Guard 0*Inf=NaN on the imaginary axis: for a pure-imaginary input
  // (sin(z.re)===0), the real lane is 0 and the imaginary lane is
  // ±sinh(im); cosh/sinh overflow to Inf for large |im|, and 0*Inf would
  // poison the result. C99 csin special-cases this — match it.
  const sr = Math.sin(z.re);
  const cr = Math.cos(z.re);
  return {
    re: sr === 0 ? 0 : sr * Math.cosh(z.im),
    im: cr === 0 ? 0 : cr * Math.sinh(z.im),
  };
}
export function mtoc2_ccos(z) {
  const cr = Math.cos(z.re);
  const sr = Math.sin(z.re);
  return {
    re: cr === 0 ? 0 : cr * Math.cosh(z.im),
    im: sr === 0 ? 0 : -sr * Math.sinh(z.im),
  };
}
export function mtoc2_ctan(z) {
  // Double-angle form, matching the interpreter (interpreter/builtins/
  // math.ts) and C99 ctan. Computing sin(z)/cos(z) with a |cos|^2
  // denominator loses the (tiny but nonzero) real part and overflows
  // for large imaginary parts — e.g. tan(1+200i) collapsed to 1i.
  const denom = Math.cos(2 * z.re) + Math.cosh(2 * z.im);
  return { re: Math.sin(2 * z.re) / denom, im: Math.sinh(2 * z.im) / denom };
}
export function mtoc2_catan(z) {
  // Real-valued input (im === 0): atan is a clean real function. The
  // complex formula leaves a spurious ±1e-17i residue (and NaN+NaNi once
  // 1±iz overflows, e.g. atan(1e300)). Take the real path.
  if (z.im === 0) return { re: Math.atan(z.re), im: 0 };
  // atan(z) = (i/2)·log((1 − iz)/(1 + iz)) — MATLAB's branch (atan(2i) =
  // −1.5708 + 0.5493i). The C runtime overrides its libc catan (which
  // uses the opposite +π/2 Annex-G branch) to match this.
  const iz = { re: -z.im, im: z.re }; // i*z
  const num = { re: 1 - iz.re, im: -iz.im }; // 1 - iz
  const denom = { re: 1 + iz.re, im: iz.im }; // 1 + iz
  const dd = denom.re * denom.re + denom.im * denom.im;
  const q = {
    re: (num.re * denom.re + num.im * denom.im) / dd,
    im: (num.im * denom.re - num.re * denom.im) / dd,
  };
  const l = clog(q);
  return { re: -l.im / 2, im: l.re / 2 };
}
// Hyperbolic sinh/cosh/tanh — byte-for-byte with the interpreter's
// complex formulas (interpreter/builtins/math.ts). Real-valued input
// (im === 0) falls out cleanly: cos(0)=1, sin(0)=0, so the imaginary
// lane is exactly 0.
export function mtoc2_csinh(z) {
  return {
    re: Math.sinh(z.re) * Math.cos(z.im),
    im: Math.cosh(z.re) * Math.sin(z.im),
  };
}
export function mtoc2_ccosh(z) {
  return {
    re: Math.cosh(z.re) * Math.cos(z.im),
    im: Math.sinh(z.re) * Math.sin(z.im),
  };
}
export function mtoc2_ctanh(z) {
  const denom = Math.cosh(2 * z.re) + Math.cos(2 * z.im);
  return { re: Math.sinh(2 * z.re) / denom, im: Math.sin(2 * z.im) / denom };
}
export function mtoc2_cfloor(z) {
  return { re: Math.floor(z.re), im: Math.floor(z.im) };
}
export function mtoc2_cceil(z) {
  return { re: Math.ceil(z.re), im: Math.ceil(z.im) };
}
export function mtoc2_cround(z) {
  // MATLAB rounds each component half-away-from-zero.
  const half = x => Math.sign(x) * Math.round(Math.abs(x));
  return { re: half(z.re), im: half(z.im) };
}
export function mtoc2_cfix(z) {
  return { re: Math.trunc(z.re), im: Math.trunc(z.im) };
}
export function mtoc2_csign(z) {
  if (z.re === 0 && z.im === 0) return { re: 0, im: 0 };
  const m = Math.hypot(z.re, z.im);
  return { re: z.re / m, im: z.im / m };
}

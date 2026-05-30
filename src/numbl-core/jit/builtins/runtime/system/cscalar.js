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
  return { re: Math.log(Math.hypot(z.re, z.im)), im: Math.atan2(z.im, z.re) };
}
function cexp(z) {
  const m = Math.exp(z.re);
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
  return {
    re: Math.sin(z.re) * Math.cosh(z.im),
    im: Math.cos(z.re) * Math.sinh(z.im),
  };
}
export function mtoc2_ccos(z) {
  return {
    re: Math.cos(z.re) * Math.cosh(z.im),
    im: -Math.sin(z.re) * Math.sinh(z.im),
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
  // catan(z) = (i/2) * (log(1 - iz) - log(1 + iz))
  // = (1/2) * Im[log((1+iz)/(1-iz))] / no — use the simpler real
  // recurrence: atan(z) = (-i/2) * log((i-z)/(i+z))
  const iz = { re: -z.im, im: z.re }; // i*z
  const num = { re: 1 - iz.re, im: -iz.im }; // 1 - iz
  const denom = { re: 1 + iz.re, im: iz.im }; // 1 + iz
  // (1-iz) / (1+iz)
  const dd = denom.re * denom.re + denom.im * denom.im;
  const q = {
    re: (num.re * denom.re + num.im * denom.im) / dd,
    im: (num.im * denom.re - num.re * denom.im) / dd,
  };
  const l = clog(q);
  // (i/2) * l
  return { re: -l.im / 2, im: l.re / 2 };
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

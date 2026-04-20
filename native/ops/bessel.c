/**
 * Bessel tensor ops for real arguments.
 *
 * C port of src/numbl-core/helpers/bessel.ts.  Matches that implementation
 * bit-for-bit so the TS and native paths return identical results.
 *
 * Integer orders 0,1 use Cephes rational polynomial approximations.
 * Higher integer orders use forward recurrence (stable for x >= n).
 * Non-integer orders use power series (small x) or Hankel asymptotic (large x).
 *
 * Cephes coefficients: Copyright 1984-2000 by Stephen L. Moshier, BSD license.
 */

#include "numbl_ops.h"

#include <math.h>
#include <stddef.h>

static const double PI_C = 3.14159265358979323846;

/* ── Polynomial evaluation (Horner, descending powers) ─────────────────── */

static inline double polyeval(const double* c, int len, double x) {
  double r = c[0];
  for (int i = 1; i < len; i++) r = r * x + c[i];
  return r;
}

/* ── Lanczos gamma (matches helpers/bessel.ts lanczosGamma) ─────────────── */

static double lgamma_lanczos(double x) {
  if (x < 0.5) {
    return PI_C / (sin(PI_C * x) * lgamma_lanczos(1.0 - x));
  }
  static const double coef[9] = {
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  };
  x -= 1.0;
  double a = coef[0];
  for (int i = 1; i < 9; i++) a += coef[i] / (x + i);
  const double g = 7.0;
  double t = x + g + 0.5;
  return sqrt(2.0 * PI_C) * pow(t, x + 0.5) * exp(-t) * a;
}

/* ── Cephes J0 ──────────────────────────────────────────────────────────── */

static double j0_large(double x);
static double j1_large(double x);
static double y0_large(double x);
static double y1_large(double x);

static double cephes_j0(double x) {
  if (x < 0) x = -x;
  if (x <= 5.0) {
    double z = x * x;
    const double DR1 = 5.78318596294678452118;
    const double DR2 = 30.4712623436620863991;
    static const double RP[4] = {
      -4.79443220978201773821e9, 1.95617491946556577543e12,
      -2.49248344360967716204e14, 9.70862251047306323952e15,
    };
    static const double RQ[9] = {
      1.0, 4.99563147152651017219e2, 1.73785401676374683123e5,
      4.84409658339962045305e7, 1.11855537045356834862e10,
      2.11277520115489217587e12, 3.10518229857422583814e14,
      3.18121955943204943306e16, 1.71086294081043136091e18,
    };
    double p = polyeval(RP, 4, z);
    double q = polyeval(RQ, 9, z);
    return (z - DR1) * (z - DR2) * (p / q);
  }
  return j0_large(x);
}

static double j0_large(double x) {
  static const double PP[7] = {
    7.96936729297347051624e-4, 8.28352392107440799803e-2,
    1.23953371646414299388, 5.4472500305876877509, 8.74716500199817011941,
    5.30324038235394892183, 9.99999999999999997821e-1,
  };
  static const double PQ[7] = {
    9.24408810558863637013e-4, 8.56288474354474431428e-2,
    1.25352743901058953537, 5.47097740330417105182, 8.76190883237069594232,
    5.30605288235394617618, 1.00000000000000000218,
  };
  static const double QP[8] = {
    -1.13663838898469149931e-2, -1.28252718670509318512,
    -1.95539544257735972385e1, -9.32060152123768231369e1,
    -1.77681167980488050595e2, -1.47077505154951170175e2,
    -5.1410532676659933022e1, -6.05014350600728481186,
  };
  static const double QQ[8] = {
    1.0, 6.43178256118178023184e1, 8.56430025976980587198e2,
    3.88240183605401609683e3, 7.24046774195652478189e3,
    5.93072701187316984827e3, 2.06209331660327847417e3,
    2.42005740240291393179e2,
  };
  double w = 5.0 / x;
  double z = w * w;
  double p = polyeval(PP, 7, z) / polyeval(PQ, 7, z);
  double q = polyeval(QP, 8, z) / polyeval(QQ, 8, z);
  double xn = x - PI_C / 4.0;
  return sqrt(2.0 / (PI_C * x)) * (p * cos(xn) - w * q * sin(xn));
}

/* ── Cephes J1 ──────────────────────────────────────────────────────────── */

static double cephes_j1(double x) {
  double sign = x < 0 ? -1.0 : 1.0;
  if (x < 0) x = -x;
  if (x <= 5.0) {
    double z = x * x;
    const double Z1 = 1.46819706421238932572e1;
    const double Z2 = 4.92184563216946036703e1;
    static const double RP[4] = {
      -8.99971225705559398224e8, 4.52228297998194034323e11,
      -7.27494245221818276015e13, 3.68295732863852883286e15,
    };
    static const double RQ[9] = {
      1.0, 6.20836478118054335476e2, 2.56987256757748830383e5,
      8.35146791431949253037e7, 2.21511595479792499675e10,
      4.74914122079991414898e12, 7.84369607876235854894e14,
      8.95222336184627338078e16, 5.32278620332680085395e18,
    };
    double p = polyeval(RP, 4, z);
    double q = polyeval(RQ, 9, z);
    return sign * x * (z - Z1) * (z - Z2) * (p / q);
  }
  return sign * j1_large(x);
}

static double j1_large(double x) {
  static const double PP[7] = {
    7.62125616208173112003e-4, 7.31397056940917570436e-2,
    1.12719608129684925192, 5.11207951146807644818, 8.42404590141772420927,
    5.21451598682361821619, 1.00000000000000000254,
  };
  static const double PQ[7] = {
    5.71323128072548699714e-4, 6.88455908754495404082e-2,
    1.10514232634061696926, 5.07386386128601488557, 8.39985554327604159757,
    5.20982848682361821619, 9.99999999999999997461e-1,
  };
  static const double QP[8] = {
    5.10862594750176621635e-2, 4.9821387295123344942, 7.58238284132545283818e1,
    3.667796093601507778e2, 7.10856304998926107277e2, 5.97489612400613639965e2,
    2.11688757100572135698e2, 2.52070205858023719784e1,
  };
  static const double QQ[8] = {
    1.0, 7.42373277035675149943e1, 1.05644886038262816351e3,
    4.98641058337653607651e3, 9.56231892404756170795e3, 7.9970416044735068365e3,
    2.826192785176390966e3, 3.36093607810698293419e2,
  };
  double w = 5.0 / x;
  double z = w * w;
  double p = polyeval(PP, 7, z) / polyeval(PQ, 7, z);
  double q = polyeval(QP, 8, z) / polyeval(QQ, 8, z);
  double xn = x - 3.0 * PI_C / 4.0;
  return sqrt(2.0 / (PI_C * x)) * (p * cos(xn) - w * q * sin(xn));
}

/* ── Cephes Y0 ──────────────────────────────────────────────────────────── */

static double cephes_y0(double x) {
  if (x <= 5.0) {
    static const double YP[8] = {
      1.55924367855235737965e4, -1.46639295903971606143e7,
      5.43526477051876500413e9, -9.82136065717911466409e11,
      8.75906394395366999549e13, -3.46628303384729719441e15,
      4.42733268572569800351e16, -1.84950800436986690637e16,
    };
    static const double YQ[8] = {
      1.0, 1.04128353664259848412e3, 6.26107330137134956842e5,
      2.68919633393814121987e8, 8.64002487103935000337e10,
      2.02979612750105546709e13, 3.17157752842975028269e15,
      2.50596256172653059228e17,
    };
    double z = x * x;
    double p = polyeval(YP, 8, z);
    double q = polyeval(YQ, 8, z);
    return p / q + (2.0 / PI_C) * log(x) * cephes_j0(x);
  }
  return y0_large(x);
}

static double y0_large(double x) {
  /* Same P/Q polynomials as J0 large; sin/cos swapped. */
  static const double PP[7] = {
    7.96936729297347051624e-4, 8.28352392107440799803e-2,
    1.23953371646414299388, 5.4472500305876877509, 8.74716500199817011941,
    5.30324038235394892183, 9.99999999999999997821e-1,
  };
  static const double PQ[7] = {
    9.24408810558863637013e-4, 8.56288474354474431428e-2,
    1.25352743901058953537, 5.47097740330417105182, 8.76190883237069594232,
    5.30605288235394617618, 1.00000000000000000218,
  };
  static const double QP[8] = {
    -1.13663838898469149931e-2, -1.28252718670509318512,
    -1.95539544257735972385e1, -9.32060152123768231369e1,
    -1.77681167980488050595e2, -1.47077505154951170175e2,
    -5.1410532676659933022e1, -6.05014350600728481186,
  };
  static const double QQ[8] = {
    1.0, 6.43178256118178023184e1, 8.56430025976980587198e2,
    3.88240183605401609683e3, 7.24046774195652478189e3,
    5.93072701187316984827e3, 2.06209331660327847417e3,
    2.42005740240291393179e2,
  };
  double w = 5.0 / x;
  double z = w * w;
  double p = polyeval(PP, 7, z) / polyeval(PQ, 7, z);
  double q = polyeval(QP, 8, z) / polyeval(QQ, 8, z);
  double xn = x - PI_C / 4.0;
  return sqrt(2.0 / (PI_C * x)) * (p * sin(xn) + w * q * cos(xn));
}

/* ── Cephes Y1 ──────────────────────────────────────────────────────────── */

static double cephes_y1(double x) {
  if (x <= 5.0) {
    static const double YP[6] = {
      1.2632047479017802644e9, -6.47355876379160291031e11,
      1.14509511541823727583e14, -8.12770255501325109621e15,
      2.02439475713594898196e17, -7.78877196265950026825e17,
    };
    static const double YQ[9] = {
      1.0, 5.94301592346128195359e2, 2.35564092943068577943e5,
      7.3481194445972170566e7, 1.87601316108706159478e10,
      3.88231277496238566008e12, 6.20557727146953693363e14,
      6.87141087355300489866e16, 3.97270608116560655612e18,
    };
    double z = x * x;
    double p = polyeval(YP, 6, z);
    double q = polyeval(YQ, 9, z);
    return x * (p / q) + (2.0 / PI_C) * (cephes_j1(x) * log(x) - 1.0 / x);
  }
  return y1_large(x);
}

static double y1_large(double x) {
  static const double PP[7] = {
    7.62125616208173112003e-4, 7.31397056940917570436e-2,
    1.12719608129684925192, 5.11207951146807644818, 8.42404590141772420927,
    5.21451598682361821619, 1.00000000000000000254,
  };
  static const double PQ[7] = {
    5.71323128072548699714e-4, 6.88455908754495404082e-2,
    1.10514232634061696926, 5.07386386128601488557, 8.39985554327604159757,
    5.20982848682361821619, 9.99999999999999997461e-1,
  };
  static const double QP[8] = {
    5.10862594750176621635e-2, 4.9821387295123344942, 7.58238284132545283818e1,
    3.667796093601507778e2, 7.10856304998926107277e2, 5.97489612400613639965e2,
    2.11688757100572135698e2, 2.52070205858023719784e1,
  };
  static const double QQ[8] = {
    1.0, 7.42373277035675149943e1, 1.05644886038262816351e3,
    4.98641058337653607651e3, 9.56231892404756170795e3, 7.9970416044735068365e3,
    2.826192785176390966e3, 3.36093607810698293419e2,
  };
  double w = 5.0 / x;
  double z = w * w;
  double p = polyeval(PP, 7, z) / polyeval(PQ, 7, z);
  double q = polyeval(QP, 8, z) / polyeval(QQ, 8, z);
  double xn = x - 3.0 * PI_C / 4.0;
  return sqrt(2.0 / (PI_C * x)) * (p * sin(xn) + w * q * cos(xn));
}

/* ── Fallbacks for non-integer orders / higher orders ──────────────────── */

static double besselj_series(double nu, double x) {
  double halfX = x * 0.5;
  double term = pow(halfX, nu) / lgamma_lanczos(nu + 1.0);
  double sum = term;
  double x2over4 = -halfX * halfX;
  for (int k = 1; k <= 300; k++) {
    term *= x2over4 / ((double)k * (k + nu));
    sum += term;
    if (fabs(term) < fabs(sum) * 1e-16) break;
  }
  return sum;
}

static double hankel_besselj(double nu, double x) {
  double mu = 4.0 * nu * nu;
  double chi = x - (nu * 0.5 + 0.25) * PI_C;
  double P = 1.0, Q = 0.0, termP = 1.0, termQ = 1.0;
  for (int k = 0; k < 30; k++) {
    if (k > 0) {
      double num = -(mu - (4.0 * k - 3.0) * (4.0 * k - 3.0)) *
                   (mu - (4.0 * k - 1.0) * (4.0 * k - 1.0));
      termP *= num / ((2.0 * k - 1.0) * (2.0 * k) * 64.0 * x * x);
      P += termP;
    }
    if (k == 0) {
      termQ = (mu - 1.0) / (8.0 * x);
      Q = termQ;
    } else {
      double num = -(mu - (4.0 * k - 1.0) * (4.0 * k - 1.0)) *
                   (mu - (4.0 * k + 1.0) * (4.0 * k + 1.0));
      termQ *= num / (2.0 * k * (2.0 * k + 1.0) * 64.0 * x * x);
      Q += termQ;
    }
    if (fabs(termP) + fabs(termQ) < 1e-16) break;
  }
  return sqrt(2.0 / (PI_C * x)) * (P * cos(chi) - Q * sin(chi));
}

/* ── Public scalar entry points ────────────────────────────────────────── */

static int is_int(double x) {
  return x == floor(x);
}

static double scalar_besselj(double nu, double x) {
  if (x == 0.0) return nu == 0.0 ? 1.0 : 0.0;
  if (x < 0.0 && !is_int(nu)) return NAN;
  if (x < 0.0) {
    double n = floor(nu + 0.5); /* round */
    double s = ((long long)n & 1LL) == 0 ? 1.0 : -1.0;
    return s * scalar_besselj(nu, -x);
  }
  if (nu < 0.0 && is_int(nu)) {
    double n = floor(-nu + 0.5);
    double s = ((long long)n & 1LL) == 0 ? 1.0 : -1.0;
    return s * scalar_besselj(-nu, x);
  }
  if (is_int(nu) && nu >= 0.0) {
    int n = (int)floor(nu + 0.5);
    if (n == 0) return cephes_j0(x);
    if (n == 1) return cephes_j1(x);
    if (x < (double)n) return besselj_series((double)n, x);
    double jm1 = cephes_j0(x);
    double j = cephes_j1(x);
    for (int k = 1; k < n; k++) {
      double jnext = ((2.0 * k) / x) * j - jm1;
      jm1 = j;
      j = jnext;
    }
    return j;
  }
  if (x <= 25.0 + fabs(nu) * 0.5) return besselj_series(nu, x);
  return hankel_besselj(nu, x);
}

static double bessely_integer(int n, double x) {
  if (n < 0) {
    double s = ((-n) & 1) == 0 ? 1.0 : -1.0;
    return s * bessely_integer(-n, x);
  }
  if (n == 0) return cephes_y0(x);
  if (n == 1) return cephes_y1(x);
  double ym1 = cephes_y0(x);
  double y = cephes_y1(x);
  for (int k = 1; k < n; k++) {
    double ynext = ((2.0 * k) / x) * y - ym1;
    ym1 = y;
    y = ynext;
  }
  return y;
}

static double scalar_bessely(double nu, double x) {
  if (x <= 0.0) return NAN;
  if (is_int(nu)) {
    int n = (int)floor(nu + (nu >= 0 ? 0.5 : -0.5));
    return bessely_integer(n, x);
  }
  double sinPi = sin(nu * PI_C);
  return (scalar_besselj(nu, x) * cos(nu * PI_C) - scalar_besselj(-nu, x)) / sinPi;
}

static double scalar_besseli(double nu, double x) {
  if (x == 0.0) return nu == 0.0 ? 1.0 : 0.0;
  if (x < 0.0 && !is_int(nu)) return NAN;
  if (x < 0.0) {
    double n = floor(nu + 0.5);
    double s = ((long long)n & 1LL) == 0 ? 1.0 : -1.0;
    return s * scalar_besseli(nu, -x);
  }
  if (nu < 0.0 && is_int(nu)) {
    return scalar_besseli(-nu, x); /* I_{-n} = I_n for integer n */
  }
  double halfX = x * 0.5;
  double term = pow(halfX, nu) / lgamma_lanczos(nu + 1.0);
  double sum = term;
  double x2over4 = halfX * halfX;
  for (int k = 1; k <= 300; k++) {
    term *= x2over4 / ((double)k * (k + nu));
    sum += term;
    if (fabs(term) < fabs(sum) * 1e-16) break;
  }
  return sum;
}

static double besselk0(double x) {
  const double euler = 0.5772156649015329;
  double i0 = scalar_besseli(0.0, x);
  double logTerm = -(log(x * 0.5) + euler) * i0;
  double halfX = x * 0.5;
  double x2over4 = halfX * halfX;
  double term = 1.0;
  double hk = 0.0;
  double sum = 0.0;
  for (int k = 1; k <= 300; k++) {
    hk += 1.0 / k;
    term *= x2over4 / ((double)k * k);
    sum += hk * term;
    if (fabs(term * hk) < fabs(sum) * 1e-16) break;
  }
  return logTerm + sum;
}

static double besselk1(double x) {
  const double euler = 0.5772156649015329;
  double halfX = x * 0.5;
  double x2over4 = halfX * halfX;
  double term = 1.0;
  double psi1 = -euler;
  double psi2 = -euler + 1.0;
  double S1 = term;
  double S2 = (psi1 + psi2) * term;
  for (int k = 1; k <= 300; k++) {
    term *= x2over4 / ((double)k * (k + 1));
    psi1 += 1.0 / k;
    psi2 += 1.0 / (k + 1.0);
    S1 += term;
    S2 += (psi1 + psi2) * term;
    if (fabs(term) < fabs(S1) * 1e-16) break;
  }
  return 1.0 / x + halfX * log(halfX) * S1 - (halfX * 0.5) * S2;
}

static double besselk_integer(int n, double x) {
  if (n == 0) return besselk0(x);
  if (n == 1) return besselk1(x);
  double km1 = besselk0(x);
  double k = besselk1(x);
  for (int i = 1; i < n; i++) {
    double knext = ((2.0 * i) / x) * k + km1;
    km1 = k;
    k = knext;
  }
  return k;
}

static double scalar_besselk(double nu, double x) {
  if (x <= 0.0) return NAN;
  if (nu < 0.0) nu = -nu;
  if (is_int(nu)) {
    int n = (int)floor(nu + 0.5);
    return besselk_integer(n, x);
  }
  double sinPi = sin(nu * PI_C);
  return (PI_C * 0.5) * (scalar_besseli(-nu, x) - scalar_besseli(nu, x)) / sinPi;
}

/* ── Public tensor entry points ────────────────────────────────────────── */

/* Scale factors match the existing TS code in special-math.ts:
 *   J, Y, I: exp(-|z|)
 *   K: exp(z)
 * (The J/Y choice mirrors the current TS builtin; MATLAB's correct factor
 * for real z is 1, but we preserve existing numerical behavior.)
 */

int numbl_bessel_real(int op, double nu, size_t n,
                      const double* z, int scale, double* out) {
  if (!z || !out) return NUMBL_ERR_NULL_PTR;
  switch (op) {
    case NUMBL_BESSEL_J: {
      if (scale) {
        for (size_t i = 0; i < n; i++) {
          double zi = z[i];
          out[i] = scalar_besselj(nu, zi) * exp(-fabs(zi));
        }
      } else {
        for (size_t i = 0; i < n; i++) out[i] = scalar_besselj(nu, z[i]);
      }
      return NUMBL_OK;
    }
    case NUMBL_BESSEL_Y: {
      if (scale) {
        for (size_t i = 0; i < n; i++) {
          double zi = z[i];
          out[i] = scalar_bessely(nu, zi) * exp(-fabs(zi));
        }
      } else {
        for (size_t i = 0; i < n; i++) out[i] = scalar_bessely(nu, z[i]);
      }
      return NUMBL_OK;
    }
    case NUMBL_BESSEL_I: {
      if (scale) {
        for (size_t i = 0; i < n; i++) {
          double zi = z[i];
          out[i] = scalar_besseli(nu, zi) * exp(-fabs(zi));
        }
      } else {
        for (size_t i = 0; i < n; i++) out[i] = scalar_besseli(nu, z[i]);
      }
      return NUMBL_OK;
    }
    case NUMBL_BESSEL_K: {
      if (scale) {
        for (size_t i = 0; i < n; i++) {
          double zi = z[i];
          out[i] = scalar_besselk(nu, zi) * exp(zi);
        }
      } else {
        for (size_t i = 0; i < n; i++) out[i] = scalar_besselk(nu, z[i]);
      }
      return NUMBL_OK;
    }
    default:
      return NUMBL_ERR_BAD_OP;
  }
}

/* besselh(nu, K, z) with real z returns J ± i*Y (K=1: +, K=2: -).
 * Scaled variant multiplies by exp(∓i*z).
 */
int numbl_bessel_h(int k_kind, double nu, size_t n,
                   const double* z, int scale,
                   double* out_re, double* out_im) {
  if (!z || !out_re || !out_im) return NUMBL_ERR_NULL_PTR;
  if (k_kind != 1 && k_kind != 2) return NUMBL_ERR_BAD_OP;
  double ysign = k_kind == 1 ? 1.0 : -1.0;
  if (!scale) {
    for (size_t i = 0; i < n; i++) {
      double zi = z[i];
      out_re[i] = scalar_besselj(nu, zi);
      out_im[i] = ysign * scalar_bessely(nu, zi);
    }
  } else {
    /* K=1 scaled: multiply by exp(-i*z) = cos(z) - i*sin(z).
     * K=2 scaled: multiply by exp(+i*z) = cos(z) + i*sin(z).
     */
    double ssign = k_kind == 1 ? -1.0 : 1.0;
    for (size_t i = 0; i < n; i++) {
      double zi = z[i];
      double J = scalar_besselj(nu, zi);
      double Y = ysign * scalar_bessely(nu, zi);
      double c = cos(zi);
      double s = ssign * sin(zi);
      /* (J + iY) * (c + is) = (J*c - Y*s) + i*(J*s + Y*c). */
      out_re[i] = J * c - Y * s;
      out_im[i] = J * s + Y * c;
    }
  }
  return NUMBL_OK;
}

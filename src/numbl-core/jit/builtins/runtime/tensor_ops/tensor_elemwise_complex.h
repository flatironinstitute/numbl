/* @snippet name=jit_tensor_elemwise_complex deps=jit_tensor_t,jit_alloc,jit_tensor_alloc_nd_complex,jit_cscalar,jit_cdiv */
/* JIT runtime helpers: elementwise binary + unary ops on complex
 * tensors. Sibling of `tensor_elemwise_real.h` — same `_tt` /
 * `_ts` / `_st` / `_bcast_tt` naming, the kernel walks both lanes,
 * and per-element math routes through `cscalar.h`'s helpers (no bare
 * `<complex.h>` operators).
 *
 * Each helper allocates a freshly-owned complex tensor via
 * `jit_tensor_alloc_nd_complex` (both lanes) and lane-writes via
 * `jit_creal` / `jit_cimag` on the per-element complex result.
 *
 * Mixed real/complex operands at the elemwise binary site are
 * routed through these same helpers by emit — the lowering layer
 * promotes a real scalar to complex via `jit_cmake(re, 0.0)`
 * before passing it in; a real tensor would need a separate
 * "promote to complex" step that Phase 3 doesn't ship.
 */

#include <stdlib.h>

/* Both _tt and _bcast_tt accept either operand with `imag == NULL`
 * (a static-real tensor) so the emit layer can hand a real tensor
 * straight into a complex op without an explicit promote step. The
 * NULL-lane branches read 0.0 instead. The JS siblings use `null`
 * as the equivalent sentinel. */
/* Per-element ops write both result lanes from the operands' lanes.
 * `ar` / `br` say the operand acts as real: no imag lane, or an imag
 * lane that is all zero (MATLAB treats such a complex array as real).
 * Only times / rdivide look at them (MIXED = 1); plus / minus pass
 * MIXED = 0 so the flags are constant 0.
 *
 * Mixed times / rdivide follow MATLAB (as the interpreter's
 * `mixedRealComplex` in helpers/arithmetic.ts): a real operand scales
 * each part of a complex one without promotion, so a result part stays
 * 0 where the complex operand's part is 0 (NaN or Inf in the real
 * operand would otherwise give NaN*0 = NaN). The lanes are written
 * directly rather than through `jit_cmake`, whose `re + im*I` turns
 * an Inf imag part into a NaN real part. */
static int jit_celem_acts_real(jit_tensor_t t) {
  if (t.imag == NULL) return 1;
  long n = 1;
  for (int i = 0; i < t.ndim; i++) n *= t.dims[i];
  for (long i = 0; i < n; i++) {
    if (t.imag[i] != 0.0) return 0;
  }
  return 1;
}

static inline void jit_celem_plus(double are, double aim, int ar, double bre,
                                    double bim, int br, double *re, double *im) {
  (void)ar;
  (void)br;
  double _Complex v = jit_cadd(jit_cmake(are, aim), jit_cmake(bre, bim));
  *re = jit_creal(v);
  *im = jit_cimag(v);
}

static inline void jit_celem_minus(double are, double aim, int ar, double bre,
                                     double bim, int br, double *re, double *im) {
  (void)ar;
  (void)br;
  double _Complex v = jit_csub(jit_cmake(are, aim), jit_cmake(bre, bim));
  *re = jit_creal(v);
  *im = jit_cimag(v);
}

static inline void jit_celem_times(double are, double aim, int ar, double bre,
                                     double bim, int br, double *re, double *im) {
  if (ar && br) {
    *re = are * bre;
    *im = 0.0;
  } else if (br) {
    *re = are == 0.0 ? 0.0 : are * bre;
    *im = aim == 0.0 ? 0.0 : aim * bre;
  } else if (ar) {
    *re = bre == 0.0 ? 0.0 : bre * are;
    *im = bim == 0.0 ? 0.0 : bim * are;
  } else {
    double _Complex v = jit_cmul(jit_cmake(are, aim), jit_cmake(bre, bim));
    *re = jit_creal(v);
    *im = jit_cimag(v);
  }
}

static inline void jit_celem_rdivide(double are, double aim, int ar,
                                       double bre, double bim, int br,
                                       double *re, double *im) {
  if (ar && br) {
    *re = are / bre;
    *im = 0.0;
  } else if (br) {
    *re = are == 0.0 ? 0.0 : are / bre;
    *im = aim == 0.0 ? 0.0 : aim / bre;
  } else if (ar && bim == 0.0) {
    *re = are / bre;
    *im = 0.0;
  } else {
    double _Complex v = jit_cdiv(jit_cmake(are, ar ? 0.0 : aim),
                                   jit_cmake(bre, bim));
    *re = jit_creal(v);
    *im = jit_cimag(v);
  }
}

#define JIT_DEFINE_CELEMWISE_TT(name, OP, MIXED)                               \
  static jit_tensor_t name(jit_tensor_t a, jit_tensor_t b) {                   \
    long n = 1;                                                                \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                           \
    int ar = (MIXED) && jit_celem_acts_real(a);                                \
    int br = (MIXED) && jit_celem_acts_real(b);                                \
    jit_tensor_t r = jit_tensor_alloc_nd_complex(a.ndim, a.dims);              \
    for (long i = 0; i < n; i++) {                                             \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      double bim = (b.imag != NULL) ? b.imag[i] : 0.0;                         \
      OP(a.real[i], aim, ar, b.real[i], bim, br, &r.real[i], &r.imag[i]);      \
    }                                                                          \
    return r;                                                                  \
  }

#define JIT_DEFINE_CELEMWISE_TS(name, OP, MIXED)                               \
  static jit_tensor_t name(jit_tensor_t a, double _Complex s) {                \
    long n = 1;                                                                \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                           \
    double sre = jit_creal(s), sim = jit_cimag(s);                             \
    int ar = (MIXED) && jit_celem_acts_real(a);                                \
    int sr = (MIXED) && sim == 0.0;                                            \
    jit_tensor_t r = jit_tensor_alloc_nd_complex(a.ndim, a.dims);              \
    for (long i = 0; i < n; i++) {                                             \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      OP(a.real[i], aim, ar, sre, sim, sr, &r.real[i], &r.imag[i]);            \
    }                                                                          \
    return r;                                                                  \
  }

#define JIT_DEFINE_CELEMWISE_ST(name, OP, MIXED)                               \
  static jit_tensor_t name(double _Complex s, jit_tensor_t b) {                \
    long n = 1;                                                                \
    for (int i = 0; i < b.ndim; i++) n *= b.dims[i];                           \
    double sre = jit_creal(s), sim = jit_cimag(s);                             \
    int sr = (MIXED) && sim == 0.0;                                            \
    int br = (MIXED) && jit_celem_acts_real(b);                                \
    jit_tensor_t r = jit_tensor_alloc_nd_complex(b.ndim, b.dims);              \
    for (long i = 0; i < n; i++) {                                             \
      double bim = (b.imag != NULL) ? b.imag[i] : 0.0;                         \
      OP(sre, sim, sr, b.real[i], bim, br, &r.real[i], &r.imag[i]);            \
    }                                                                          \
    return r;                                                                  \
  }

#define JIT_DEFINE_CELEMWISE_BCAST_TT(name, OP, MIXED)                         \
  static jit_tensor_t name(jit_tensor_t a, jit_tensor_t b) {                   \
    int rnd = a.ndim > b.ndim ? a.ndim : b.ndim;                               \
    long adim[JIT_MAX_NDIM], bdim[JIT_MAX_NDIM];                               \
    long rdim[JIT_MAX_NDIM];                                                   \
    long astride[JIT_MAX_NDIM], bstride[JIT_MAX_NDIM];                         \
    long aacc = 1, bacc = 1;                                                   \
    long n = 1;                                                                \
    for (int i = 0; i < rnd; i++) {                                            \
      adim[i] = (i < a.ndim) ? a.dims[i] : 1;                                  \
      bdim[i] = (i < b.ndim) ? b.dims[i] : 1;                                  \
      rdim[i] = (adim[i] == 1) ? bdim[i] : adim[i];                            \
      astride[i] = (adim[i] == 1) ? 0 : aacc;                                  \
      bstride[i] = (bdim[i] == 1) ? 0 : bacc;                                  \
      aacc *= adim[i];                                                         \
      bacc *= bdim[i];                                                         \
      n *= rdim[i];                                                            \
    }                                                                          \
    int ar = (MIXED) && jit_celem_acts_real(a);                                \
    int br = (MIXED) && jit_celem_acts_real(b);                                \
    jit_tensor_t r = jit_tensor_alloc_nd_complex(rnd, rdim);                   \
    long ix[JIT_MAX_NDIM] = {0};                                               \
    for (long k = 0; k < n; k++) {                                             \
      long ai = 0, bi = 0;                                                     \
      for (int i = 0; i < rnd; i++) {                                          \
        ai += ix[i] * astride[i];                                              \
        bi += ix[i] * bstride[i];                                              \
      }                                                                        \
      double aim = (a.imag != NULL) ? a.imag[ai] : 0.0;                        \
      double bim = (b.imag != NULL) ? b.imag[bi] : 0.0;                        \
      OP(a.real[ai], aim, ar, b.real[bi], bim, br, &r.real[k], &r.imag[k]);    \
      for (int i = 0; i < rnd; i++) {                                          \
        ix[i]++;                                                               \
        if (ix[i] < rdim[i]) break;                                            \
        ix[i] = 0;                                                             \
      }                                                                        \
    }                                                                          \
    return r;                                                                  \
  }

JIT_DEFINE_CELEMWISE_TT(jit_tensor_plus_complex_tt, jit_celem_plus, 0)
JIT_DEFINE_CELEMWISE_TT(jit_tensor_minus_complex_tt, jit_celem_minus, 0)
JIT_DEFINE_CELEMWISE_TT(jit_tensor_times_complex_tt, jit_celem_times, 1)
JIT_DEFINE_CELEMWISE_TT(jit_tensor_rdivide_complex_tt, jit_celem_rdivide, 1)

JIT_DEFINE_CELEMWISE_TS(jit_tensor_plus_complex_ts, jit_celem_plus, 0)
JIT_DEFINE_CELEMWISE_TS(jit_tensor_minus_complex_ts, jit_celem_minus, 0)
JIT_DEFINE_CELEMWISE_TS(jit_tensor_times_complex_ts, jit_celem_times, 1)
JIT_DEFINE_CELEMWISE_TS(jit_tensor_rdivide_complex_ts, jit_celem_rdivide, 1)

JIT_DEFINE_CELEMWISE_ST(jit_tensor_minus_complex_st, jit_celem_minus, 0)
JIT_DEFINE_CELEMWISE_ST(jit_tensor_rdivide_complex_st, jit_celem_rdivide, 1)

JIT_DEFINE_CELEMWISE_BCAST_TT(jit_tensor_plus_complex_bcast_tt, jit_celem_plus, 0)
JIT_DEFINE_CELEMWISE_BCAST_TT(jit_tensor_minus_complex_bcast_tt, jit_celem_minus, 0)
JIT_DEFINE_CELEMWISE_BCAST_TT(jit_tensor_times_complex_bcast_tt, jit_celem_times, 1)
JIT_DEFINE_CELEMWISE_BCAST_TT(jit_tensor_rdivide_complex_bcast_tt, jit_celem_rdivide, 1)

/* Unary minus on a complex tensor — per-element negate via cneg. */
static jit_tensor_t jit_tensor_uminus_complex(jit_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  jit_tensor_t r = jit_tensor_alloc_nd_complex(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    double aim = (a.imag != NULL) ? a.imag[i] : 0.0;
    double _Complex av = jit_cmake(a.real[i], aim);
    double _Complex rv = jit_cneg(av);
    r.real[i] = jit_creal(rv);
    r.imag[i] = jit_cimag(rv);
  }
  return r;
}

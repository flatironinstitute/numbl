/* mtoc2 runtime helpers: elementwise binary + unary ops on complex
 * tensors. Sibling of `tensor_elemwise_real.h` — same `_tt` /
 * `_ts` / `_st` / `_bcast_tt` naming, the kernel walks both lanes,
 * and per-element math routes through `cscalar.h`'s helpers so the
 * c2js backend can translate the bodies straight (no bare
 * `<complex.h>` operators).
 *
 * Each helper allocates a freshly-owned complex tensor via
 * `mtoc2_tensor_alloc_nd_complex` (both lanes) and lane-writes via
 * `mtoc2_creal` / `mtoc2_cimag` on the per-element complex result.
 *
 * Mixed real/complex operands at the elemwise binary site are
 * routed through these same helpers by emit — the lowering layer
 * promotes a real scalar to complex via `mtoc2_cmake(re, 0.0)`
 * before passing it in; a real tensor would need a separate
 * "promote to complex" step that Phase 3 doesn't ship.
 */

#include <stdlib.h>

/* Both _tt and _bcast_tt accept either operand with `imag == NULL`
 * (a static-real tensor) so the emit layer can hand a real tensor
 * straight into a complex op without an explicit promote step. The
 * NULL-lane branches read 0.0 instead. The c2js backend uses `null`
 * as the equivalent sentinel; the JS-side helpers see `null` and the
 * same `cur.imag != null ? cur.imag[i] : 0.0` branch falls out. */
#define MTOC2_DEFINE_CELEMWISE_TT(name, FN)                                    \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, mtoc2_tensor_t b) {             \
    long n = 1;                                                                \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                           \
    mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);          \
    for (long i = 0; i < n; i++) {                                             \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      double bim = (b.imag != NULL) ? b.imag[i] : 0.0;                         \
      double _Complex av = mtoc2_cmake(a.real[i], aim);                        \
      double _Complex bv = mtoc2_cmake(b.real[i], bim);                        \
      double _Complex rv = FN(av, bv);                                         \
      r.real[i] = mtoc2_creal(rv);                                             \
      r.imag[i] = mtoc2_cimag(rv);                                             \
    }                                                                          \
    return r;                                                                  \
  }

#define MTOC2_DEFINE_CELEMWISE_TS(name, FN)                                    \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, double _Complex s) {            \
    long n = 1;                                                                \
    for (int i = 0; i < a.ndim; i++) n *= a.dims[i];                           \
    mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);          \
    for (long i = 0; i < n; i++) {                                             \
      double aim = (a.imag != NULL) ? a.imag[i] : 0.0;                         \
      double _Complex av = mtoc2_cmake(a.real[i], aim);                        \
      double _Complex rv = FN(av, s);                                          \
      r.real[i] = mtoc2_creal(rv);                                             \
      r.imag[i] = mtoc2_cimag(rv);                                             \
    }                                                                          \
    return r;                                                                  \
  }

#define MTOC2_DEFINE_CELEMWISE_ST(name, FN)                                    \
  static mtoc2_tensor_t name(double _Complex s, mtoc2_tensor_t b) {            \
    long n = 1;                                                                \
    for (int i = 0; i < b.ndim; i++) n *= b.dims[i];                           \
    mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(b.ndim, b.dims);          \
    for (long i = 0; i < n; i++) {                                             \
      double bim = (b.imag != NULL) ? b.imag[i] : 0.0;                         \
      double _Complex bv = mtoc2_cmake(b.real[i], bim);                        \
      double _Complex rv = FN(s, bv);                                          \
      r.real[i] = mtoc2_creal(rv);                                             \
      r.imag[i] = mtoc2_cimag(rv);                                             \
    }                                                                          \
    return r;                                                                  \
  }

#define MTOC2_DEFINE_CELEMWISE_BCAST_TT(name, FN)                              \
  static mtoc2_tensor_t name(mtoc2_tensor_t a, mtoc2_tensor_t b) {             \
    int rnd = a.ndim > b.ndim ? a.ndim : b.ndim;                               \
    long adim[MTOC2_MAX_NDIM], bdim[MTOC2_MAX_NDIM];                           \
    long rdim[MTOC2_MAX_NDIM];                                                 \
    long astride[MTOC2_MAX_NDIM], bstride[MTOC2_MAX_NDIM];                     \
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
    mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(rnd, rdim);               \
    long ix[MTOC2_MAX_NDIM] = {0};                                             \
    for (long k = 0; k < n; k++) {                                             \
      long ai = 0, bi = 0;                                                     \
      for (int i = 0; i < rnd; i++) {                                          \
        ai += ix[i] * astride[i];                                              \
        bi += ix[i] * bstride[i];                                              \
      }                                                                        \
      double aim = (a.imag != NULL) ? a.imag[ai] : 0.0;                        \
      double bim = (b.imag != NULL) ? b.imag[bi] : 0.0;                        \
      double _Complex av = mtoc2_cmake(a.real[ai], aim);                       \
      double _Complex bv = mtoc2_cmake(b.real[bi], bim);                       \
      double _Complex rv = FN(av, bv);                                         \
      r.real[k] = mtoc2_creal(rv);                                             \
      r.imag[k] = mtoc2_cimag(rv);                                             \
      for (int i = 0; i < rnd; i++) {                                          \
        ix[i]++;                                                               \
        if (ix[i] < rdim[i]) break;                                            \
        ix[i] = 0;                                                             \
      }                                                                        \
    }                                                                          \
    return r;                                                                  \
  }

MTOC2_DEFINE_CELEMWISE_TT(mtoc2_tensor_plus_complex_tt, mtoc2_cadd)
MTOC2_DEFINE_CELEMWISE_TT(mtoc2_tensor_minus_complex_tt, mtoc2_csub)
MTOC2_DEFINE_CELEMWISE_TT(mtoc2_tensor_times_complex_tt, mtoc2_cmul)
MTOC2_DEFINE_CELEMWISE_TT(mtoc2_tensor_rdivide_complex_tt, mtoc2_cdiv)

MTOC2_DEFINE_CELEMWISE_TS(mtoc2_tensor_plus_complex_ts, mtoc2_cadd)
MTOC2_DEFINE_CELEMWISE_TS(mtoc2_tensor_minus_complex_ts, mtoc2_csub)
MTOC2_DEFINE_CELEMWISE_TS(mtoc2_tensor_times_complex_ts, mtoc2_cmul)
MTOC2_DEFINE_CELEMWISE_TS(mtoc2_tensor_rdivide_complex_ts, mtoc2_cdiv)

MTOC2_DEFINE_CELEMWISE_ST(mtoc2_tensor_minus_complex_st, mtoc2_csub)
MTOC2_DEFINE_CELEMWISE_ST(mtoc2_tensor_rdivide_complex_st, mtoc2_cdiv)

MTOC2_DEFINE_CELEMWISE_BCAST_TT(mtoc2_tensor_plus_complex_bcast_tt, mtoc2_cadd)
MTOC2_DEFINE_CELEMWISE_BCAST_TT(mtoc2_tensor_minus_complex_bcast_tt, mtoc2_csub)
MTOC2_DEFINE_CELEMWISE_BCAST_TT(mtoc2_tensor_times_complex_bcast_tt, mtoc2_cmul)
MTOC2_DEFINE_CELEMWISE_BCAST_TT(mtoc2_tensor_rdivide_complex_bcast_tt, mtoc2_cdiv)

/* Unary minus on a complex tensor — per-element negate via cneg. */
static mtoc2_tensor_t mtoc2_tensor_uminus_complex(mtoc2_tensor_t a) {
  long n = 1;
  for (int i = 0; i < a.ndim; i++) n *= a.dims[i];
  mtoc2_tensor_t r = mtoc2_tensor_alloc_nd_complex(a.ndim, a.dims);
  for (long i = 0; i < n; i++) {
    double aim = (a.imag != NULL) ? a.imag[i] : 0.0;
    double _Complex av = mtoc2_cmake(a.real[i], aim);
    double _Complex rv = mtoc2_cneg(av);
    r.real[i] = mtoc2_creal(rv);
    r.imag[i] = mtoc2_cimag(rv);
  }
  return r;
}

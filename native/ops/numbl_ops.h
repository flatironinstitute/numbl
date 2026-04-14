/**
 * numbl_ops — pure C tensor-ops library.
 *
 * Stable C ABI for tensor operations dispatched by integer op-code.
 * Caller owns input AND output memory; functions never allocate output.
 *
 * Mirrored 1:1 by the TS implementation in src/numbl-core/ops/.
 * The N-API addon thin-wraps these entry points; the future C-JIT links
 * directly against the same library.
 *
 * All numeric data is column-major (Fortran/MATLAB) double precision.
 * Complex tensors use split storage: separate `re` and `im` Float64 buffers.
 * For complex inputs, `im` may be NULL to indicate "all zero".
 *
 * Return codes: 0 on success, negative on error (see numbl_strerror).
 */

#ifndef NUMBL_OPS_H
#define NUMBL_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes ──────────────────────────────────────────────────────── */

#define NUMBL_OK                 0
#define NUMBL_ERR_BAD_OP        -1
#define NUMBL_ERR_NULL_PTR      -2

/** Human-readable message for a numbl error code. */
const char* numbl_strerror(int code);

/* ── Real binary element-wise ops ─────────────────────────────────────── */

typedef enum {
  NUMBL_REAL_BIN_ADD = 0,
  NUMBL_REAL_BIN_SUB = 1,
  NUMBL_REAL_BIN_MUL = 2,
  NUMBL_REAL_BIN_DIV = 3
} numbl_real_bin_op_t;

/**
 * out[i] = a[i] OP b[i]   for i in [0, n).
 * a, b, out must each point to at least n doubles.
 */
int numbl_real_binary_elemwise(int op, size_t n,
                               const double* a,
                               const double* b,
                               double* out);

/**
 * Scalar-tensor variant.
 * If scalar_on_left:  out[i] = scalar OP arr[i]
 * Else:               out[i] = arr[i] OP scalar
 */
int numbl_real_scalar_binary_elemwise(int op, size_t n,
                                      double scalar,
                                      const double* arr,
                                      int scalar_on_left,
                                      double* out);

/* ── Complex binary element-wise ops ──────────────────────────────────── */

typedef enum {
  NUMBL_COMPLEX_BIN_ADD = 0,
  NUMBL_COMPLEX_BIN_SUB = 1,
  NUMBL_COMPLEX_BIN_MUL = 2,
  NUMBL_COMPLEX_BIN_DIV = 3
} numbl_complex_bin_op_t;

/**
 * Complex element-wise binary op, split storage.
 * a_im or b_im may be NULL → treat as zero (mixed real/complex).
 * out_re and out_im are required (caller allocates both, even if result is real).
 */
int numbl_complex_binary_elemwise(int op, size_t n,
                                  const double* a_re, const double* a_im,
                                  const double* b_re, const double* b_im,
                                  double* out_re, double* out_im);

/**
 * Complex-scalar / tensor variant.
 * arr_im may be NULL → treat tensor as purely real.
 * If scalar_on_left:  out[i] = scalar OP arr[i]
 * Else:               out[i] = arr[i] OP scalar
 */
int numbl_complex_scalar_binary_elemwise(int op, size_t n,
                                         double s_re, double s_im,
                                         const double* arr_re,
                                         const double* arr_im,
                                         int scalar_on_left,
                                         double* out_re, double* out_im);

/* ── Unary element-wise ops (real + complex) ──────────────────────────── */

typedef enum {
  NUMBL_UNARY_EXP   = 0,
  NUMBL_UNARY_LOG   = 1,
  NUMBL_UNARY_LOG2  = 2,
  NUMBL_UNARY_LOG10 = 3,
  NUMBL_UNARY_SQRT  = 4,
  NUMBL_UNARY_ABS   = 5,
  NUMBL_UNARY_FLOOR = 6,
  NUMBL_UNARY_CEIL  = 7,
  NUMBL_UNARY_ROUND = 8,
  NUMBL_UNARY_TRUNC = 9,
  NUMBL_UNARY_SIN   = 10,
  NUMBL_UNARY_COS   = 11,
  NUMBL_UNARY_TAN   = 12,
  NUMBL_UNARY_ASIN  = 13,
  NUMBL_UNARY_ACOS  = 14,
  NUMBL_UNARY_ATAN  = 15,
  NUMBL_UNARY_SINH  = 16,
  NUMBL_UNARY_COSH  = 17,
  NUMBL_UNARY_TANH  = 18,
  NUMBL_UNARY_SIGN  = 19
} numbl_unary_op_t;

/**
 * Real unary element-wise: out[i] = OP(a[i]).
 * a, out must point to at least n doubles.
 */
int numbl_real_unary_elemwise(int op, size_t n,
                              const double* a, double* out);

/**
 * Complex unary element-wise, split storage.
 * a_im may be NULL → treat as zero.
 * out_re and out_im are both required.
 *
 * ABS is unsupported here (output would be real-valued — use
 * numbl_complex_abs for that).  Other ops are supported.
 */
int numbl_complex_unary_elemwise(int op, size_t n,
                                 const double* a_re, const double* a_im,
                                 double* out_re, double* out_im);

/**
 * Complex absolute value: out[i] = sqrt(re[i]^2 + im[i]^2).
 * Output is real-valued.  a_im may be NULL.
 */
int numbl_complex_abs(size_t n,
                      const double* a_re, const double* a_im,
                      double* out);

/* ── Comparisons (logical output stored as 0.0/1.0 in double buffer) ──── */

typedef enum {
  NUMBL_CMP_EQ = 0,  /* ==  */
  NUMBL_CMP_NE = 1,  /* !=  */
  NUMBL_CMP_LT = 2,  /* <   */
  NUMBL_CMP_LE = 3,  /* <=  */
  NUMBL_CMP_GT = 4,  /* >   */
  NUMBL_CMP_GE = 5   /* >=  */
} numbl_cmp_op_t;

/**
 * Real comparison: out[i] = (a[i] OP b[i]) ? 1.0 : 0.0.
 * NaN in either operand yields 0 (except for !=, which yields 1).
 */
int numbl_real_comparison(int op, size_t n,
                          const double* a, const double* b,
                          double* out);

/** Scalar-tensor variant. */
int numbl_real_scalar_comparison(int op, size_t n,
                                 double scalar, const double* arr,
                                 int scalar_on_left, double* out);

/**
 * Complex comparison.  EQ / NE compare both parts; LT/LE/GT/GE compare
 * only the real parts (MATLAB semantics).  a_im, b_im may be NULL.
 */
int numbl_complex_comparison(int op, size_t n,
                             const double* a_re, const double* a_im,
                             const double* b_re, const double* b_im,
                             double* out);

/** Complex-scalar / tensor comparison. */
int numbl_complex_scalar_comparison(int op, size_t n,
                                    double s_re, double s_im,
                                    const double* arr_re,
                                    const double* arr_im,
                                    int scalar_on_left, double* out);

/* ── Op-code dump (for drift detection) ───────────────────────────────── */

/**
 * Writes a small C string describing the op-code enum values into buf.
 * Format is a JSON-like string of category=op_name=value triples.
 * Returns the number of bytes that would be written (excluding terminator).
 * If buf is NULL or buf_size is too small, nothing is written but the
 * required size is still returned.
 */
size_t numbl_dump_op_codes(char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* NUMBL_OPS_H */

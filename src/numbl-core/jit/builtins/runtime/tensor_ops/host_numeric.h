/* @snippet name=jit_host_numeric */
/* JIT host matrix multiply.
 *
 * numbl binds callbacks with `jit_set_host_numeric` (`bindHostNumeric` in
 * executors/jit/valueAdapterC.ts) that run a product through the
 * interpreter's LAPACK provider stack (BLAS when the native addon is
 * loaded), so large products compute the same bits as --opt 0. Products of
 * at most `inline_max` multiply-adds, and every product when nothing is
 * bound, run the caller's own loop, the same loop the stack's inline
 * provider runs. A callback returns nonzero on failure; the caller bails.
 * `ai` / `bi` may be NULL (zero imaginary part).
 *
 * `jit_set_host_numeric` has external linkage so koffi can resolve it after
 * dlopen; everything else is internal.
 */

typedef int (*jit_host_matmul_fn)(const double *a, long m, long k,
                                  const double *b, long n, double *out);
typedef int (*jit_host_matmul_complex_fn)(const double *ar, const double *ai,
                                          long m, long k, const double *br,
                                          const double *bi, long n,
                                          double *out_re, double *out_im);

static jit_host_matmul_fn jit_host_matmul = 0;
static jit_host_matmul_complex_fn jit_host_matmul_complex = 0;
static double jit_host_matmul_inline_max = 0;

void jit_set_host_numeric(jit_host_matmul_fn re, jit_host_matmul_complex_fn cx,
                          double inline_max) {
  jit_host_matmul = re;
  jit_host_matmul_complex = cx;
  jit_host_matmul_inline_max = inline_max;
}

/* Whether `A (m×k) * B (k×n)` goes to the host. */
static inline int jit_host_matmul_wanted(long m, long k, long n) {
  return (double)m * (double)k * (double)n > jit_host_matmul_inline_max;
}

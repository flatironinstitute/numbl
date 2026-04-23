/*
 * koffi_overhead_bench.c — fused kernel + no-op probes for
 * koffi_overhead_bench.ts.
 *
 * The driver compiles this into a .so with the same flags numbl's C-JIT
 * uses (-O2 -march=native -ffast-math -fopenmp-simd etc.) so the per-
 * element cost matches what a C-JIT'd fused loop would achieve.
 *
 * x and y are read/written through pointers that koffi binds directly to
 * JS Float64Arrays, so there is no JS→C copy on input or output.
 */

#include <math.h>
#include <stdint.h>

/* Fused kernel: y[i] = exp(1 + sqrt(x[i])). x must be >= 0. */
void numbl_bench_fused(int64_t n, const double *x, double *y) {
  #pragma omp simd
  for (int64_t i = 0; i < n; i++) {
    y[i] = exp(1.0 + sqrt(x[i]));
  }
}

/* Pure call overhead — no arguments, no work. */
void numbl_bench_noop(void) { }

/* Call overhead including the same argument marshalling as the fused
 * kernel, but no element-wise work. `volatile` on the reads prevents
 * the compiler from optimizing the whole body away. */
void numbl_bench_noop_args(int64_t n, const double *x, double *y) {
  volatile int64_t sink_n = n;
  volatile const double *sink_x = x;
  volatile double *sink_y = y;
  (void)sink_n; (void)sink_x; (void)sink_y;
}

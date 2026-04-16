/*
 * tensor_ops_bench_perop.c — self-contained C baseline matching
 * tensor_ops_bench.m, evaluated *one operation per loop* (no fusion).
 *
 * Each MATLAB statement becomes its own loop over the whole buffer,
 * with an explicit scratch buffer for the output. This is the same
 * evaluation model numbl's interpreter / JS-JIT / C-JIT Phase 3 uses —
 * no cross-statement fusion, each op sees a full pass over N doubles.
 * Purpose: measure the ceiling of "libnumbl_ops-style" per-statement
 * loops without any JS↔C boundary cost at all.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp-simd -fno-math-errno -ffast-math \
 *       benchmarks/tensor_ops_bench_perop.c -o tensor_ops_bench_perop -lm
 *
 * Run:
 *   ./tensor_ops_bench_perop
 *
 * Check values at the end should match the numbl / MATLAB / Octave
 * runs of tensor_ops_bench.m to many digits.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__GNUC__) || defined(__clang__)
#define KEEP(p) __asm__ volatile("" : : "r"(p) : "memory")
#else
#define KEEP(p) ((void)(p))
#endif

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ── Per-op kernels (one loop per MATLAB statement) ─────────────────── */

static void kernel_binary(const double *x, const double *y, double *r,
                          double *scratch, long N) {
  /* r = x + y */
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = x[i] + y[i];
  /* tmp = 0.5 .* x ; r = r - tmp */
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = 0.5 * x[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = r[i] - scratch[i];
  /* tmp = r .* y ; r = tmp + 3.0 */
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = r[i] * y[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = scratch[i] + 3.0;
  /* tmp = abs(y) ; tmp = 1 + tmp ; r = r ./ tmp */
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = fabs(y[i]);
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = 1.0 + scratch[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = r[i] / scratch[i];
}

static void kernel_unary(const double *x, double *u,
                         double *t1, double *t2, long N) {
  /* u = exp(-x .* x) */
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = x[i] * x[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = -t1[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) u[i] = exp(t1[i]);
  /* u = u .* cos(5 .* x) */
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = 5.0 * x[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = cos(t1[i]);
  #pragma omp simd
  for (long i = 0; i < N; i++) u[i] = u[i] * t1[i];
  /* u = u + sin(x+1) .* sin(x+1)  — MATLAB evaluates sin(x+1) twice (no CSE) */
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = x[i] + 1.0;     /* x+1 (lhs)     */
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = sin(t1[i]);     /* sin(x+1) (lhs)*/
  #pragma omp simd
  for (long i = 0; i < N; i++) t2[i] = x[i] + 1.0;     /* x+1 (rhs)     */
  #pragma omp simd
  for (long i = 0; i < N; i++) t2[i] = sin(t2[i]);     /* sin(x+1) (rhs)*/
  #pragma omp simd
  for (long i = 0; i < N; i++) t1[i] = t1[i] * t2[i];  /* product       */
  #pragma omp simd
  for (long i = 0; i < N; i++) u[i] = u[i] + t1[i];
  /* u = abs(u) */
  #pragma omp simd
  for (long i = 0; i < N; i++) u[i] = fabs(u[i]);
  /* u = tanh(u) */
  #pragma omp simd
  for (long i = 0; i < N; i++) u[i] = tanh(u[i]);
}

static double kernel_compare(const double *x, const double *y,
                             double *c1, double *c2, long N) {
  /* c1 = x > 0 */
  #pragma omp simd
  for (long i = 0; i < N; i++) c1[i] = (x[i] > 0.0) ? 1.0 : 0.0;
  /* c2 = y < 0.5 */
  #pragma omp simd
  for (long i = 0; i < N; i++) c2[i] = (y[i] < 0.5) ? 1.0 : 0.0;
  /* c1 = c1 .* c2 */
  #pragma omp simd
  for (long i = 0; i < N; i++) c1[i] = c1[i] * c2[i];
  /* s = sum(c1) */
  double s = 0.0;
  #pragma omp simd reduction(+ : s)
  for (long i = 0; i < N; i++) s += c1[i];
  return s;
}

static double kernel_reduce(const double *x, long N) {
  /* s = sum(x) + mean(x) + max(x) + min(x)  — four separate passes */
  double sum = 0.0;
  #pragma omp simd reduction(+ : sum)
  for (long i = 0; i < N; i++) sum += x[i];
  double mean_sum = 0.0;
  #pragma omp simd reduction(+ : mean_sum)
  for (long i = 0; i < N; i++) mean_sum += x[i];
  double mean = mean_sum / (double)N;
  double mx = -INFINITY;
  #pragma omp simd reduction(max : mx)
  for (long i = 0; i < N; i++) if (x[i] > mx) mx = x[i];
  double mn = INFINITY;
  #pragma omp simd reduction(min : mn)
  for (long i = 0; i < N; i++) if (x[i] < mn) mn = x[i];
  return sum + mean + mx + mn;
}

static double kernel_chain(const double *x, const double *y,
                           double *r, double *scratch, long N) {
  /* r = x .* y + 0.5 */
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = x[i] * y[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = scratch[i] + 0.5;
  /* r = exp(-r .* r) */
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = r[i] * r[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) scratch[i] = -scratch[i];
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = exp(scratch[i]);
  /* r = r .* x */
  #pragma omp simd
  for (long i = 0; i < N; i++) r[i] = r[i] * x[i];
  /* s = sum(r) */
  double s = 0.0;
  #pragma omp simd reduction(+ : s)
  for (long i = 0; i < N; i++) s += r[i];
  return s;
}

int main(void) {
  const long N = 2000000L;
  const int trials = 50;

  printf("N=%ld, trials=%d\n", N, trials);
  printf("----------------------------------------\n");

  double *t  = malloc((size_t)N * sizeof(double));
  double *x  = malloc((size_t)N * sizeof(double));
  double *y  = malloc((size_t)N * sizeof(double));
  double *r  = malloc((size_t)N * sizeof(double));
  double *u  = malloc((size_t)N * sizeof(double));
  double *s1 = malloc((size_t)N * sizeof(double));
  double *s2 = malloc((size_t)N * sizeof(double));
  /* Chain kernel needs its own buffer; overwriting r would clobber the
   * check value sum(r), which the .m script reports from kernel_binary. */
  double *rc = malloc((size_t)N * sizeof(double));
  if (!t || !x || !y || !r || !u || !s1 || !s2 || !rc) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  /* Match MATLAB's linspace(-1, 1, N). */
  const double a = -1.0, b = 1.0;
  for (long i = 0; i < N; i++) {
    double ti = a + (double)i * (b - a) / (double)(N - 1);
    t[i] = ti;
    x[i] = sin(3.1 * ti) * 0.9;
    y[i] = cos(2.7 * ti + 0.4) * 0.8;
  }

  /* Warm-up (touch all kernels once, just like the .m script) */
  kernel_binary(x, y, r, s1, N);
  kernel_unary(x, u, s1, s2, N);
  (void)kernel_compare(x, y, s1, s2, N);
  (void)kernel_reduce(x, N);
  (void)kernel_chain(x, y, rc, s1, N);

  /* ── 1. Binary elemwise ──────────────────────────────────────────── */
  double t1s = now_sec();
  for (int k = 0; k < trials; k++) {
    kernel_binary(x, y, r, s1, N);
    KEEP(r);
  }
  double t1 = now_sec() - t1s;

  /* ── 2. Unary elemwise ───────────────────────────────────────────── */
  double t2s = now_sec();
  for (int k = 0; k < trials; k++) {
    kernel_unary(x, u, s1, s2, N);
    KEEP(u);
  }
  double t2 = now_sec() - t2s;

  /* ── 3. Comparisons + reduction ──────────────────────────────────── */
  double t3s = now_sec();
  double cmp_acc = 0.0;
  for (int k = 0; k < trials; k++) {
    cmp_acc += kernel_compare(x, y, s1, s2, N);
    KEEP(&cmp_acc);
  }
  double t3 = now_sec() - t3s;

  /* ── 4. Reductions ───────────────────────────────────────────────── */
  double t4s = now_sec();
  double red_acc = 0.0;
  for (int k = 0; k < trials; k++) {
    red_acc += kernel_reduce(x, N);
    KEEP(&red_acc);
  }
  double t4 = now_sec() - t4s;

  /* ── 5. Chain ────────────────────────────────────────────────────── */
  double t5s = now_sec();
  double chain_acc = 0.0;
  for (int k = 0; k < trials; k++) {
    chain_acc += kernel_chain(x, y, rc, s1, N);
    KEEP(&chain_acc);
  }
  double t5 = now_sec() - t5s;

  double total = t1 + t2 + t3 + t4 + t5;

  printf("Real binary elemwise:   %7.3f s\n", t1);
  printf("Real unary elemwise:    %7.3f s\n", t2);
  printf("Comparisons + reduce:   %7.3f s\n", t3);
  printf("Reductions:             %7.3f s\n", t4);
  printf("Chained pipeline:       %7.3f s\n", t5);
  printf("----------------------------------------\n");
  printf("elapsed = %.3f s\n", total);

  double sum_r = 0.0, sum_u = 0.0;
  for (long i = 0; i < N; i++) { sum_r += r[i]; sum_u += u[i]; }
  printf("\nCheck values (must match across runtimes):\n");
  printf("  sum(r)       = %.10g\n", sum_r);
  printf("  sum(u)       = %.10g\n", sum_u);
  printf("  cmp_acc      = %.10g\n", cmp_acc);
  printf("  red_acc      = %.10g\n", red_acc);
  printf("  chain_acc    = %.10g\n", chain_acc);
  printf("SUCCESS\n");

  free(t); free(x); free(y); free(r); free(u); free(s1); free(s2); free(rc);
  return 0;
}

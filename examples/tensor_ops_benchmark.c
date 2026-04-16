/*
 * C equivalent of examples/tensor_ops_benchmark.m for a head-to-head
 * performance comparison against MATLAB (and numbl).
 *
 * Each MATLAB statement becomes its own vectorizable loop, so the
 * evaluation model matches an interpreter that evaluates one op per
 * temporary — no cross-statement fusion. The one place we need a scratch
 * buffer is `sin(x+1).*sin(x+1)`, which MATLAB evaluates into a temp
 * once and then squares.
 *
 * Build (same flags numbl uses in binding.gyp — `-fopenmp-simd` is what
 * lets gcc call libmvec's vector exp/log/sin/cos/tanh):
 *
 *   gcc -O3 -march=native -fopenmp-simd -fno-math-errno -ffast-math \
 *       examples/tensor_ops_benchmark.c -o tensor_ops_benchmark -lm
 *
 * Run:
 *   ./tensor_ops_benchmark
 *
 * The "check values" at the end should match the MATLAB / numbl runs to
 * many digits (the deterministic inputs avoid summation-order pitfalls).
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__GNUC__) || defined(__clang__)
/* Compiler barrier: prevents hoisting an idempotent trial-loop body
 * out to run only once. Zero-cost at runtime. */
#define KEEP(p) __asm__ volatile("" : : "r"(p) : "memory")
#else
#define KEEP(p) ((void)(p))
#endif

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void) {
  const long N = 2000000L;
  const int trials = 50;

  printf("N=%ld, trials=%d\n", N, trials);
  printf("----------------------------------------\n");

  double *t = malloc((size_t)N * sizeof(double));
  double *x = malloc((size_t)N * sizeof(double));
  double *y = malloc((size_t)N * sizeof(double));
  double *r = malloc((size_t)N * sizeof(double));
  double *u = malloc((size_t)N * sizeof(double));
  double *tmp = malloc((size_t)N * sizeof(double));
  double _Complex *z = malloc((size_t)N * sizeof(double _Complex));
  double _Complex *w = malloc((size_t)N * sizeof(double _Complex));
  if (!t || !x || !y || !r || !u || !tmp || !z || !w) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  /* Match MATLAB's linspace(-1, 1, N): a + i*(b-a)/(n-1). */
  const double a = -1.0, b = 1.0;
  for (long i = 0; i < N; i++) {
    double ti = a + (double)i * (b - a) / (double)(N - 1);
    t[i] = ti;
    x[i] = sin(3.1 * ti) * 0.9;
    y[i] = cos(2.7 * ti + 0.4) * 0.8;
    double zr = sin(5.0 * ti);
    double zi = cos(4.2 * ti);
    z[i] = zr + I * zi;
  }

  /* ── 1. Real binary element-wise ─────────────────────────────────── */
  double t1s = now_sec();
  for (int k = 0; k < trials; k++) {
    #pragma omp simd
    for (long i = 0; i < N; i++) r[i] = x[i] + y[i];
    #pragma omp simd
    for (long i = 0; i < N; i++) r[i] = r[i] - 0.5 * x[i];
    #pragma omp simd
    for (long i = 0; i < N; i++) r[i] = r[i] * y[i] + 3.0;
    #pragma omp simd
    for (long i = 0; i < N; i++) r[i] = r[i] / (1.0 + fabs(y[i]));
    KEEP(r);
  }
  double t1 = now_sec() - t1s;

  /* ── 2. Real unary element-wise ──────────────────────────────────── */
  double t2s = now_sec();
  for (int k = 0; k < trials; k++) {
    #pragma omp simd
    for (long i = 0; i < N; i++) u[i] = exp(-x[i] * x[i]);
    #pragma omp simd
    for (long i = 0; i < N; i++) u[i] = u[i] * cos(5.0 * x[i]);
    /* sin(x+1) evaluated once, then squared — MATLAB temp semantics */
    #pragma omp simd
    for (long i = 0; i < N; i++) tmp[i] = sin(x[i] + 1.0);
    #pragma omp simd
    for (long i = 0; i < N; i++) u[i] = u[i] + tmp[i] * tmp[i];
    #pragma omp simd
    for (long i = 0; i < N; i++) u[i] = sqrt(fabs(u[i]));
    #pragma omp simd
    for (long i = 0; i < N; i++) u[i] = log(1.0 + u[i]);
    #pragma omp simd
    for (long i = 0; i < N; i++) u[i] = tanh(u[i]);
    KEEP(u);
  }
  double t2 = now_sec() - t2s;

  /* ── 3. Comparisons ──────────────────────────────────────────────── */
  double t3s = now_sec();
  double count = 0.0;
  for (int k = 0; k < trials; k++) {
    double s = 0.0;
    #pragma omp simd reduction(+ : s)
    for (long i = 0; i < N; i++) {
      double c1 = (x[i] > 0.0) ? 1.0 : 0.0;
      double c2 = (y[i] < 0.5) ? 1.0 : 0.0;
      s += c1 * c2;
    }
    #pragma omp simd reduction(+ : s)
    for (long i = 0; i < N; i++) s += (x[i] == y[i]) ? 1.0 : 0.0;
    #pragma omp simd reduction(+ : s)
    for (long i = 0; i < N; i++) s += (x[i] != y[i]) ? 1.0 : 0.0;
    #pragma omp simd reduction(+ : s)
    for (long i = 0; i < N; i++) s += (x[i] <= 0.3) ? 1.0 : 0.0;
    #pragma omp simd reduction(+ : s)
    for (long i = 0; i < N; i++) s += (x[i] >= -0.3) ? 1.0 : 0.0;
    count += s;
  }
  double t3 = now_sec() - t3s;

  /* ── 4. Reductions (sum, mean, max, min, any, all) ───────────────── */
  double t4s = now_sec();
  double acc = 0.0;
  for (int k = 0; k < trials; k++) {
    double s = 0.0;
    #pragma omp simd reduction(+ : s)
    for (long i = 0; i < N; i++) s += x[i];
    double mean = s / (double)N;
    double mx = -INFINITY, mn = INFINITY;
    #pragma omp simd reduction(max : mx) reduction(min : mn)
    for (long i = 0; i < N; i++) {
      if (x[i] > mx) mx = x[i];
      if (x[i] < mn) mn = x[i];
    }
    int any_hit = 0, all_hit = 1;
    for (long i = 0; i < N; i++) {
      if (x[i] > 0.99) any_hit = 1;
      if (!(x[i] > -2.0)) all_hit = 0;
    }
    acc += s + mean + mx + mn + (double)any_hit + (double)all_hit;
  }
  double t4 = now_sec() - t4s;

  /* ── 5. Complex binary + unary elemwise ──────────────────────────── */
  double t5s = now_sec();
  for (int k = 0; k < trials; k++) {
    for (long i = 0; i < N; i++) w[i] = z[i] + 1.0;
    for (long i = 0; i < N; i++) w[i] = w[i] * z[i];
    for (long i = 0; i < N; i++) w[i] = w[i] / (cabs(z[i]) + 1.0);
    for (long i = 0; i < N; i++) w[i] = cexp(w[i]);
    for (long i = 0; i < N; i++) w[i] = csqrt(w[i]);
    KEEP(w);
  }
  double t5 = now_sec() - t5s;

  double total = t1 + t2 + t3 + t4 + t5;

  printf("Real binary elemwise:   %7.3f s\n", t1);
  printf("Real unary elemwise:    %7.3f s\n", t2);
  printf("Comparisons:            %7.3f s\n", t3);
  printf("Reductions:             %7.3f s\n", t4);
  printf("Complex elemwise:       %7.3f s\n", t5);
  printf("----------------------------------------\n");
  printf("Total:                  %7.3f s\n", total);

  /* Check values */
  double sum_r = 0.0, sum_u = 0.0, sum_wr = 0.0, sum_wi = 0.0;
  for (long i = 0; i < N; i++) {
    sum_r += r[i];
    sum_u += u[i];
    sum_wr += creal(w[i]);
    sum_wi += cimag(w[i]);
  }
  printf("\nCheck values (should match between numbl and MATLAB):\n");
  printf("  sum(r)       = %.10g\n", sum_r);
  printf("  sum(u)       = %.10g\n", sum_u);
  printf("  acc          = %.10g\n", acc);
  printf("  count        = %lld\n", (long long)count);
  printf("  sum(real(w)) = %.10g\n", sum_wr);
  printf("  sum(imag(w)) = %.10g\n", sum_wi);
  printf("SUCCESS\n");

  free(t); free(x); free(y); free(r); free(u); free(tmp); free(z); free(w);
  return 0;
}

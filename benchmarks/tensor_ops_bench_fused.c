/*
 * tensor_ops_bench_fused.c — self-contained C ceiling benchmark matching
 * tensor_ops_bench.m, with *each kernel fused into a single element-wise
 * loop* (no intermediate whole-buffer temporaries).
 *
 * This is the theoretical upper bound for a "Phase 5" C-JIT that emits
 * one fused per-element loop per MATLAB function, bypassing the
 * libnumbl_ops per-statement calling convention entirely. Each element
 * is loaded from memory once, flows through all the arithmetic/trig in
 * registers, and is stored once — the whole pipeline pays one round
 * trip to RAM instead of one per op.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp-simd -fno-math-errno -ffast-math \
 *       benchmarks/tensor_ops_bench_fused.c -o tensor_ops_bench_fused -lm
 *
 * Run:
 *   ./tensor_ops_bench_fused
 *
 * Check values at the end should match tensor_ops_bench_perop and the
 * numbl / MATLAB / Octave runs of tensor_ops_bench.m to many digits.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* ── Fused kernels (one loop per MATLAB function) ───────────────────── */

static void kernel_binary(const double *x, const double *y,
                          double *r, long N) {
  /* r = x + y
   * r = r - 0.5 .* x
   * r = r .* y + 3.0
   * r = r ./ (1 + abs(y))                                               */
  #pragma omp simd
  for (long i = 0; i < N; i++) {
    double xi = x[i], yi = y[i];
    double ri = xi + yi;
    ri = ri - 0.5 * xi;
    ri = ri * yi + 3.0;
    ri = ri / (1.0 + fabs(yi));
    r[i] = ri;
  }
}

static void kernel_unary(const double *x, double *u, long N) {
  /* u = exp(-x .* x)
   * u = u .* cos(5 .* x)
   * u = u + sin(x+1) .* sin(x+1)
   * u = abs(u)
   * u = tanh(u)                                                         */
  #pragma omp simd
  for (long i = 0; i < N; i++) {
    double xi = x[i];
    double ui = exp(-xi * xi);
    ui = ui * cos(5.0 * xi);
    double s = sin(xi + 1.0);
    ui = ui + s * s;
    ui = fabs(ui);
    ui = tanh(ui);
    u[i] = ui;
  }
}

static double kernel_compare(const double *x, const double *y, long N) {
  /* s = sum((x>0) .* (y<0.5))  — fused directly into a reduction       */
  double s = 0.0;
  #pragma omp simd reduction(+ : s)
  for (long i = 0; i < N; i++) {
    double c1 = (x[i] > 0.0) ? 1.0 : 0.0;
    double c2 = (y[i] < 0.5) ? 1.0 : 0.0;
    s += c1 * c2;
  }
  return s;
}

static double kernel_reduce(const double *x, long N) {
  /* s = sum(x) + mean(x) + max(x) + min(x)
   * Fuse the four passes into one: one load per element, four accs.    */
  double sum = 0.0, mx = -INFINITY, mn = INFINITY;
  #pragma omp simd reduction(+ : sum) reduction(max : mx) reduction(min : mn)
  for (long i = 0; i < N; i++) {
    double xi = x[i];
    sum += xi;
    if (xi > mx) mx = xi;
    if (xi < mn) mn = xi;
  }
  double mean = sum / (double)N;
  return sum + mean + mx + mn;
}

static double kernel_chain(const double *x, const double *y, long N) {
  /* r = x .* y + 0.5
   * r = exp(-r .* r)
   * r = r .* x
   * s = sum(r)   — fuse the entire pipeline into one reduction loop.   */
  double s = 0.0;
  #pragma omp simd reduction(+ : s)
  for (long i = 0; i < N; i++) {
    double xi = x[i], yi = y[i];
    double ri = xi * yi + 0.5;
    ri = exp(-ri * ri);
    ri = ri * xi;
    s += ri;
  }
  return s;
}

/* For producing a stored r-vector so sum(r) matches the .m script's
 * check value (same fused loop, writes the final r out). */
static void kernel_binary_store(const double *x, const double *y,
                                double *r, long N) {
  kernel_binary(x, y, r, N);
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
  if (!t || !x || !y || !r || !u) {
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

  /* Warm-up */
  kernel_binary(x, y, r, N);
  kernel_unary(x, u, N);
  (void)kernel_compare(x, y, N);
  (void)kernel_reduce(x, N);
  (void)kernel_chain(x, y, N);

  /* ── 1. Binary elemwise ──────────────────────────────────────────── */
  double t1s = now_sec();
  for (int k = 0; k < trials; k++) {
    kernel_binary_store(x, y, r, N);
    KEEP(r);
  }
  double t1 = now_sec() - t1s;

  /* ── 2. Unary elemwise ───────────────────────────────────────────── */
  double t2s = now_sec();
  for (int k = 0; k < trials; k++) {
    kernel_unary(x, u, N);
    KEEP(u);
  }
  double t2 = now_sec() - t2s;

  /* ── 3. Comparisons + reduction ──────────────────────────────────── */
  double t3s = now_sec();
  double cmp_acc = 0.0;
  for (int k = 0; k < trials; k++) {
    cmp_acc += kernel_compare(x, y, N);
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
    chain_acc += kernel_chain(x, y, N);
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

  free(t); free(x); free(y); free(r); free(u);
  return 0;
}

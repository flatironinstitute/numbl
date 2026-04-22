/*
 * parallel_sweep.c — sweep (op × pragma variant × thread count × N)
 * to figure out where `#pragma omp parallel for simd` beats plain
 * `#pragma omp simd` for each element-wise op in native/ops/.
 *
 * Emits CSV on stdout; progress on stderr.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -fno-math-errno -ffast-math \
 *       benchmarks/parallel_sweep.c -o parallel_sweep -lm
 *
 * Run:
 *   ./parallel_sweep > results.csv
 *
 * CSV columns:
 *   op,variant,threads,N,trials,time_ns_min,time_ns_median,gbytes_per_s,checksum
 *
 * Notes:
 *   - "simd" variant is the current production pragma. Thread count is
 *     recorded as 1 but OpenMP is not engaged for it (no parallel region).
 *   - Each (op, variant, threads, N) cell gets its own warmup trial
 *     before timing, so the first trial is not cold-cache biased.
 *   - Trial count is auto-tuned to hit ~30 ms total per cell.
 *   - Checksum is a scalar digest of the output; printed to prevent the
 *     optimizer from DCE'ing the loops and to sanity-check correctness
 *     across variants. All variants of the same op must match.
 */

#include <math.h>
#include <omp.h>
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

/* ─── Kernel generators ──────────────────────────────────────────────── */

/* _Pragma requires a string literal; these expand to the argument. */
#define V_SIMD       "omp simd"
#define V_PAR_STAT   "omp parallel for simd schedule(static)"
#define V_PAR_DYN    "omp parallel for simd schedule(dynamic, 4096)"
#define V_PAR_GUIDED "omp parallel for simd schedule(guided)"
/* Capped variants: num_threads() clause, ignores env OMP_NUM_THREADS.
 * Emulates the strategy we'd bake into the ops for memory-bound loops. */
#define V_PAR_CAP4   "omp parallel for simd schedule(static) num_threads(4)"
#define V_PAR_CAP8   "omp parallel for simd schedule(static) num_threads(8)"

#define GEN_UNARY(fn, PRAG, body)                                           \
  static void fn(const double *a, double *out, long N) {                    \
    _Pragma(PRAG)                                                           \
    for (long i = 0; i < N; i++) { body; }                                  \
  }

#define GEN_BINARY(fn, PRAG, body)                                          \
  static void fn(const double *a, const double *b, double *out, long N) {   \
    _Pragma(PRAG)                                                           \
    for (long i = 0; i < N; i++) { body; }                                  \
  }

#define GEN_REDUCE(fn, PRAG, decl, update)                                  \
  static double fn(const double *a, long N) {                               \
    double s = 0.0;                                                         \
    _Pragma(PRAG)                                                           \
    for (long i = 0; i < N; i++) { update; }                                \
    return s;                                                               \
  }

/* ADD: out = a + b  — 1 flop, 2 reads, 1 write per element (mem-bound) */
GEN_BINARY(add_simd,    V_SIMD,       out[i] = a[i] + b[i])
GEN_BINARY(add_par_s,   V_PAR_STAT,   out[i] = a[i] + b[i])
GEN_BINARY(add_par_d,   V_PAR_DYN,    out[i] = a[i] + b[i])
GEN_BINARY(add_par_g,   V_PAR_GUIDED, out[i] = a[i] + b[i])
GEN_BINARY(add_par_c4,  V_PAR_CAP4,   out[i] = a[i] + b[i])
GEN_BINARY(add_par_c8,  V_PAR_CAP8,   out[i] = a[i] + b[i])

/* MUL: out = a * b  — same memory profile as ADD */
GEN_BINARY(mul_simd,    V_SIMD,       out[i] = a[i] * b[i])
GEN_BINARY(mul_par_s,   V_PAR_STAT,   out[i] = a[i] * b[i])
GEN_BINARY(mul_par_d,   V_PAR_DYN,    out[i] = a[i] * b[i])
GEN_BINARY(mul_par_g,   V_PAR_GUIDED, out[i] = a[i] * b[i])
GEN_BINARY(mul_par_c4,  V_PAR_CAP4,   out[i] = a[i] * b[i])
GEN_BINARY(mul_par_c8,  V_PAR_CAP8,   out[i] = a[i] * b[i])

/* DIV: out = a / b  — higher-latency arith, can benefit more from threads */
GEN_BINARY(div_simd,    V_SIMD,       out[i] = a[i] / b[i])
GEN_BINARY(div_par_s,   V_PAR_STAT,   out[i] = a[i] / b[i])
GEN_BINARY(div_par_d,   V_PAR_DYN,    out[i] = a[i] / b[i])
GEN_BINARY(div_par_g,   V_PAR_GUIDED, out[i] = a[i] / b[i])
GEN_BINARY(div_par_c4,  V_PAR_CAP4,   out[i] = a[i] / b[i])
GEN_BINARY(div_par_c8,  V_PAR_CAP8,   out[i] = a[i] / b[i])

/* ABS: out = fabs(a)  — cheap unary, mem-bound */
GEN_UNARY(abs_simd,    V_SIMD,       out[i] = fabs(a[i]))
GEN_UNARY(abs_par_s,   V_PAR_STAT,   out[i] = fabs(a[i]))
GEN_UNARY(abs_par_d,   V_PAR_DYN,    out[i] = fabs(a[i]))
GEN_UNARY(abs_par_g,   V_PAR_GUIDED, out[i] = fabs(a[i]))
GEN_UNARY(abs_par_c4,  V_PAR_CAP4,   out[i] = fabs(a[i]))
GEN_UNARY(abs_par_c8,  V_PAR_CAP8,   out[i] = fabs(a[i]))

/* SQRT: vector sqrt intrinsic on x86, still mostly mem-bound */
GEN_UNARY(sqrt_simd,   V_SIMD,       out[i] = sqrt(fabs(a[i])))
GEN_UNARY(sqrt_par_s,  V_PAR_STAT,   out[i] = sqrt(fabs(a[i])))
GEN_UNARY(sqrt_par_d,  V_PAR_DYN,    out[i] = sqrt(fabs(a[i])))
GEN_UNARY(sqrt_par_g,  V_PAR_GUIDED, out[i] = sqrt(fabs(a[i])))
GEN_UNARY(sqrt_par_c4, V_PAR_CAP4,   out[i] = sqrt(fabs(a[i])))
GEN_UNARY(sqrt_par_c8, V_PAR_CAP8,   out[i] = sqrt(fabs(a[i])))

/* EXP: libmvec if available, compute-heavy */
GEN_UNARY(exp_simd,    V_SIMD,       out[i] = exp(a[i]))
GEN_UNARY(exp_par_s,   V_PAR_STAT,   out[i] = exp(a[i]))
GEN_UNARY(exp_par_d,   V_PAR_DYN,    out[i] = exp(a[i]))
GEN_UNARY(exp_par_g,   V_PAR_GUIDED, out[i] = exp(a[i]))
GEN_UNARY(exp_par_c4,  V_PAR_CAP4,   out[i] = exp(a[i]))
GEN_UNARY(exp_par_c8,  V_PAR_CAP8,   out[i] = exp(a[i]))

/* COS: compute-heavy, range-reduction inside */
GEN_UNARY(cos_simd,    V_SIMD,       out[i] = cos(a[i]))
GEN_UNARY(cos_par_s,   V_PAR_STAT,   out[i] = cos(a[i]))
GEN_UNARY(cos_par_d,   V_PAR_DYN,    out[i] = cos(a[i]))
GEN_UNARY(cos_par_g,   V_PAR_GUIDED, out[i] = cos(a[i]))
GEN_UNARY(cos_par_c4,  V_PAR_CAP4,   out[i] = cos(a[i]))
GEN_UNARY(cos_par_c8,  V_PAR_CAP8,   out[i] = cos(a[i]))

/* CMP_LT: out = (a < b) ? 1 : 0  — branchy-ish but vectorizable */
GEN_BINARY(cmp_simd,   V_SIMD,       out[i] = (a[i] < b[i]) ? 1.0 : 0.0)
GEN_BINARY(cmp_par_s,  V_PAR_STAT,   out[i] = (a[i] < b[i]) ? 1.0 : 0.0)
GEN_BINARY(cmp_par_d,  V_PAR_DYN,    out[i] = (a[i] < b[i]) ? 1.0 : 0.0)
GEN_BINARY(cmp_par_g,  V_PAR_GUIDED, out[i] = (a[i] < b[i]) ? 1.0 : 0.0)
GEN_BINARY(cmp_par_c4, V_PAR_CAP4,   out[i] = (a[i] < b[i]) ? 1.0 : 0.0)
GEN_BINARY(cmp_par_c8, V_PAR_CAP8,   out[i] = (a[i] < b[i]) ? 1.0 : 0.0)

/* SUM reduction: classic bandwidth-bound reduction */
GEN_REDUCE(sum_simd,    "omp simd reduction(+:s)", , s += a[i])
GEN_REDUCE(sum_par_s,   "omp parallel for simd reduction(+:s) schedule(static)", , s += a[i])
GEN_REDUCE(sum_par_d,   "omp parallel for simd reduction(+:s) schedule(dynamic,4096)", , s += a[i])
GEN_REDUCE(sum_par_g,   "omp parallel for simd reduction(+:s) schedule(guided)", , s += a[i])
GEN_REDUCE(sum_par_c4,  "omp parallel for simd reduction(+:s) schedule(static) num_threads(4)", , s += a[i])
GEN_REDUCE(sum_par_c8,  "omp parallel for simd reduction(+:s) schedule(static) num_threads(8)", , s += a[i])

/* Complex split-storage multiply: 6 flops per element, 4 reads + 2 writes.
 * Higher arithmetic intensity; may tolerate threading overhead earlier. */
static void cmul_body(const double *ar, const double *ai,
                      const double *br, const double *bi,
                      double *outr, double *outi, long N);
#define GEN_CMUL(fn, PRAG)                                                  \
  static void fn(const double *ar, const double *ai,                        \
                 const double *br, const double *bi,                        \
                 double *outr, double *outi, long N) {                      \
    _Pragma(PRAG)                                                           \
    for (long i = 0; i < N; i++) {                                          \
      double x_r = ar[i], x_i = ai[i];                                      \
      double y_r = br[i], y_i = bi[i];                                      \
      outr[i] = x_r * y_r - x_i * y_i;                                      \
      outi[i] = x_r * y_i + x_i * y_r;                                      \
    }                                                                       \
  }
GEN_CMUL(cmul_simd,   V_SIMD)
GEN_CMUL(cmul_par_s,  V_PAR_STAT)
GEN_CMUL(cmul_par_d,  V_PAR_DYN)
GEN_CMUL(cmul_par_g,  V_PAR_GUIDED)
GEN_CMUL(cmul_par_c4, V_PAR_CAP4)
GEN_CMUL(cmul_par_c8, V_PAR_CAP8)

/* ─── Op dispatch table ─────────────────────────────────────────────── */

typedef enum {
  KIND_UNARY,
  KIND_BINARY,
  KIND_REDUCE,
  KIND_CMUL,
} kernel_kind_t;

typedef struct {
  const char *op_name;
  const char *variant_name;
  kernel_kind_t kind;
  void *fn;          /* function pointer (cast by kind) */
  double bytes_per_elem; /* for GB/s reporting */
} kernel_t;

typedef void (*unary_fn_t)(const double *, double *, long);
typedef void (*binary_fn_t)(const double *, const double *, double *, long);
typedef double (*reduce_fn_t)(const double *, long);
typedef void (*cmul_fn_t)(const double *, const double *,
                          const double *, const double *,
                          double *, double *, long);

#define K_UNARY(op, var, fn, bpe)   { op, var, KIND_UNARY,  (void*)fn, bpe }
#define K_BINARY(op, var, fn, bpe)  { op, var, KIND_BINARY, (void*)fn, bpe }
#define K_REDUCE(op, var, fn, bpe)  { op, var, KIND_REDUCE, (void*)fn, bpe }
#define K_CMUL(op, var, fn, bpe)    { op, var, KIND_CMUL,   (void*)fn, bpe }

/* bytes_per_elem: reads + writes, in bytes. For reductions, just the read.
 * "par_cap4" / "par_cap8" use num_threads() clause and do not participate in
 * the thread sweep (their thread count is baked into the pragma). */
static const kernel_t KERNELS[] = {
  K_BINARY("add",     "simd",     add_simd,    24),
  K_BINARY("add",     "par_stat", add_par_s,   24),
  K_BINARY("add",     "par_dyn",  add_par_d,   24),
  K_BINARY("add",     "par_guid", add_par_g,   24),
  K_BINARY("add",     "par_cap4", add_par_c4,  24),
  K_BINARY("add",     "par_cap8", add_par_c8,  24),

  K_BINARY("mul",     "simd",     mul_simd,    24),
  K_BINARY("mul",     "par_stat", mul_par_s,   24),
  K_BINARY("mul",     "par_dyn",  mul_par_d,   24),
  K_BINARY("mul",     "par_guid", mul_par_g,   24),
  K_BINARY("mul",     "par_cap4", mul_par_c4,  24),
  K_BINARY("mul",     "par_cap8", mul_par_c8,  24),

  K_BINARY("div",     "simd",     div_simd,    24),
  K_BINARY("div",     "par_stat", div_par_s,   24),
  K_BINARY("div",     "par_dyn",  div_par_d,   24),
  K_BINARY("div",     "par_guid", div_par_g,   24),
  K_BINARY("div",     "par_cap4", div_par_c4,  24),
  K_BINARY("div",     "par_cap8", div_par_c8,  24),

  K_UNARY("abs",      "simd",     abs_simd,    16),
  K_UNARY("abs",      "par_stat", abs_par_s,   16),
  K_UNARY("abs",      "par_dyn",  abs_par_d,   16),
  K_UNARY("abs",      "par_guid", abs_par_g,   16),
  K_UNARY("abs",      "par_cap4", abs_par_c4,  16),
  K_UNARY("abs",      "par_cap8", abs_par_c8,  16),

  K_UNARY("sqrt",     "simd",     sqrt_simd,   16),
  K_UNARY("sqrt",     "par_stat", sqrt_par_s,  16),
  K_UNARY("sqrt",     "par_dyn",  sqrt_par_d,  16),
  K_UNARY("sqrt",     "par_guid", sqrt_par_g,  16),
  K_UNARY("sqrt",     "par_cap4", sqrt_par_c4, 16),
  K_UNARY("sqrt",     "par_cap8", sqrt_par_c8, 16),

  K_UNARY("exp",      "simd",     exp_simd,    16),
  K_UNARY("exp",      "par_stat", exp_par_s,   16),
  K_UNARY("exp",      "par_dyn",  exp_par_d,   16),
  K_UNARY("exp",      "par_guid", exp_par_g,   16),
  K_UNARY("exp",      "par_cap4", exp_par_c4,  16),
  K_UNARY("exp",      "par_cap8", exp_par_c8,  16),

  K_UNARY("cos",      "simd",     cos_simd,    16),
  K_UNARY("cos",      "par_stat", cos_par_s,   16),
  K_UNARY("cos",      "par_dyn",  cos_par_d,   16),
  K_UNARY("cos",      "par_guid", cos_par_g,   16),
  K_UNARY("cos",      "par_cap4", cos_par_c4,  16),
  K_UNARY("cos",      "par_cap8", cos_par_c8,  16),

  K_BINARY("cmp_lt",  "simd",     cmp_simd,    24),
  K_BINARY("cmp_lt",  "par_stat", cmp_par_s,   24),
  K_BINARY("cmp_lt",  "par_dyn",  cmp_par_d,   24),
  K_BINARY("cmp_lt",  "par_guid", cmp_par_g,   24),
  K_BINARY("cmp_lt",  "par_cap4", cmp_par_c4,  24),
  K_BINARY("cmp_lt",  "par_cap8", cmp_par_c8,  24),

  K_REDUCE("sum",     "simd",     sum_simd,     8),
  K_REDUCE("sum",     "par_stat", sum_par_s,    8),
  K_REDUCE("sum",     "par_dyn",  sum_par_d,    8),
  K_REDUCE("sum",     "par_guid", sum_par_g,    8),
  K_REDUCE("sum",     "par_cap4", sum_par_c4,   8),
  K_REDUCE("sum",     "par_cap8", sum_par_c8,   8),

  K_CMUL("cmul",      "simd",     cmul_simd,   48),
  K_CMUL("cmul",      "par_stat", cmul_par_s,  48),
  K_CMUL("cmul",      "par_dyn",  cmul_par_d,  48),
  K_CMUL("cmul",      "par_guid", cmul_par_g,  48),
  K_CMUL("cmul",      "par_cap4", cmul_par_c4, 48),
  K_CMUL("cmul",      "par_cap8", cmul_par_c8, 48),
};
static const int NUM_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));

/* ─── Harness ───────────────────────────────────────────────────────── */

static int cmp_double(const void *x, const void *y) {
  double a = *(const double*)x, b = *(const double*)y;
  return a < b ? -1 : a > b ? 1 : 0;
}

typedef struct {
  double *a, *b;          /* binary/unary inputs */
  double *out;            /* unary/binary/cmul output (re part for cmul) */
  double *out2;           /* cmul im output */
  double *ai, *bi, *outi; /* cmul extra inputs */
  long N_alloc;
} buffers_t;

static void buffers_init(buffers_t *bf, long N) {
  bf->N_alloc = N;
  bf->a    = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  bf->b    = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  bf->out  = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  bf->out2 = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  bf->ai   = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  bf->bi   = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  bf->outi = (double*)aligned_alloc(64, (size_t)N * sizeof(double));
  if (!bf->a || !bf->b || !bf->out || !bf->out2 || !bf->ai || !bf->bi || !bf->outi) {
    fprintf(stderr, "alloc failed at N=%ld\n", N);
    exit(1);
  }
  /* Deterministic non-trivial content */
  for (long i = 0; i < N; i++) {
    double t = (double)i / (double)N;
    bf->a[i]  = sin(3.1 * t) * 0.9 + 0.1;
    bf->b[i]  = cos(2.7 * t + 0.4) * 0.8 + 0.2;
    bf->ai[i] = cos(1.3 * t) * 0.7;
    bf->bi[i] = sin(2.1 * t) * 0.6 + 0.05;
  }
}

static void buffers_free(buffers_t *bf) {
  free(bf->a); free(bf->b); free(bf->out); free(bf->out2);
  free(bf->ai); free(bf->bi); free(bf->outi);
}

/* Run one kernel trial, return elapsed ns and a checksum of the output. */
static double run_trial(const kernel_t *k, buffers_t *bf, long N,
                        double *checksum_out) {
  double t0 = now_sec();
  double checksum = 0.0;
  switch (k->kind) {
    case KIND_UNARY:
      ((unary_fn_t)k->fn)(bf->a, bf->out, N);
      break;
    case KIND_BINARY:
      ((binary_fn_t)k->fn)(bf->a, bf->b, bf->out, N);
      break;
    case KIND_REDUCE:
      checksum = ((reduce_fn_t)k->fn)(bf->a, N);
      break;
    case KIND_CMUL:
      ((cmul_fn_t)k->fn)(bf->a, bf->ai, bf->b, bf->bi, bf->out, bf->outi, N);
      break;
  }
  double t1 = now_sec();
  KEEP(bf->out);
  KEEP(bf->outi);

  if (k->kind != KIND_REDUCE) {
    /* Sparse checksum — touches ~64 elems, cheap compared to the kernel */
    long step = N / 64 > 0 ? N / 64 : 1;
    for (long i = 0; i < N; i += step) checksum += bf->out[i];
    if (k->kind == KIND_CMUL) {
      for (long i = 0; i < N; i += step) checksum += bf->outi[i];
    }
  }
  *checksum_out = checksum;
  return (t1 - t0) * 1e9;
}

static void measure(const kernel_t *k, buffers_t *bf, long N, int threads,
                    const double target_ns, int min_trials, int max_trials,
                    double *min_ns, double *med_ns, int *trials_out,
                    double *checksum_out) {
  omp_set_num_threads(threads);

  /* Warmup — multiple iterations so branch predictor, TLB, thread pool
   * are all stabilized before we start timing. */
  double dummy;
  for (int w = 0; w < 5; w++) {
    (void)run_trial(k, bf, N, &dummy);
  }

  /* Probe one trial for auto-tuning */
  double probe = run_trial(k, bf, N, &dummy);
  int trials = (int)(target_ns / (probe > 1.0 ? probe : 1.0));
  if (trials < min_trials) trials = min_trials;
  if (trials > max_trials) trials = max_trials;

  double *samples = (double*)malloc((size_t)trials * sizeof(double));
  double checksum = 0.0;
  for (int t = 0; t < trials; t++) {
    double ns = run_trial(k, bf, N, &checksum);
    samples[t] = ns;
  }
  qsort(samples, trials, sizeof(double), cmp_double);
  *min_ns = samples[0];
  *med_ns = samples[trials / 2];
  *trials_out = trials;
  *checksum_out = checksum;
  free(samples);
}

int main(int argc, char **argv) {
  /* N sweep — log-ish grid */
  static const long NS[] = {
    1000, 3000, 10000, 30000, 100000, 300000,
    1000000, 3000000, 10000000
  };
  const int NUM_NS = (int)(sizeof(NS) / sizeof(NS[0]));

  /* Thread counts to sweep for parallel variants */
  static const int THREADS[] = { 1, 2, 4, 6, 8, 12 };
  const int NUM_THREADS = (int)(sizeof(THREADS) / sizeof(THREADS[0]));

  long max_N = NS[NUM_NS - 1];
  fprintf(stderr, "Allocating buffers at N=%ld (~%ld MB total)\n",
          max_N, (7L * max_N * 8) / (1024 * 1024));
  buffers_t bf;
  buffers_init(&bf, max_N);

  /* Quick probe: cost of an empty parallel region, per thread count. */
  fprintf(stderr, "\nEmpty parallel region overhead (µs):\n");
  for (int ti = 0; ti < NUM_THREADS; ti++) {
    int th = THREADS[ti];
    omp_set_num_threads(th);
    /* Warm */
    #pragma omp parallel
    { KEEP(&th); }
    double t0 = now_sec();
    const int reps = 2000;
    for (int r = 0; r < reps; r++) {
      #pragma omp parallel
      { KEEP(&r); }
    }
    double t1 = now_sec();
    fprintf(stderr, "  %2d threads: %.2f µs/region\n",
            th, (t1 - t0) * 1e6 / reps);
  }

  /* CSV header */
  printf("op,variant,threads,N,trials,time_ns_min,time_ns_median,"
         "gbytes_per_s,checksum\n");

  /* Optional filter: ./parallel_sweep <op_filter> */
  const char *filter = argc > 1 ? argv[1] : NULL;

  for (int ki = 0; ki < NUM_KERNELS; ki++) {
    const kernel_t *k = &KERNELS[ki];
    if (filter && strcmp(k->op_name, filter) != 0) continue;

    int is_simd_variant = (strcmp(k->variant_name, "simd") == 0);
    /* Capped variants bake thread count into the pragma via num_threads() —
     * sweeping omp_set_num_threads doesn't change their behavior, so we
     * only run one measurement (still record threads=cap in the CSV). */
    int is_capped = (strstr(k->variant_name, "cap") != NULL);
    int capped_threads = 0;
    if (is_capped) {
      if (strstr(k->variant_name, "4")) capped_threads = 4;
      else if (strstr(k->variant_name, "8")) capped_threads = 8;
    }

    /* Target 60 ms per cell, min 20 trials so even short runs get
     * enough samples for a stable min. */
    const double target_ns = 60e6;
    const int min_trials = 20;
    const int max_trials = 4000;

    for (int ni = 0; ni < NUM_NS; ni++) {
      long N = NS[ni];

      if (is_simd_variant) {
        double min_ns, med_ns, checksum;
        int trials;
        measure(k, &bf, N, 1, target_ns, min_trials, max_trials,
                &min_ns, &med_ns, &trials, &checksum);
        double gbs = (double)N * k->bytes_per_elem / med_ns;
        printf("%s,%s,1,%ld,%d,%.0f,%.0f,%.3f,%.9g\n",
               k->op_name, k->variant_name, N, trials,
               min_ns, med_ns, gbs, checksum);
        fflush(stdout);
      } else if (is_capped) {
        /* One measurement — the cap is baked in via num_threads() clause */
        double min_ns, med_ns, checksum;
        int trials;
        measure(k, &bf, N, capped_threads, target_ns, min_trials,
                max_trials, &min_ns, &med_ns, &trials, &checksum);
        double gbs = (double)N * k->bytes_per_elem / med_ns;
        printf("%s,%s,%d,%ld,%d,%.0f,%.0f,%.3f,%.9g\n",
               k->op_name, k->variant_name, capped_threads, N, trials,
               min_ns, med_ns, gbs, checksum);
        fflush(stdout);
      } else {
        for (int ti = 0; ti < NUM_THREADS; ti++) {
          int th = THREADS[ti];
          double min_ns, med_ns, checksum;
          int trials;
          measure(k, &bf, N, th, target_ns, min_trials, max_trials,
                  &min_ns, &med_ns, &trials, &checksum);
          double gbs = (double)N * k->bytes_per_elem / med_ns;
          printf("%s,%s,%d,%ld,%d,%.0f,%.0f,%.3f,%.9g\n",
                 k->op_name, k->variant_name, th, N, trials,
                 min_ns, med_ns, gbs, checksum);
          fflush(stdout);
        }
      }
      fprintf(stderr, "  %-8s %-9s N=%-9ld done\n",
              k->op_name, k->variant_name, N);
    }
  }

  buffers_free(&bf);
  return 0;
}

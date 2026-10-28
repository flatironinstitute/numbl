/**
 * Benchmark cases for `npm run bench` (see benchmarks/bench.ts).
 *
 * Each case is a MATLAB template. `setup` runs once per size with `n` set
 * (and `rng(0)` called first); it defines the variables listed in `inputs`.
 * `body` is the timed operation and must assign the numeric result `y`. The
 * runner wraps `body` in a kernel function
 *
 *   function [bench_t, y] = bk_<name>(bench_inner, <inputs>)
 *     bench_t0 = tic;
 *     for bench_it = 1:bench_inner
 *       <body>
 *     end
 *     bench_t = toc(bench_t0);
 *   end
 *
 * so the op runs inside a function body under every execution mode, and
 * small sizes are timed in batches of `bench_inner` calls.
 */

export interface BenchCase {
  name: string;
  /** Short human-readable description (shown in the summary). */
  description: string;
  /** What the size parameter `n` means. */
  sizeLabel: string;
  sizes: number[];
  /** Sizes used by --quick; the case is in the quick set iff present. */
  quickSizes?: number[];
  setup: string;
  inputs: string[];
  body: string;
  /** Approximate cost exponent in `n` (1 = linear). Used to skip sizes
   *  whose predicted call time exceeds the per-call limit. */
  costExponent?: number;
  /** True when the case cannot touch the native addon (pure scalar loops),
   *  so the --no-native modes are skipped unless asked for explicitly. */
  nativeIndependent?: boolean;
}

const POW10 = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7];

const VEC5 =
  "a = rand(n, 1); b = rand(n, 1); c = rand(n, 1); d = rand(n, 1); e = rand(n, 1);";
const SQUARE = "m = round(sqrt(n)); A = rand(m, m);";

export const CASES: BenchCase[] = [
  // ── Element-wise ──────────────────────────────────────────────────────
  {
    name: "ew_arith",
    description: "y = a.*b + c.*d - e",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    quickSizes: [1e3, 1e6],
    setup: VEC5,
    inputs: ["a", "b", "c", "d", "e"],
    body: "y = a .* b + c .* d - e;",
  },
  {
    name: "ew_transcendental",
    description: "y = exp(-a.*a) .* sin(b) + c",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    setup: "a = rand(n, 1); b = rand(n, 1); c = rand(n, 1);",
    inputs: ["a", "b", "c"],
    body: "y = exp(-a .* a) .* sin(b) + c;",
  },
  {
    name: "ew_chain_reduce",
    description: "y = sum(exp(-(a.*b + 0.5).^2) .* a)",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    setup: "a = rand(n, 1); b = rand(n, 1);",
    inputs: ["a", "b"],
    body: "y = sum(exp(-(a .* b + 0.5) .^ 2) .* a);",
  },
  {
    name: "ew_complex",
    description: "y = z.*z + w (complex vectors)",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    setup:
      "z = complex(rand(n, 1), rand(n, 1)); w = complex(rand(n, 1), rand(n, 1));",
    inputs: ["z", "w"],
    body: "y = z .* z + w;",
  },

  // ── Reductions ────────────────────────────────────────────────────────
  {
    name: "sum_vec",
    description: "y = sum(a)",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    setup: "a = rand(n, 1);",
    inputs: ["a"],
    body: "y = sum(a);",
  },
  {
    name: "sum_dim1",
    description: "y = sum(A, 1)",
    sizeLabel: "n (elements of a square matrix)",
    sizes: POW10,
    quickSizes: [1e4, 1e6],
    setup: SQUARE,
    inputs: ["A"],
    body: "y = sum(A, 1);",
  },
  {
    name: "sum_dim2",
    description: "y = sum(A, 2)",
    sizeLabel: "n (elements of a square matrix)",
    sizes: POW10,
    setup: SQUARE,
    inputs: ["A"],
    body: "y = sum(A, 2);",
  },
  {
    name: "cumsum_dim1",
    description: "y = cumsum(A, 1)",
    sizeLabel: "n (elements of a square matrix)",
    sizes: POW10,
    setup: SQUARE,
    inputs: ["A"],
    body: "y = cumsum(A, 1);",
  },
  {
    name: "cumsum_dim2",
    description: "y = cumsum(A, 2)",
    sizeLabel: "n (elements of a square matrix)",
    sizes: POW10,
    setup: SQUARE,
    inputs: ["A"],
    body: "y = cumsum(A, 2);",
  },

  // ── Dense linear algebra, FFT, sort ───────────────────────────────────
  {
    name: "matmul",
    description: "y = A * B (n x n)",
    sizeLabel: "n (matrix dimension)",
    sizes: [16, 64, 256, 600, 1000],
    quickSizes: [64, 600],
    setup: "A = rand(n, n); B = rand(n, n);",
    inputs: ["A", "B"],
    body: "y = A * B;",
    costExponent: 3,
  },
  {
    name: "mldivide",
    description: "y = A \\ b (n x n, diagonally dominant)",
    sizeLabel: "n (matrix dimension)",
    sizes: [16, 64, 256, 600, 1000],
    setup: "A = rand(n, n) + n * eye(n); b = rand(n, 1);",
    inputs: ["A", "b"],
    body: "y = A \\ b;",
    costExponent: 3,
  },
  {
    name: "fft",
    description: "y = fft(a) (real input)",
    sizeLabel: "n (vector length, power of 2)",
    sizes: [2 ** 7, 2 ** 10, 2 ** 14, 2 ** 17, 2 ** 20, 2 ** 23],
    quickSizes: [2 ** 10, 2 ** 20],
    setup: "a = rand(n, 1);",
    inputs: ["a"],
    body: "y = fft(a);",
  },
  {
    name: "sort",
    description: "y = sort(a)",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    setup: "a = rand(n, 1);",
    inputs: ["a"],
    body: "y = sort(a);",
  },

  // ── Indexing ──────────────────────────────────────────────────────────
  {
    name: "gather_vec",
    description: "y = a(idx), idx a random permutation",
    sizeLabel: "n (vector length)",
    sizes: POW10,
    setup: "a = rand(n, 1); idx = randperm(n)';",
    inputs: ["a", "idx"],
    body: "y = a(idx);",
  },
  {
    name: "loop_gather",
    description: "scalar loop: s = s + a(idx(i))",
    sizeLabel: "n (loop iterations)",
    sizes: POW10,
    quickSizes: [1e3, 1e6],
    setup: "a = rand(n, 1); idx = randperm(n)';",
    inputs: ["a", "idx"],
    body: [
      "s = 0;",
      "for i = 1:numel(idx)",
      "  s = s + a(idx(i));",
      "end",
      "y = s;",
    ].join("\n"),
    nativeIndependent: true,
  },
  {
    name: "loop_scatter",
    description: "scalar loop: y(i) = 2*a(i) + b(i) into a preallocated y",
    sizeLabel: "n (loop iterations)",
    sizes: POW10,
    setup: "a = rand(n, 1); b = rand(n, 1);",
    inputs: ["a", "b"],
    body: [
      "y = zeros(numel(a), 1);",
      "for i = 1:numel(a)",
      "  y(i) = 2 * a(i) + b(i);",
      "end",
    ].join("\n"),
    nativeIndependent: true,
  },

  // ── Scalar loops ──────────────────────────────────────────────────────
  {
    name: "scalar_series",
    description: "nested scalar loop: sum_k sin(x*k)/k^2 over 100 terms",
    sizeLabel: "n (outer iterations; 100 inner each)",
    sizes: [1e1, 1e2, 1e3, 1e4, 1e5],
    quickSizes: [1e2, 1e4],
    setup: "",
    inputs: ["n"],
    body: [
      "total = 0;",
      "for i = 1:n",
      "  x = i * 0.001;",
      "  acc = 0;",
      "  for k = 1:100",
      "    acc = acc + sin(x * k) / (k * k);",
      "  end",
      "  total = total + acc;",
      "end",
      "y = total;",
    ].join("\n"),
    nativeIndependent: true,
  },
  {
    name: "complex_scalar_loop",
    description: "scalar loop: z = z*z + c on complex scalars",
    sizeLabel: "n (loop iterations)",
    sizes: POW10,
    setup: "",
    inputs: ["n"],
    body: [
      "c = 0.001 + 0.001i;",
      "z = 0.1i;",
      "acc = 0;",
      "for i = 1:n",
      "  z = z * z + c;",
      "  acc = acc + real(z);",
      "end",
      "y = acc;",
    ].join("\n"),
    nativeIndependent: true,
  },

  // ── Structs and cells ─────────────────────────────────────────────────
  {
    name: "struct_loop",
    description: "loop updating struct fields: s.acc = s.acc + a(i)",
    sizeLabel: "n (loop iterations)",
    sizes: POW10,
    quickSizes: [1e3, 1e5],
    setup: "a = rand(n, 1);",
    inputs: ["a"],
    body: [
      "s = struct('acc', 0, 'cnt', 0);",
      "for i = 1:numel(a)",
      "  s.acc = s.acc + a(i);",
      "  s.cnt = s.cnt + 1;",
      "end",
      "y = s.acc / s.cnt;",
    ].join("\n"),
    nativeIndependent: true,
  },
  {
    name: "cell_loop",
    description: "loop accumulating into cell elements: c{k} = c{k} + a(i)",
    sizeLabel: "n (loop iterations)",
    sizes: POW10,
    setup: "a = rand(n, 1);",
    inputs: ["a"],
    body: [
      "c = {0, 0, 0, 0, 0, 0, 0, 0};",
      "for i = 1:numel(a)",
      "  k = mod(i, 8) + 1;",
      "  c{k} = c{k} + a(i);",
      "end",
      "y = 0;",
      "for k = 1:8",
      "  y = y + c{k};",
      "end",
    ].join("\n"),
    nativeIndependent: true,
  },

  // ── Realistic workload ────────────────────────────────────────────────
  {
    // One implicit time step of the Kirchhoff-Love plate from
    // benchmarks/sparse_direct_bench.m: a sparse mat-vec and two sparse
    // triangular solves against a fixed Cholesky factor.
    name: "sparse_plate_step",
    description: "plate time step: rhs = w + g - B*w; y = R \\ (R' \\ rhs)",
    sizeLabel: "n (grid side; (n-2)^2 unknowns)",
    sizes: [21, 41, 81, 121],
    setup: [
      "mx = n - 2; dx = 1 / (n - 1); dt = dx^2;",
      "e1 = ones(mx, 1); D2 = spdiags([e1 -2*e1 e1], -1:1, mx, mx) / dx^2;",
      "I1 = speye(mx); L = kron(I1, D2) + kron(D2, I1);",
      "B = kron(I1, D2*D2) + 2*kron(D2, D2) + kron(D2*D2, I1);",
      "A = (1 + dt) * speye(mx^2) - 0.005*dt*L + 0.5*dt^2*B;",
      "R = chol(A); Rt = R'; Bs = 0.5*dt^2*B;",
      "w = rand(mx^2, 1); g = dt^2 * rand(mx^2, 1);",
    ].join("\n"),
    inputs: ["R", "Rt", "Bs", "w", "g"],
    body: "rhs = w + g - Bs * w;\ny = R \\ (Rt \\ rhs);",
    costExponent: 3,
  },
];

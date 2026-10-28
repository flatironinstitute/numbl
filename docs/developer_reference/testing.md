# Testing

Two test suites, one runner. Both produce a unified coverage report.

## Unit tests

Vitest specs in `src/__tests__/`. Best for:

- lexer and parser edge cases;
- individual builtin branches and error paths;
- runtime helpers (tensor math, COW behavior, type predicates);
- JIT invariants that are hard to trigger from user code;
- drift checks (e.g., op-code synchronization between the TS ops layer and the native addon).

Unit tests are fast and should be the default choice for anything that does not need to exercise user-facing MATLAB syntax end to end.

## Integration test scripts

`.m` scripts under `numbl_test_scripts/`, organized by category. Each script must end with `disp('SUCCESS')` when all assertions pass; the runner checks that the last line of output is exactly `SUCCESS`.

These are the canonical place to encode MATLAB compatibility behavior:

- A script that fails in numbl but passes in real MATLAB documents a known divergence.
- A script that passes in both is a contract that numbl promises to keep.

Integration scripts are wrapped by a Vitest spec (`src/__tests__/test-scripts.test.ts`) so they participate in the same test run and coverage report as the unit tests.

`npm run test:scripts` runs the same scripts through the CLI at `--opt 0` and `--opt 1`; the Vitest wrapper runs them once on the default path. `npm run test:scripts:c` runs them at `--opt 2` (needs a C compiler), which is also what enforces the `%!numbl:assert_jit c` directives.

## JIT parity

`jit_parity/` holds scripts that `npm run jit-parity` runs at `--opt 0`, `1` and `2` and compares for byte-identical output. A divergence between modes is a bug in one of them. See `jit_parity/README.md`.

## Continuous integration

The `lint` job checks formatting, ESLint, `tsc -b` and that the committed generated files are in sync (see [Generated files](../../CONTRIBUTING.md#generated-files)). The `test` job builds everything and runs Vitest; `test-browser` runs Playwright. The `test-lapack` job builds the native addon and then runs the addon parity tests, `npm run test:scripts`, `npm run test:scripts:c`, and `npm run jit-parity` (including `--opt 2`, which needs a C compiler).

## Running tests in both environments

An integration test is only meaningful if it is also checked against the MATLAB reference. Running `matlab -batch "run('path/to/test.m')"` on the same file verifies that the expected behavior is actually MATLAB's behavior. A numbl-only pass is not sufficient: it is easy to assert the wrong thing if you have not confirmed what MATLAB does.

MATLAB may segfault on exit. This is a known MATLAB issue unrelated to the test; the `SUCCESS` line printed before the crash is what matters.

## Coverage

Both suites run under Vitest in the coverage command, so coverage numbers reflect the union of the two suites. When increasing coverage, pick the form of test that fits the code best: unit for surgical logic, integration for user-visible behavior.

## Benchmarks

`npm run bench` (`benchmarks/bench.ts`) answers "how fast is operation X at size N under each execution mode". The cases are MATLAB templates in `benchmarks/cases.ts`: element-wise chains (real and complex), `sum` and `cumsum` along each dimension, `A*B`, `A\b`, `fft`, `sort`, a vectorized gather, indexing loops, scalar and complex-scalar loops, struct and cell loops, and a sparse plate time step. Sizes run from about 1e2 to 1e7 elements (matrix dimensions 16 to 1000 for the dense linear algebra). The modes are `opt0`, `opt1` and `opt2` (the last only when `cc` and koffi are available), each also with `--no-native` (`opt0-nonative`, ...; skipped for the pure scalar loops, which never reach the addon).

For each mode the runner starts one numbl process that runs the cases in turn; if the process dies, the remaining cases run in a new one. Each operation is wrapped in a generated kernel function that repeats it `inner` times between `tic` and `toc`, with `inner` chosen so that a batch takes about 20 ms. The reported time is the median over 5 batches divided by `inner`, that is, the steady-state cost of the operation inside a loop in a function body. Process start-up (a trivial script) and the first call at each size (which includes JIT compilation; the C-JIT cache starts empty unless `--warm-cjit` is given) are reported separately. Note that for small sizes, per-call costs such as copying arguments into a C-JIT kernel are amortized over the batch. Larger sizes are skipped once a call is predicted to exceed 3 s (`--max-call`). The runner also records a checksum of each result, so the summary flags modes that disagree, and the JIT declines and warnings printed while each kernel ran.

```bash
npm run bench                                  # full matrix (tens of minutes)
npm run bench -- --quick                       # 7 cases, 2 sizes, opt0/1/2, 3 reps
npm run bench -- --case 'sum_*,matmul' --size 1e4,600 --mode opt1,opt2 --reps 7
npm run bench -- --list                        # cases, sizes and modes
npm run bench -- --cli dist-cli/cli.js         # time a built CLI (faster start-up)
npm run bench -- report <result.json>          # regenerate the markdown summary
```

Results go to `benchmarks/results/local/` (not committed) unless `--out` is given: a JSON file with the machine (CPU, cores, OS, Node, numbl version and commit, whether the native addon loaded, the `cc` version and any `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` setting) and every entry, and a markdown summary next to it. A baseline for one machine is committed under `benchmarks/results/`. Compare two results with

```bash
npm run bench -- compare <base.json> <new.json> [--threshold 0.15]
```

which prints the geometric-mean time ratio per mode and the entries that got slower or faster by more than the threshold, and exits with status 1 if any entry regressed or newly fails. Comparisons are only meaningful on the same machine, and timings on a busy machine are noisy: rerun flagged cases with `--case` and more `--reps` before drawing conclusions. The benchmarks are not run in CI.

`benchmarks/standalone/` holds two larger workloads that are run by hand: `sparse_direct_bench.m` (sparse direct solvers on a plate operator, see its `.md`) and `chunkie_helmholtz_starfish.m` (a Helmholtz boundary integral solve with the chunkie package, installed through mip on the first run).

## Browser tests

Playwright specs (`tests-browser/`, `npm run test:browser`) cover flows that can only be checked end to end: the IDE, the `numbl/browser` session, the plot gallery and the in-browser test runner. They are separate from the core test suites and run against a built web app (`npm run build`, plus `npm run build:browser`).

See `CONTRIBUTING.md` for the current commands and workflow.

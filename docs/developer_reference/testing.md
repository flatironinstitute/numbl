# Testing

Two test suites, one runner. Both produce a unified coverage report.

## Unit tests

Vitest specs sitting next to `numbl-core`. Best for:

- lexer and parser edge cases;
- individual builtin branches and error paths;
- runtime helpers (tensor math, COW behavior, type predicates);
- JIT invariants that are hard to trigger from user code;
- drift checks (e.g., op-code synchronization between the TS ops layer and the native addon).

Unit tests are fast and should be the default choice for anything that does not need to exercise user-facing MATLAB syntax end to end.

## Integration test scripts

`.m` scripts under the test scripts tree, organized by category. Each script must end with `disp('SUCCESS')` on the last line when all assertions pass — the runner keys on that exact string.

These are the canonical place to encode MATLAB compatibility behavior:

- A script that fails in numbl but passes in real MATLAB documents a known divergence.
- A script that passes in both is a contract that numbl promises to keep.

Integration scripts are wrapped by a Vitest spec so they participate in the same test run and coverage report as the unit tests.

## Running tests in both environments

An integration test is only meaningful if it is also checked against the MATLAB reference. Running `matlab -batch "run('path/to/test.m')"` on the same file verifies that the expected behavior is actually MATLAB's behavior. A numbl-only pass is not sufficient — it is easy to assert the wrong thing if you have not confirmed what MATLAB does.

MATLAB may segfault on exit. This is a known MATLAB issue unrelated to the test; the `SUCCESS` line printed before the crash is what matters.

## Coverage

Both suites run under Vitest in the coverage command, so coverage numbers reflect the union of the two suites. When increasing coverage, pick the form of test that fits the code best: unit for surgical logic, integration for user-visible behavior.

## Browser tests

Playwright specs cover the web-app flows that can only be checked end to end (worker protocol, rendering). They are separate from the core test suites and run against a built web app.

See `CONTRIBUTING.md` for the current commands and workflow.

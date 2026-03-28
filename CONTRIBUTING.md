# Contributing to Numbl

## Local Development Setup

### Prerequisites

- Node.js >= 18
- npm

### Install

```bash
git clone https://github.com/flatironinstitute/numbl.git
cd numbl
npm install
```

### Build

```bash
npm run build          # build everything (web app, CLI, plot viewer)
npm run build:cli      # CLI only
npm run build:web      # web app only
npm run dev            # start web dev server (hot reload)
```

## Project Components

### numbl-core (`src/numbl-core/`)

The compiler and runtime engine. Contains the lexer, parser, lowering/IR, code generator, executor, runtime library, and built-in functions. This is shared between the CLI and the web app.

### Web App (`src/`)

A React + Vite browser-based IDE. Users can write and run `.m` scripts entirely in the browser. Start a development server with `npm run dev`.

### CLI (`src/cli.ts`)

The `numbl` command-line tool. Supports running `.m` files, an interactive REPL, inline evaluation, and more. See `numbl --help` for all commands.

### Execution Server (`server/`)

An optional Node.js HTTP service that executes numbl scripts on behalf of the web IDE. Useful for providing fast, native-backed execution from a remote server. See [server/README.md](server/README.md).

### Native Addon (`native/`)

C++ bindings to LAPACK/OpenBLAS and FFTW for fast linear algebra and FFT. Optional — numbl falls back to JavaScript implementations when the addon is not built.

## The stdlib Bundle

The standard library consists of `.m` files in [`src/stdlib/`](src/stdlib/). These implement functions that are part of MATLAB's built-in surface area but are expressed in numbl itself (e.g., `inputParser` helpers, `readmatrix`, `decomposition`). There is also a `shims/` subdirectory for compatibility wrappers that need to be resolved via a search path (e.g., `+matlab/+internal/...` namespace packages).

Because the web app and CLI both need these files at runtime but cannot read the filesystem arbitrarily (especially in the browser), they are compiled into a single generated TypeScript file:

```
src/numbl-core/stdlib-bundle.ts   ← auto-generated, do not edit
```

This file exports two arrays of `WorkspaceFile` objects — one for the flat stdlib files and one for the shims — which the executor loads automatically before running any user code.

**Regenerate the bundle** any time you add, remove, or modify a file in `src/stdlib/`:

```bash
npx tsx scripts/bundle-stdlib.ts
```

The `build:web` and `build:cli` scripts run this automatically, but when working from source with `npx tsx src/cli.ts` you need to run it once manually (or re-run it after stdlib changes). The script is a no-op if nothing has changed.

> Do not edit `stdlib-bundle.ts` directly — it will be overwritten the next time the script runs.

## Running the CLI from Source

To run the CLI directly from TypeScript source without a build step, use `tsx`:

```bash
npx tsx src/cli.ts                        # interactive REPL
npx tsx src/cli.ts run script.m           # run a .m file
npx tsx src/cli.ts eval "disp(eye(3))"    # evaluate inline code
npx tsx src/cli.ts --help                 # show all commands
```

This is the fastest way to test changes during development.

## Before Committing

Make sure the following all pass before pushing:

```bash
npm run build:cli       # confirm the CLI builds without errors
npm run format:check    # confirm code is formatted (Prettier)
npm test                # run the unit test suite (Vitest)
```

To auto-fix formatting:

```bash
npm run format
```

The pre-commit hook (Husky) runs `lint-staged` (Prettier on staged files) and ESLint automatically on `git commit`.

## Tests

There are two test suites. Both count toward coverage, so use whichever is more appropriate for the code under test.

### Unit Tests (`src/__tests__/`)

```bash
npm test                # run tests with float64 precision
npm run test:float32    # run tests with float32 precision
```

Unit tests use [Vitest](https://vitest.dev/) and are best for testing individual functions, edge cases, and internal logic.

### Integration Tests (`numbl_test_scripts/`)

The integration test suite lives in [`numbl_test_scripts/`](numbl_test_scripts/). Each test is a `.m` script that prints `SUCCESS` as its last line of output if all assertions pass. These are best for testing end-to-end behavior and MATLAB compatibility.

Run all integration tests:

```bash
npm run test:scripts
```

Or run a single test directly from source:

```bash
npx tsx src/cli.ts run numbl_test_scripts/arithmetic/basic_ops.m
```

### Coverage

Both test suites run under vitest (integration scripts are wrapped by `src/__tests__/test-scripts.test.ts`), so a single command produces a unified coverage report:

```bash
npm run test:coverage:all
```

When working to increase test coverage, either add unit tests in `src/__tests__/` or integration scripts in `numbl_test_scripts/` — whichever makes more sense for the code being tested.

## Filing an Issue and Adding a Test

When you find a behavior that differs from MATLAB, the workflow is:

1. **Open a GitHub issue** describing what numbl does versus what MATLAB does. Include a minimal `.m` snippet that reproduces the problem.

2. **Add a test script** to [`numbl_test_scripts/`](numbl_test_scripts/). Pick the most relevant category directory (e.g., `arithmetic/`, `linear_algebra/`, `strings/`) or create a new one. The test should:
   - Reproduce the failing case with `assert(...)` calls.
   - Print `SUCCESS` as the **last line** if everything passes (the test runner looks for this exact string).
   - Be named descriptively, e.g., `test_negative_base_power.m`.

   Example test script:

   ```matlab
   % Test that (-2)^3 gives the correct signed result
   assert((-2)^3 == -8)
   assert((-3)^2 == 9)

   disp('SUCCESS')
   ```

3. **Verify the test fails in numbl but passes in MATLAB** (or at least describes correct MATLAB behavior). The CI will track it as a known failure until the underlying issue is fixed in `numbl-core`.

4. **Fix the issue** in `numbl-core` (lexer, parser, runtime, builtins, etc.), then confirm the test now passes:

   ```bash
   npx tsx src/cli.ts run numbl_test_scripts/<category>/<test>.m
   ```

5. Open a pull request with both the fix and the new test.

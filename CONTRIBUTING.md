# Contributing to Numbl

## Local Development Setup

### Prerequisites

- Node.js >= 18.7
- npm

### Install

```bash
git clone https://github.com/flatironinstitute/numbl.git
cd numbl
npm install
```

### Build

```bash
npm run build          # web app, test runner, CLI, plot viewer, site viewer
npm run build:cli      # CLI only
npm run build:web      # web app only
npm run dev            # start web dev server (hot reload)
```

The published library entry points are built separately: `npm run build:lib` (`numbl`, `numbl/node`), `npm run build:graphics` (`numbl/graphics`) and `npm run build:browser` (`numbl/browser`). All of them run in `prepublishOnly`.

## Project Components

### numbl-core (`src/numbl-core/`)

The language core: lexer, parser, interpreter, JIT compiler, runtime and builtins. It is shared by the CLI, the web app and the library entry points. See [docs/developer_reference/overview.md](docs/developer_reference/overview.md) for how the pieces fit together.

### Web App (`src/`)

A React + Vite browser-based IDE. Users can write and run `.m` scripts entirely in the browser. Start a development server with `npm run dev`.

### CLI (`src/cli.ts`)

The `numbl` command-line tool. Supports running `.m` files, an interactive REPL, inline evaluation, and more. See `numbl --help` for all commands.

### Execution Server (`server/`)

An optional Node.js HTTP service that executes numbl scripts on behalf of the web IDE. Useful for providing fast, native-backed execution from a remote server. See [server/README.md](server/README.md).

### Native Addon (`native/`)

C++ bindings to LAPACK (OpenBLAS on Linux, Accelerate on macOS) and FFTW for fast linear algebra and FFT. It is optional: numbl falls back to JavaScript implementations when the addon is not built.

## The stdlib Bundle

The standard library is a set of `.m` files in [`src/stdlib/`](src/stdlib/) (with namespace and compatibility shims in `src/stdlib/shims/`) that implement parts of MATLAB's function surface in numbl itself, e.g. `inputParser` and `readmatrix`. Because the browser cannot read them from disk, they are compiled into the generated, committed file `src/numbl-core/stdlib-bundle.ts`. Run `npm run bundle:stdlib` after adding, removing or changing a file in `src/stdlib/` and commit the result; see [Generated files](#generated-files) below and [docs/developer_reference/stdlib.md](docs/developer_reference/stdlib.md) for details.

## Generated files

These files are generated. Do not edit them by hand. The first four are committed so a fresh clone works without a build step, and CI fails if a committed copy is out of date (each generator has a `:check` variant).

| File                                                                                                                    | Produced by                                                                                                            | Regenerate when                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/numbl-core/stdlib-bundle.ts`                                                                                       | `npm run bundle:stdlib` (`scripts/bundle-stdlib.ts`)                                                                   | A file under `src/stdlib/` changes. `npm test`, `npm run test:scripts`, `npm run test:scripts:c`, `npm run jit-parity`, `npm run dev` and most `build:*` scripts run it automatically; run it yourself before `npx tsx src/cli.ts`. |
| `src/numbl-core/jit/builtins/runtime/snippets.gen.ts`, `snippets.c.gen.ts`                                              | `npm run build:snippets` (`scripts/build_runtime_snippets.ts`)                                                         | A `.h` or `.js` file under `src/numbl-core/jit/builtins/runtime/` changes. `npm run dev` and most `build:*` scripts run it; tests and `npx tsx` do not.                                                                             |
| `src/builtinNames.gen.ts`                                                                                               | `npm run gen:builtin-names` (`scripts/gen_builtin_names.ts`)                                                           | A builtin or constant is added, removed or renamed. Nothing runs it automatically; a unit test also checks it.                                                                                                                      |
| `dist`, `dist-cli`, `dist-lib`, `dist-browser`, `dist-graphics`, `dist-plot-viewer`, `dist-site-viewer` (not committed) | `npm run build`, `build:cli`, `build:lib`, `build:browser`, `build:graphics`, `build:plot-viewer`, `build:site-viewer` | Before using that bundle. In particular, `numbl --plot` serves `dist-plot-viewer/`, so rebuild it after changing plot rendering code (see [docs/agents.md](docs/agents.md#building)).                                               |

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

CI checks formatting, lint, types, the generated files above, the import-cycle count in `src/numbl-core` (`npm run check:cycles`), the unit tests and the `.m` test scripts. Before pushing, run at least:

```bash
npm run lint            # ESLint
npm run typecheck       # tsc -b
npm test                # unit tests (Vitest)
npm run test:scripts    # .m integration scripts at --opt 0 and --opt 1
```

The pre-commit hook (Husky + lint-staged) runs Prettier on staged files, and ESLint on staged `.ts`/`.tsx` files. To format the whole tree by hand, run `npm run format`.

## Tests

There are two test suites. Both count toward coverage, so use whichever is more appropriate for the code under test.

### Unit Tests (`src/__tests__/`)

```bash
npm test
```

Unit tests use [Vitest](https://vitest.dev/) and are best for testing individual functions, edge cases, and internal logic.

### Integration Tests (`numbl_test_scripts/`)

The integration test suite lives in [`numbl_test_scripts/`](numbl_test_scripts/). Each test is a `.m` script that prints `SUCCESS` as its last line of output if all assertions pass. These are best for testing end-to-end behavior and MATLAB compatibility.

`npm run test:scripts` runs them at `--opt 0` and `--opt 1`; `npm run test:scripts:c` runs them at `--opt 2` (needs a C compiler). To run a single test from source:

```bash
npx tsx src/cli.ts run numbl_test_scripts/arithmetic/basic_ops.m
```

### JIT parity (`jit_parity/`)

`npm run jit-parity` runs the scripts in `jit_parity/` at `--opt 0`, `1` and `2` and fails on any difference in output. See [jit_parity/README.md](jit_parity/README.md).

### Coverage

Both test suites run under vitest (integration scripts are wrapped by `src/__tests__/test-scripts.test.ts`), so a single command produces a unified coverage report:

```bash
npm run test:coverage:all
```

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

3. **Verify the test fails in numbl but passes in MATLAB** (or at least describes correct MATLAB behavior). There is no known-failure list, so a failing script fails CI: commit the test together with its fix.

4. **Fix the issue** in `numbl-core` (lexer, parser, runtime, builtins, etc.), then confirm the test now passes:

   ```bash
   npx tsx src/cli.ts run numbl_test_scripts/<category>/<test>.m
   ```

5. Open a pull request with both the fix and the new test.

# Numbl

Numbl is an open-source numerical computing environment that aims to be compatible with Matlab.

**Early stage project.** Numbl is under active development and new functionality is being added regularly.

[Intro presentation](https://magland.github.io/mip-numbl-presentation)

## Try it in the browser

[![numbl REPL](docs/repl-preview.svg)](https://numbl.org/embed-repl)

Click the terminal above to launch an interactive REPL — no installation required. All execution happens locally in your browser. The full browser IDE with file management and plotting is available at <https://numbl.org>.

Numbl scripts can also be embedded in HTML and Markdown pages (including GitHub Pages). See the [numbl-embed-example](https://magland.github.io/numbl-embed-example/) for usage and a [live demo](https://magland.github.io/numbl-embed-example/example1).

## Command-line usage

For full performance, use the command-line version. If you have Node.js installed, you can run numbl directly with `npx` (no install needed):

```bash
npx numbl                      # interactive REPL
npx numbl eval "disp(eye(3))"  # evaluate inline code
npx numbl run script.m         # run a .m file
```

Or install globally for regular use:

```bash
npm install -g numbl
```

To enable fast linear algebra, build the native LAPACK addon:

```bash
# Best effort auto-build happens during npm install when compatible system libraries are detected.
# Manual build:
numbl build-addon
numbl build-addon --with-deps
```

The native addon links against any compatible BLAS/LAPACK plus FFTW provider it
can detect with `pkg-config`, and it also accepts explicit overrides through
`NUMBL_NATIVE_LIBS`, `NUMBL_NATIVE_INCLUDE_DIRS`, and `NUMBL_NATIVE_CFLAGS`.
When metadata is missing, the install helper also makes a bounded best-effort
attempt with common system link lines such as `openblas/lapack/fftw3` and
`blis/libflame/fftw3` before falling back to TS. System OpenBLAS is the most
common fast path when available, while FLAME/BLIS stacks are also supported
when they expose the expected BLAS/LAPACK symbols. The local fallback builder
stages BLIS + libFLAME + ducc0 by default. See
[native-addon.md](docs/native-addon.md).

## Browser Wasm kernels

Browser-side performance kernels can be built into static `.wasm` assets with:

```bash
npm run build:browser-wasm
```

Configured targets live in `browser-wasm/targets/`, source roots are provided
through env vars or `browser-wasm/local-sources.json`, and generated artifacts
are written to `public/wasm-kernels/`. The runtime can load DUCC0 FFT together
with both browser linalg targets side-by-side: it prefers `blas-lapack` for
`matmul`, `inv`, and square `linsolve`, and uses `flame-blas-lapack` for
rectangular `linsolve` when available. See [browser-wasm.md](docs/browser-wasm.md).

Successful builds replace the runtime manifest by default. Use `--merge` only
when you intentionally want to keep existing manifest entries and add explicit
targets on top.

Experimental browser kernels can still be built explicitly, for example:

```bash
npm run build:browser-wasm -- flame-blas-lapack
npm run build:browser-wasm -- blas-lapack
```

## Backend benchmarks

Use the benchmark harness to compare the available TS, native, and Wasm
backends and verify that their results match before timing. The current harness
uses Tinybench for time-based sampling and percentile statistics:

```bash
npm run bench:backends -- --quick
npm run bench:backends -- --quick --verify-only
npm run bench:backends -- --backend wasm:blas-lapack,wasm:flame-blas-lapack --markdown bench/results/wasm-report.md
```

For in-browser measurements, start the web app and open `/bench`. The browser
page runs the same quick scenarios as the CLI harness and validates outputs
before timing.

## Native addon configuration

Native addon auto-build can be controlled with:

- `NUMBL_SKIP_NATIVE_INSTALL=1` to skip the install-time build attempt
- `NUMBL_FORCE_NATIVE_BUILD=1` to force an install-time build
- `NUMBL_NATIVE_BUILD_FALLBACK=1` to try a local BLIS/libFLAME/ducc0 fallback build during install
- `NUMBL_NATIVE_LIBS` for custom linker flags such as `-L/opt/lib -lopenblas -lfftw3`
- `NUMBL_NATIVE_INCLUDE_DIRS` for extra include directories using the platform path separator
- `NUMBL_NATIVE_CFLAGS` for extra compiler flags
- `NUMBL_NATIVE_BLAS_PKGS`, `NUMBL_NATIVE_LAPACK_PKGS`, `NUMBL_NATIVE_FFT_PKGS` to override pkg-config package search order
- `NUMBL_DEBUG_NATIVE=1` to print provider resolution details during native builds

## Usage

<!-- BEGIN CLI HELP -->
```
Usage: numbl <command> [options]

Commands:
  run <file.m>       Run a .m file
  eval "<code>"      Evaluate inline code
  run-tests [dir]    Run .m test scripts (default: numbl_test_scripts/)
  build-addon        Build native addon [--with-deps]
  info               Print machine-readable info (JSON)
  list-builtins      List available built-in functions
  mip <subcommand>   Package manager (install, uninstall, list, avail, info)
  (no command)       Start interactive REPL

Global options:
  --version, -V      Print version and exit
  --help, -h         Print this help message

Options (for REPL):
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)

Options (for run and eval):
  --dump-js <file>   Write JIT-generated JavaScript to file
  --dump-ast         Print AST as JSON
  --verbose          Detailed logging to stderr
  --stream           NDJSON output mode
  --path <dir>       Add extra workspace directory
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)
  --add-script-path  Add the script's directory to the workspace (run only)
  --opt <level>      Optimization level (0=none, 1=JIT scalar functions; default: 1)

Environment variables:
  NUMBL_PATH         Extra workspace directories (separated by :)
```
<!-- END CLI HELP -->

## VS Code extension

The [Numbl extension for VS Code](https://marketplace.visualstudio.com/items?itemName=jmagland.numbl) lets you run `.m` scripts directly in the editor with inline error diagnostics and a built-in figure viewer.

## Upgrading

```bash
npm install -g numbl@latest
```

Note: if you previously built the native addon, you'll need to run `numbl build-addon` again after upgrading.

## Authors

Jeremy Magland and Dan Fortunato, Center for Computational Mathematics, Flatiron Institute.

## License

Apache 2.0.

## Acknowledgements

See [acknowledgements.md](docs/acknowledgements.md).

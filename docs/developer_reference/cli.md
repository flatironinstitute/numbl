# CLI

The `numbl` command-line entry point is a small orchestration layer over `executeCode`. It provides the same execution engine used by the browser and the server, backed by Node-side adapters for file I/O and system info.

## Surfaces

- **Script runner** — run a `.m` file and exit.
- **Eval** — evaluate an inline code string.
- **REPL** — interactive session with persistent variables between inputs.
- **Test runner** — run `.m` test scripts under a directory, gathering `SUCCESS` markers.
- **Addon builder** — compile the native addon from source.
- **Introspection** — machine-readable system/version info, list of registered builtins.

## Adapters

The CLI provides:

- a file-I/O adapter backed by Node's `fs`;
- a system adapter providing env vars, cwd, and platform info;
- an output router that writes to stdout/stderr (or NDJSON stream mode);
- an optional plot callback that forwards figures to an embedded HTTP server for a browser viewer.

## Options of note

- `--opt <0|1|e1>` — interpreter / JS-JIT / JS-JIT with inline C kernels (see [jit/overview.md](jit/overview.md)).
- `--par` — parallelize fused chain kernels with OpenMP under `--opt e1`.
- `--dump-js` — write JIT-generated JavaScript to disk for inspection (under `--opt e1`, the inline C kernel source is embedded as JS string literals).
- `--dump-ast` — print the parsed AST as JSON.
- `--stream` — NDJSON output, suitable for programmatic consumers.
- `--path <dir>` and the `NUMBL_PATH` environment variable — add extra workspace directories to the search path.
- `--plot` / `--plot-port` — enable the embedded plot server for figures.

C kernel compilation is controlled by `NUMBL_CC`, `NUMBL_CFLAGS`, and `NUMBL_NO_NATIVE_CFLAGS`. `NUMBL_OMP_THRESHOLD` sets the minimum element count before `--par`'s parallel-for kicks in (default 100 000).

## REPL specifics

The REPL keeps a long-lived `Environment` across inputs so variables persist. It also maintains plot hold state so a sequence of plot commands composes as expected. Errors do not tear down the session; they are formatted by the diagnostics layer and the prompt returns.

See `numbl --help` for the current, canonical option list; this file describes roles, not the exact option set.

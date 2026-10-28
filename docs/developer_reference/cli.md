# CLI

The `numbl` command-line entry point is a small orchestration layer over `executeCode`. It provides the same execution engine used by the browser and the server, backed by Node-side adapters for file I/O and system info.

## Surfaces

- **Script runner** — run a `.m` file and exit.
- **Eval** — evaluate an inline code string.
- **Parse** — tokenize, lex, and parse a `.m` file without executing it (optionally dumping the AST).
- **REPL** — interactive session with persistent variables between inputs.
- **Test runner** — run `.m` test scripts under a directory, gathering `SUCCESS` markers.
- **Addon builder** — compile the native addon from source.
- **Site builder** — `numbl build-site` bundles a project directory plus the prebuilt browser IDE (`dist-site-viewer/`) into a static, deployable site. See [web-app.md](web-app.md#static-site-viewer).
- **Introspection** — machine-readable system/version info, list of registered builtins.
- **Execution server** — local HTTP server (`numbl serve`) used by the browser IDE to run code outside the worker sandbox.

## Source layout

`src/cli.ts` is the entry point: qhull setup, `--help`, and the command dispatch for `run`, `eval`, `parse`, the REPL and the smaller commands. Larger pieces live in sibling modules:

- `node-platform.ts` — `initNodePlatform`: native addon loading, C-JIT registration, the koffi bridge and the mip core package install. Also exported from `numbl/node` for library hosts (mip install is opt-in there).
- `cli-args.ts` — option parsing on `node:util` `parseArgs` (shared run/eval/REPL options, `NUMBL_PATH`).
- `cli-repl.ts` — the interactive REPL.
- `cli-run-tests.ts` — `run-tests`; its `runTestScript` is also used by the vitest wrapper (`src/__tests__/test-scripts.test.ts`) and `scripts/annotate_assert_jit.ts`.
- `cli-scan.ts` — workspace scanning (`scanMFiles`) and test-script discovery (`findTestScripts`).
- `cli-build-site.ts`, `cli-profile.ts` — `build-site`, and `--profile` output / `show-profile`.
- `cli-fileio.ts`, `cli-system.ts`, `cli-plot-*.ts` — the adapters below.

## Adapters

The CLI provides:

- a file-I/O adapter backed by Node's `fs` (byte-level primitives; fids and stream semantics are in the core `FileTable`, see [architecture.md](architecture.md#file-io));
- a system adapter providing env vars, cwd, and platform info;
- an output router that writes to stdout/stderr (or NDJSON stream mode);
- an optional plot callback that forwards figures to an embedded HTTP server for a browser viewer.

## Options of note

- `--opt <0|1|2>` — interpreter only / JS-JIT (default) / C-JIT via `cc` + koffi (Node only; falls back to JS-JIT when unavailable). See [jit/overview.md](jit/overview.md).
- `--dump-js` — write JS-JIT-generated JavaScript to disk for inspection (`--opt 1`, plus the JS-JIT sections that pick up the slack at `--opt 2`).
- `--dump-c` — write C-JIT-generated C to disk for inspection (`--opt 2`). At `--opt 2` both flags can be used together: C-JIT kernels land in the `--dump-c` file, JS-JIT fallbacks in the `--dump-js` file.
- `--dump-ast` — print the parsed AST as JSON.
- `--stream` — NDJSON output, suitable for programmatic consumers.
- `--path <dir>` and the `NUMBL_PATH` environment variable — add extra workspace directories to the search path.
- `--plot` / `--plot-port` — enable the embedded plot server for figures.

## REPL specifics

The REPL persists variables across inputs by re-invoking `executeCode` for each line and threading a `variableValues` record (`Record<string, RuntimeValue>`) in and out — each call builds a fresh `Environment` seeded from that record, rather than holding one long-lived `Environment` object. The threading is done by `ExecSession` (`src/numbl-core/execSession.ts`), which also carries plot hold state, figure handles, search paths changed by `addpath`/`rmpath`, the implicit cwd and the live uihtml session; the IDE worker (`src/numbl-worker.ts`) and the `numbl/browser` session worker use the same class. Errors do not tear down the session; they are formatted by the diagnostics layer, the state is left as it was, and the prompt returns.

See `numbl --help` for the current, canonical option list; this file describes roles, not the exact option set.

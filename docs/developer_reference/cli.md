# CLI

The `numbl` command-line entry point is a small orchestration layer over `executeCode`. It provides the same execution engine used by the browser and the server, backed by Node-side adapters for file I/O and system info.

## Surfaces

- **Script runner**: run a `.m` file and exit.
- **Eval**: evaluate an inline code string.
- **Parse**: tokenize, lex, and parse a `.m` file without executing it (optionally dumping the AST).
- **REPL**: interactive session with persistent variables between inputs.
- **Test runner**: run `.m` test scripts under a directory, gathering `SUCCESS` markers.
- **Addon builder**: compile the native addon from source.
- **Site builder**: `numbl build-site` bundles a project directory plus the prebuilt browser IDE (`dist-site-viewer/`) into a static, deployable site. See [web-app.md](web-app.md#static-site-viewer).
- **Introspection**: machine-readable system/version info, list of registered builtins.
- **Execution server**: local HTTP server (`numbl serve`) used by the browser IDE to run code outside the worker sandbox.

## Source layout

`src/cli.ts` is the entry point: `--help`, and the command dispatch for `run`, `eval`, `parse`, the REPL and the smaller commands. Larger pieces live in sibling modules:

- `node-platform.ts`: `initNodePlatform`: fills the process-default host (native addon, qhull, the koffi bridge, the C backend) and installs the mip core package. Also exported from `numbl/node` for library hosts (mip install is opt-in there).
- `cli-args.ts`: option parsing on `node:util` `parseArgs` (shared run/eval/REPL options, `NUMBL_PATH`).
- `cli-repl.ts`: the interactive REPL.
- `cli-run-tests.ts`: `run-tests`; its `runTestScript` is also used by the vitest wrapper (`src/__tests__/test-scripts.test.ts`) and `scripts/annotate_assert_jit.ts`.
- `cli-scan.ts`: workspace scanning (`scanMFiles`) and test-script discovery (`findTestScripts`).
- `cli-build-site.ts`, `cli-profile.ts`: `build-site`, and `--profile` output / `show-profile`.
- `cli-fileio.ts`, `cli-system.ts`, `cli-plot-*.ts`: the adapters below.

## Adapters

The CLI provides:

- a file-I/O adapter backed by Node's `fs` (byte-level primitives; fids and stream semantics are in the core `FileTable`, see [architecture.md](architecture.md#file-io));
- a system adapter providing env vars, cwd, and platform info;
- an output router that writes to stdout/stderr (or NDJSON stream mode). `fprintf(2, ...)` text reaches the host through `ExecOptions.onErrorOutput` (falling back to `onOutput` when a host does not set it, as the browser does); the CLI writes it to stderr, and stream mode marks its `output` messages with `stream: "stderr"`;
- an optional plot callback that forwards figures to an embedded HTTP server for a browser viewer.

## Options of note

The full option list is the `--help` output, mirrored in [docs/agents.md](../agents.md#cli-usage-for-numbl). A few points it does not spell out:

- At `--opt 2`, `--dump-js` and `--dump-c` can be used together: C-JIT kernels land in the `--dump-c` file and the JS-JIT units that picked up the rest in the `--dump-js` file.
- `--profile <file>` (run/eval) writes per-builtin timing JSON, which `numbl show-profile <file>` pretty-prints. Neither is listed in `--help`.
- `NUMBL_CC` selects the C compiler for the C-JIT (default `cc`) and `NUMBL_NO_QHULL` skips loading the qhull WASM backend. Neither is listed in `--help`.

## Stream protocol

With `--stream`, `run` and `eval` write one JSON message per line to stdout. The `StreamMessage` union and its `encodeStreamMessage` / `decodeStreamMessage` live in `src/streamProtocol.ts`, which the CLI, the execution server and the web app share. Every message carries `v` (currently 1) and a `type`:

- `output` (`text`) and `drawnow` (`plotInstructions`) as the program runs;
- `input_request` (`prompt`) when the program calls `input()`; the reply is read as one line from stdin;
- `error`, one per diagnostic: `message`, `errorType`, `identifier`, `file`, `line`, `snippet`, `callStack`;
- `done` (`success`, plus `generatedJS` / `generatedC`) as the last message.

The process exits 0 after a successful run and 1 after an error, as without `--stream`. Version 1 only adds to the earlier unversioned format (which had no `v`, `success`, `identifier`, `callStack` or `input_request`, and exited 0 after errors), so the decoder accepts unversioned messages; it ignores messages of other versions or unknown types. The execution server relays the messages as server-sent events and appends `close` (`success`) once the process has exited.

## REPL specifics

The REPL re-invokes `executeCode` for each input through an `ExecSession` (`src/numbl-core/execSession.ts`). Each call builds a fresh runtime and interpreter, but runs against the session's `SessionState` (see [architecture.md](architecture.md#sessions)): the base workspace `Environment`, globals, persistents, warning and PRNG state, and so on. The session also carries plot hold state, figure handles, search paths changed by `addpath`/`rmpath`, the implicit cwd and the live uihtml session; the IDE worker (`src/numbl-worker.ts`) and the `numbl/browser` session worker use the same class. Errors do not tear down the session; they are formatted by the diagnostics layer and the prompt returns. As at the MATLAB prompt, what the failed input did before the error (variables, globals) stays.

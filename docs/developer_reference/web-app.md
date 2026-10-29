# Web App

A React + Vite single-page app that lets users write and run `.m` scripts in the browser. All numbl execution happens inside a dedicated Web Worker; the main thread is the UI.

## Three-layer structure

- **Main thread (UI)**: React components for the editor, console, plot view, workspace, file tree. Owns the virtual file system visible to the user and orchestrates the worker. The IDE is `IDEWorkspace`, which holds run state and layout and composes the panels in `src/components/ide/` (editor and run controls, output, figures, compact-layout output, file browser, execution settings).
- **Worker (runtime)**: imports numbl-core and calls `executeCode`. Holds persistent REPL state (variables, hold state, opt level, worker-side VFS mirror).
- **Numbl-core**: the same TypeScript that runs under the CLI. The browser-specific adapters plug it in.

## Worker protocol

A message-passing contract between the main thread and the worker (`src/numbl-worker.ts`), typed as discriminated unions in `src/numblWorkerProtocol.ts`:

- main → worker: run a script (optionally after a hidden preamble, optionally sharing state with the REPL), execute a REPL command, set optimization level, install the optional WASM LAPACK accelerator (`set_wasm_bridge`), enable `input()` (`set_input`), update workspace files, clear state, forward a uihtml iframe event.
- worker → main: output chunks, plot-draw events, `input()` requests (blocking-channel messages, below), uihtml events for an iframe, run completion (`done`) or one `error` per diagnostic (the final one flagged `last`), REPL completion (`result`), workspace (VFS) deltas.

The main-thread side lives in `NumblWorkerClient` (`src/numblWorkerClient.ts`): it owns the `Worker`, the cancel buffer, the server that answers `input()` requests, the uihtml relay (`src/graphics/uihtmlRelay.ts`) and the dispatch of incoming messages to host callbacks. React hosts (embed, embed REPL, figure page and plot gallery) use it through `useNumblWorker`, which starts the worker on mount and terminates it on unmount, and the IDE through its worker execution backend (below); the benchmark page uses the class directly, one worker per benchmark. `buildWorkerFiles` (`src/buildWorkerFiles.ts`) assembles the VFS and workspace files sent with a run. The embedding API (`src/browser/`) has its own worker and protocol.

The worker is the only thing that ever calls `executeCode`. The main thread never runs numbl-core logic directly; it imports only types and small pure helpers (the figures reducer, `formatDiagnostic`). The builtin names used for syntax highlighting and the landing page come from `src/builtinNames.gen.ts`, generated from the registry by `npm run gen:builtin-names` (CI and a unit test check that it is current).

## Execution backends

The IDE runs scripts through an `ExecutionBackend` (`src/execution/ExecutionBackend.ts`): `run(request, events)`, `stop()`, `sendInput(text)`, and a `capabilities` record. A run reports output, plot instructions, input requests, one `onError` per diagnostic and, last and exactly once, `onDone` with a status (`success`, `error` or `cancelled`). `useExecutionBackends` provides both implementations and picks the active one:

- `WorkerExecutionBackend` wraps the IDE's `NumblWorkerClient`. It also runs REPL commands (`execute`), which share state with persistent runs. `stop()` cancels a persistent run cooperatively and otherwise replaces the worker. Cooperative cancellation sets the cancel `SharedArrayBuffer`, which the interpreter polls between statements and JIT-compiled code polls in its loops (see [executors.md](executors.md#cancellation)), so Stop also ends a JIT'd loop, and wakes a worker blocked in `input()`, `pause` or `gpu.call` (below); the run ends with a `CancellationError` and the worker keeps its state. Output from JIT-compiled code streams to the console as it is produced ([executors.md](executors.md#output-streaming)).
- `RemoteExecutionBackend` posts the project's text files to a numbl execution server (`numbl serve`, see [server.md](server.md)) and decodes the server-sent events with the `--stream` protocol (`src/streamProtocol.ts`, see [cli.md](cli.md#stream-protocol)). It has no REPL or persistent workspace, `input()` returns an empty value, and uihtml figures are not interactive.

## Virtual file system

An in-memory filesystem (`VirtualFileSystem`) on the main thread provides the user's files. A companion adapter (`BrowserFileIOAdapter`) wraps it to implement the core's `FileIOAdapter` interface inside the worker. Changes the running script makes (via `writematrix`, `save`, etc.) are captured as a delta and replayed back to the main thread after each run so the UI stays consistent.

When the worker runs a script, it first sets the VFS cwd to the directory of that script, mirroring the CLI `run` command, which chdir's into `dirname(filepath)`. The script's folder thus becomes the first-priority implicit search path (see `executeCode`'s cwd-as-search-path handling), so a driver script in a subdirectory resolves sibling functions and relative file I/O against its own folder rather than the project root.

## Blocking waits

The interpreter runs synchronously, so `input()`, `pause` and `gpu.call` must block the worker while another thread works. All three go through one `BlockingChannel` (`src/numbl-core/runtime/blockingChannel.ts`), bound to the run's cancel buffer, so Stop interrupts each of them:

- **Request.** The worker posts a `blocking_channel` message with the request body (structured-cloned) to a server thread and waits on a state word in the cancel buffer with `Atomics.wait`. The server (`createChannelServer`) writes the reply into a shared reply region that grows on demand (JSON plus the raw bytes of typed arrays) and completes the request with a compare-exchange on its sequence number, so a late reply cannot complete a newer request.
- **Stop.** The worker waits in 50 ms slices and checks the cancel flag after each; the host's `signalCancel` sets the flag and also notifies the state word, so the wait ends at once. The worker then tells the server it gave up and throws the interpreter's `CancellationError`.
- **Users.** `input()`: `channelInput` in the worker and `InputServer` on the main thread (`src/inputChannel.ts`), which asks the host (`onInputRequest`, otherwise `window.prompt`) and answers. `pause`: a wait with nobody to answer. `gpu.call`: a request to the WebGPU helper worker (see [diagnostics-and-extensions.md](diagnostics-and-extensions.md#webgpu-from-numbljs-browser)).

`websave`/`webread` still block with a synchronous XHR (`BrowserFileIOAdapter`), which Stop does not interrupt. Everything else uses normal async messaging.

## Plot viewer

Figures render in-page with `FigureView` (`src/graphics/`), which the web app imports from source and compiles with the rest of the app. The separate plot viewer bundle (`dist-plot-viewer/`) wraps the same component for the CLI's `--plot` flag; it is not used by the web app, but it must be rebuilt after renderer changes. See [plotting.md](plotting.md).

## Static site viewer

A second Vite entry (`src/site-viewer/`, built to `dist-site-viewer/` via `vite.site-viewer.config.ts`) packages the same `IDEWorkspace` as a standalone, deployable app. Instead of IndexedDB or the URL hash, it loads files from a baked-in `project.zip` through the `useStaticProjectFiles` hook. It and `useShareProjectFiles` (the URL-hash share page) both wrap `useInMemoryProjectFiles`, so edits live in memory and reset on reload; the static hook is binary-safe because it never round-trips content through text. It uses a relative asset base (`base: "./"`) so it works under any deploy subpath.

The `numbl build-site` CLI command (`src/cli-build-site.ts`) copies this bundle into an output directory, zips a project tree into `project.zip` (honoring `.numblignore`), writes a `numbl-project.json` manifest (title, entry file, repository URL, initial output-panel height; flags override an existing manifest), and injects the deploy base. A reusable GitHub Action (`.github/actions/build-site/`) and a starter template (`examples/numbl-project-template/`) wire it to GitHub Pages. Markdown files render via the shared `MarkdownView` component (also used by the docs page), with a rendered/source toggle in the editor pane.

The base the Action passes is the one `actions/configure-pages` reports rather than a guess from the repository name, so a site served anywhere other than `<owner>.github.io/<repo>/` (a custom domain, or a host such as a git vault that serves pages under its own path) is built for where it will actually live. Should the base be wrong regardless, `useStaticProjectFiles` retries `project.zip` relative to the document, so a mismatch costs one 404 rather than an unusable site.

## Environment differences from the CLI

- No access to the native addon; LAPACK and FFT use the in-tree JS fallbacks, optionally accelerated by a WASM LAPACK bridge (`set_wasm_bridge`). The worker's capabilities (this accelerator, qhull, the WebGPU helper) are its host, built by `bootBrowserHost` (`src/browserHost.ts`); messages wait until `host.ready` resolves.
- No C-JIT (no C compiler); `--opt 2` is unavailable in the browser.
- File I/O is VFS-only; scripts cannot reach the real disk.
- `input()` is interactive through the UI, not `stdin`.

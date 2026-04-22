# Web App

A React + Vite single-page app that lets users write and run `.m` scripts in the browser. All numbl execution happens inside a dedicated Web Worker; the main thread is the UI.

## Three-layer structure

- **Main thread (UI)** — React components for the editor, console, plot view, workspace, file tree. Owns the virtual file system visible to the user and orchestrates the worker.
- **Worker (runtime)** — imports numbl-core and calls `executeCode`. Holds persistent REPL state (variables, hold state, opt level, worker-side VFS mirror).
- **Numbl-core** — the same TypeScript that runs under the CLI. The browser-specific adapters plug it in.

## Worker protocol

A message-passing contract between the main thread and the worker:

- main → worker: run source, set optimization level, update workspace files, clear state.
- worker → main: output chunks, run completion with result, error, plot-draw events, workspace deltas.

The worker is the only thing that ever calls `executeCode`. The main thread never runs numbl-core logic directly.

## Virtual file system

An in-memory filesystem (`VirtualFileSystem`) on the main thread provides the user's files. A companion adapter (`BrowserFileIOAdapter`) wraps it to implement the core's `FileIOAdapter` interface inside the worker. Changes the running script makes (via `writematrix`, `save`, etc.) are captured as a delta and replayed back to the main thread after each run so the UI stays consistent.

## Synchronous input in an async world

MATLAB code that calls `input(...)` expects to block until the user types a response. In the worker this is implemented with a `SharedArrayBuffer` plus `Atomics.wait`/`Atomics.notify`: the worker blocks on the shared buffer, the main thread collects the response from the UI and writes it back, and `Atomics.notify` wakes the worker. This is the only synchronous cross-thread bridge; everything else uses normal async messaging.

## Plot viewer

Figures render in a separate Vite-built bundle that the browser app embeds (and the CLI's `--plot` flag serves over HTTP). See [plotting.md](plotting.md). Unit tests do not exercise this bundle — changes under the graphics source tree need a rebuild before they show up.

## Environment differences from the CLI

- No access to the native addon; LAPACK and FFT use the in-tree JS fallbacks.
- No C-JIT (no C compiler); `--opt 2` behaves like `--opt 1`.
- File I/O is VFS-only; scripts cannot reach the real disk.
- `input()` is interactive through the UI, not `stdin`.

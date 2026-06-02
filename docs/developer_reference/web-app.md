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

When the worker runs a script, it first sets the VFS cwd to the directory of that script — mirroring the CLI `run` command, which chdir's into `dirname(filepath)`. The script's folder thus becomes the first-priority implicit search path (see `executeCode`'s cwd-as-search-path handling), so a driver script in a subdirectory resolves sibling functions and relative file I/O against its own folder rather than the project root.

## Synchronous input in an async world

MATLAB code that calls `input(...)` expects to block until the user types a response. In the worker this is implemented with a `SharedArrayBuffer` plus `Atomics.wait`/`Atomics.notify`: the worker blocks on the shared buffer, the main thread collects the response from the UI and writes it back, and `Atomics.notify` wakes the worker. This is the only synchronous cross-thread bridge; everything else uses normal async messaging.

## Plot viewer

Figures render in a separate Vite-built bundle that the browser app embeds (and the CLI's `--plot` flag serves over HTTP). See [plotting.md](plotting.md). Unit tests do not exercise this bundle — changes under the graphics source tree need a rebuild before they show up.

## Static site viewer

A second Vite entry (`src/site-viewer/`, built to `dist-site-viewer/` via
`vite.site-viewer.config.ts`) packages the same `IDEWorkspace` as a standalone,
deployable app. Instead of IndexedDB or the URL hash, it loads files from a
baked-in `project.zip` through the `useStaticProjectFiles` hook (a binary-safe
sibling of `useShareProjectFiles` — edits live in memory and reset on reload).
It uses a relative asset base (`base: "./"`) so it works under any deploy
subpath.

The `numbl build-site` CLI command copies this bundle into an output directory,
zips a project tree into `project.zip` (honoring `.numblignore`), writes an
optional `numbl-project.json` manifest (title + entry file), and injects the
deploy base. A reusable GitHub Action (`.github/actions/build-site/`) and a
starter template (`examples/numbl-project-template/`) wire it to GitHub Pages.
Markdown files render via the shared `MarkdownView` component (also used by the
docs page), with a rendered/source toggle in the editor pane.

## Environment differences from the CLI

- No access to the native addon; LAPACK and FFT use the in-tree JS fallbacks.
- No C-JIT (no C compiler); `--opt 2` is unavailable in the browser.
- File I/O is VFS-only; scripts cannot reach the real disk.
- `input()` is interactive through the UI, not `stdin`.

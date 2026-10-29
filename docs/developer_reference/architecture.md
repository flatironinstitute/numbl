# Architecture

## Layers

Numbl is organized as a platform-agnostic language core surrounded by thin platform adapters.

- **numbl-core**: lexer, parser, interpreter, JIT, runtime, built-ins, and the stdlib bundle. Pure TypeScript with no Node- or browser-specific imports. Every platform drives it through the same entry point.
- **Platform surfaces**: the CLI, the browser web app (running inside a Web Worker), and the optional HTTP execution server. Each provides a host (`NumblHost`, see [Host capabilities](#host-capabilities)), a `FileIOAdapter` and `SystemAdapter` plus output callbacks, then calls the core's `executeCode` function.
- **Native addon (optional)**: a Node-API C++ module wrapping LAPACK/OpenBLAS, FFTW, and selected element-wise kernels. Numbl detects it at runtime and falls back to pure-JS implementations when absent.

## Execution pipeline

1. **Lex**: source text to tokens.
2. **Parse**: tokens to an AST.
3. **Index**: the `Workspace` collects the visible files and registers them with its `CodeIndex`, which caches parsed ASTs, maps files to function and class names (search paths, `+pkg`, `@Class`, `private/`), holds the `ClassTable`, and builds the `FunctionIndex` that name resolution (`src/numbl-core/resolution/`) uses.
4. **Interpret**: the `Interpreter` walks the AST against a `Runtime` and a chain of call frames (each holding an `Environment`). Control flow is signalled by typed signal objects (break, continue, return) returned up the statement-execution stack: not thrown.
5. **JIT (optional)**: the interpreter offers each top-level script, loop and user-function call to the executor registry, which type-specializes what it can (there is no hotness threshold). The JIT (in-tree, self-contained under `src/numbl-core/jit`) lowers AST to a typed IR, then emits JS (`--opt 1`, the default) or C compiled via `cc` + koffi (`--opt 2`, Node only). Both backends register executors that compete via the registry. Constructs the JIT can't lower are declined statically and run on the interpreter.
6. **Runtime values**: results are `RuntimeValue`s (numbers, tensors, strings, structs, cells, class instances, etc.). Output, plot instructions, and workspace changes flow back through callbacks on `ExecOptions`.

## Component map

| Component        | Responsibility                                                                                                             |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Lexer            | Tokenize `.m` source, handling MATLAB's whitespace- and context-sensitive rules.                                           |
| Parser           | Produce the AST (statements and expressions).                                                                              |
| Workspace        | The visible file set (caller files, implicit cwd, search path, stdlib); addpath, rmpath and cd (`workspace/workspace.ts`). |
| CodeIndex        | AST cache; function/class name registry; builds the `FunctionIndex`; owns the `ClassTable` (class info, inheritance).      |
| Interpreter      | AST walker; drives builtin dispatch and JIT triggers.                                                                      |
| Resolution       | What a name means in a scope (variable, local/private/package function, class member, builtin); target to function body.   |
| Runtime          | Value constructors, plot accumulator, output router; per-run, with the session's state in `rt.session`.                    |
| SessionState     | What outlives a run: base workspace, globals, persistents, warning and PRNG state, tic, appdata, listeners.                |
| JIT              | Type-specializes hot code to JS (`--opt 1`) and to C (`--opt 2`).                                                          |
| Builtin registry | `IBuiltin` lookup by name, with arity and help; the JIT keeps its own builtin set with type rules and JS/C emitters.       |
| Ops layer        | Dispatches tensor kernels to the native addon or JS fallbacks.                                                             |
| Stdlib bundle    | `.m` files loaded into every run's workspace.                                                                              |
| FileTable        | MATLAB fid semantics (fopen modes, positions, EOF, encodings, write-back) over the adapter's byte primitives.              |
| Platform adapter | Implements file I/O, system info, and output/plot callbacks for one environment.                                           |
| Host             | A session's platform capabilities: numeric providers, geometry, C-JIT, native libraries, gpu (`numbl-core/host.ts`).       |

## Sessions

Each `executeCode` call builds a fresh `Runtime` and `Interpreter`. What MATLAB keeps between commands at the prompt lives in a `SessionState` (`numbl-core/sessionState.ts`), passed as `ExecOptions.session` and held by the runtime as `rt.session`: the base workspace `Environment` (reused as every run's base frame, so variables and the base workspace's `global` links carry over, also past an error), globals (`rt.$g`), persistents (`rt.persistentStore`), the warning on/off state, the PRNG state, the `tic` time, `spparms`, appdata, event listeners and the last JIT decline reason. An `ExecSession` owns one; a one-shot run (CLI `run`/`eval`) gets a fresh one, so two sessions in one process share none of it. Helpers without a runtime handle (the PRNG, `emitWarning`, the JIT decline log) reach it through `currentSession()`, which reads the active runtime and falls back to a detached session outside any run (unit tests). Platform capabilities are not session state: they are the session's host (below).

`clear` follows MATLAB: `clear` and `clear variables` remove the calling scope's variables and global links (globals keep their values), `clear global` the globals, `clear functions` (or `clear name` for a function file) the persistents, and `clear all` all three. Persistents of a function whose file changed since they were made are cleared at the start of the next run. The PRNG stream, warning state and appdata survive `clear all`, as in MATLAB.

The JIT's compile caches stay per run: each run parses its files afresh and a compiled unit binds the run's `Runtime`. A unit that raises an error in a run whose base workspace persists is re-run by the interpreter (`Interpreter.baseWorkspacePersists`), so the assignments it made before the error are kept, as at opt 0.

## Host capabilities

What the platform provides lives in one object, a `NumblHost` (`numbl-core/host.ts`), rather than in module globals:

| Field              | What                                                                                | Read by                                              |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `numeric`          | the LAPACK/BLAS provider stack ([native-addon.md](runtime/native-addon.md))         | `getLapackBridge()`, `providerFor()`                 |
| `geometry`         | qhull Delaunay and convex-hull backends                                             | `getDelaunayBackend()`, `getConvexHullBackend()`     |
| `cjit`             | the C backend for `--opt 2` (opaque; a `JitBackend`)                                | `registerExecutorsForOpt`                            |
| `nativeBridge`     | the koffi bridge (native libraries for `.numbl.js`, and the C-JIT)                  | `cBackend` (`interp.rt.host`), `loadJsUserFunctions` |
| `gpu`              | the `gpu` object given to `.numbl.js` files (`src/gpuHelper.ts`)                    | `loadJsUserFunctions`                                |
| `fileIO`, `system` | defaults for runs that do not pass `ExecOptions.fileIO` / `system`                  | `executeCode`                                        |
| `ready`            | a promise that resolves once the capabilities that load asynchronously are in place | the browser workers, before their first run          |

A run gets its host from `ExecOptions.host`; an `ExecSession` is constructed with one (`new ExecSession({ host })`) and passes it to every run. A run given none uses the process-default host (`defaultHost()`). The runtime holds it as `rt.host`; helpers without a runtime handle use `currentHost()`, which reads the active runtime's host and falls back to the default, as `currentSession()` does for session state. So two sessions in one process with different hosts do not share any capability (`src/__tests__/host.test.ts`). `createHost(init)` makes a host; fields it is not given are capabilities it does not have (its `numeric` stack starts with only the inline provider), nothing is inherited from the default.

Hosts are built at platform boot:

- **Node**: `initNodePlatform` (`src/node-platform.ts`, also in `numbl/node`) fills a host, by default the process-default one: the native addon provider, qhull, the koffi bridge, the C backend (`nodeCJitBackend()` in `executors/jit/cjit.node.ts`, imported only here so the browser bundle never sees the C compile graph), and, with `adapters: true`, the Node file and system adapters. The CLI uses the default host and passes it explicitly.
- **Browser workers**: both workers build theirs with `bootBrowserHost` (`src/browserHost.ts`), which also documents which capabilities each worker has. The IDE worker gets the WebGPU helper, qhull and (when the settings name one) the wasm LAPACK provider; the `numbl/browser` session worker gets the WebGPU helper only. The IDE worker queues messages until `host.ready` resolves (`deferUntilReady`), and the session worker awaits it in `boot`, so no run sees a capability that is still loading. The wasm accelerator is not part of `ready`: until it is installed, products use the other providers.

The process-wide setters remain for code that does not pass a host, and edit the default host: `registerProvider`/`unregisterProvider`, `setDelaunayBackend`/`setConvexHullBackend`, `setJsGpuHelper` and `registerNodeCompileC` (the last three deprecated). Per-run wiring stays on `ExecOptions` (`onInput`, `cancelSAB`, `onHtmlSourceEvent`, output callbacks), since it belongs to one run's caller rather than to the platform. Two things stay process-wide by design: the C runtime-snippet table (static C source, installed by `nodeCJitBackend`) and the `setBridgeLog` diagnostic sink.

To add a capability (for example a compute device): add an optional field to `NumblHost` (and `NumblHostInit`), typed structurally or as `object`/`unknown` if its real type would make `host.ts` import beyond the leaf modules it uses now (`check:cycles` counts type imports); fill it in `initNodePlatform` and/or `bootBrowserHost`, adding its load to `ready` with `addPending` if it is asynchronous; and read it where it is used from `ctx.rt.host` (builtins) or `currentHost()` (helpers). Nothing else needs to change: sessions, runs and the workers already carry the host.

## File I/O

All stream semantics live in `numbl-core/fileTable.ts`. A `FileIOAdapter` supplies byte primitives (`readFile`, `writeFile`, `existsPath`, `resolvePath`, and optionally random access through `handles`: `openHandle`/`readAt`/`writeAt`/`size`/`close`) plus the operations behind individual builtins (dir, copyfile, zip, ...). All fids on the same path share one `FileStore` (`numbl-core/fileStore.ts`), so a reader sees another fid's writes at once, as in MATLAB. With `handles` (Node) the store is a `HandleStore`: reads go through a 64 KB cache and writes through a 64 KB buffer, so memory does not grow with file size. Without them (the browser VFS) it is a `MemoryStore` holding the whole file. Buffered writes reach the host on `fclose`, before reads, before any other adapter call (the runtime sees the adapter through `fileIOView`, which flushes first), and when `executeCode` returns. There is one table per adapter object, so fids stay open across runs that reuse the adapter (the CLI REPL). Adapters that predate this and implement only the old `fopen`/`freadBytes`/... methods still work: the table emulates `readFile`/`writeFile` with them.

## Data flow

- **Source → AST → RuntimeValue** is the critical path.
- **Side channels**: output text, plot instructions, and diagnostics go out through callbacks on `ExecOptions`. Input (e.g., `input()`) comes back through a platform-provided callback: in the browser worker this is a request on the blocking channel (`SharedArrayBuffer` + `Atomics.wait`, see [web-app.md](web-app.md#blocking-waits)) so it can be synchronous and still be stopped.
- **JIT events** go through their own callbacks on `ExecOptions`: `onJitCompile` (generated code per unit) and `onJitBail` (runtime bails to the interpreter).

# Architecture

## Layers

Numbl is organized as a platform-agnostic language core surrounded by thin platform adapters.

- **numbl-core**: lexer, parser, interpreter, JIT, runtime, built-ins, and the stdlib bundle. Pure TypeScript with no Node- or browser-specific imports. Every platform drives it through the same entry point.
- **Platform surfaces**: the CLI, the browser web app (running inside a Web Worker), and the optional HTTP execution server. Each provides a `FileIOAdapter` and `SystemAdapter` plus output callbacks, then calls the core's `executeCode` function.
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

## Sessions

Each `executeCode` call builds a fresh `Runtime` and `Interpreter`. What MATLAB keeps between commands at the prompt lives in a `SessionState` (`numbl-core/sessionState.ts`), passed as `ExecOptions.session` and held by the runtime as `rt.session`: the base workspace `Environment` (reused as every run's base frame, so variables and the base workspace's `global` links carry over, also past an error), globals (`rt.$g`), persistents (`rt.persistentStore`), the warning on/off state, the PRNG state, the `tic` time, `spparms`, appdata, event listeners and the last JIT decline reason. An `ExecSession` owns one; a one-shot run (CLI `run`/`eval`) gets a fresh one, so two sessions in one process share none of it. Helpers without a runtime handle (the PRNG, `emitWarning`, the JIT decline log) reach it through `currentSession()`, which reads the active runtime and falls back to a detached session outside any run (unit tests). Host capabilities that belong to the session rather than a run, such as a device context, belong there too.

`clear` follows MATLAB: `clear` and `clear variables` remove the calling scope's variables and global links (globals keep their values), `clear global` the globals, `clear functions` (or `clear name` for a function file) the persistents, and `clear all` all three. Persistents of a function whose file changed since they were made are cleared at the start of the next run. The PRNG stream, warning state and appdata survive `clear all`, as in MATLAB.

The JIT's compile caches stay per run: each run parses its files afresh and a compiled unit binds the run's `Runtime`. A unit that raises an error in a run whose base workspace persists is re-run by the interpreter (`Interpreter.baseWorkspacePersists`), so the assignments it made before the error are kept, as at opt 0.

## File I/O

All stream semantics live in `numbl-core/fileTable.ts`. A `FileIOAdapter` supplies byte primitives (`readFile`, `writeFile`, `existsPath`, `resolvePath`, and optionally random access through `handles`: `openHandle`/`readAt`/`writeAt`/`size`/`close`) plus the operations behind individual builtins (dir, copyfile, zip, ...). All fids on the same path share one `FileStore` (`numbl-core/fileStore.ts`), so a reader sees another fid's writes at once, as in MATLAB. With `handles` (Node) the store is a `HandleStore`: reads go through a 64 KB cache and writes through a 64 KB buffer, so memory does not grow with file size. Without them (the browser VFS) it is a `MemoryStore` holding the whole file. Buffered writes reach the host on `fclose`, before reads, before any other adapter call (the runtime sees the adapter through `fileIOView`, which flushes first), and when `executeCode` returns. There is one table per adapter object, so fids stay open across runs that reuse the adapter (the CLI REPL). Adapters that predate this and implement only the old `fopen`/`freadBytes`/... methods still work: the table emulates `readFile`/`writeFile` with them.

## Data flow

- **Source → AST → RuntimeValue** is the critical path.
- **Side channels**: output text, plot instructions, and diagnostics go out through callbacks on `ExecOptions`. Input (e.g., `input()`) comes back through a platform-provided callback: in the browser worker this is a request on the blocking channel (`SharedArrayBuffer` + `Atomics.wait`, see [web-app.md](web-app.md#blocking-waits)) so it can be synchronous and still be stopped.
- **JIT events** go through their own callbacks on `ExecOptions`: `onJitCompile` (generated code per unit) and `onJitBail` (runtime bails to the interpreter).

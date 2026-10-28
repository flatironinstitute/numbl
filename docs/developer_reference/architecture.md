# Architecture

## Layers

Numbl is organized as a platform-agnostic language core surrounded by thin platform adapters.

- **numbl-core** — lexer, parser, interpreter, JIT, runtime, built-ins, and the stdlib bundle. Pure TypeScript with no Node- or browser-specific imports. Every platform drives it through the same entry point.
- **Platform surfaces** — the CLI, the browser web app (running inside a Web Worker), and the optional HTTP execution server. Each provides a `FileIOAdapter` and `SystemAdapter` plus output callbacks, then calls the core's `executeCode` function.
- **Native addon (optional)** — a Node-API C++ module wrapping LAPACK/OpenBLAS, FFTW, and selected element-wise kernels. Numbl detects it at runtime and falls back to pure-JS implementations when absent.

## Execution pipeline

1. **Lex** — source text to tokens.
2. **Parse** — tokens to an AST.
3. **Index** — the `Workspace` collects the visible files and registers them with its `CodeIndex`, which caches parsed ASTs, maps files to function and class names (search paths, `+pkg`, `@Class`, `private/`), holds the `ClassTable`, and builds the `FunctionIndex` that name resolution (`src/numbl-core/resolution/`) uses.
4. **Interpret** — the `Interpreter` walks the AST against a `Runtime` and a chain of call frames (each holding an `Environment`). Control flow is signalled by typed signal objects (break, continue, return) returned up the statement-execution stack — not thrown.
5. **JIT (optional)** — hot user functions, hot loops, and feasible top-level scripts are specialized. The JIT (in-tree, self-contained under `src/numbl-core/jit`) lowers AST to a typed IR, then emits JS (`--opt 1`, the default) or C compiled via `cc` + koffi (`--opt 2`, Node only). Both backends register executors that compete via the registry. Constructs the JIT can't lower are declined statically and run on the interpreter.
6. **Runtime values** — results are `RuntimeValue`s (numbers, tensors, strings, structs, cells, class instances, etc.). Output, plot instructions, and workspace changes flow back through callbacks on `ExecOptions`.

## Component map

| Component        | Responsibility                                                                                                             |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Lexer            | Tokenize `.m` source, handling MATLAB's whitespace- and context-sensitive rules.                                           |
| Parser           | Produce the AST (statements and expressions).                                                                              |
| Workspace        | The visible file set (caller files, implicit cwd, search path, stdlib); addpath, rmpath and cd (`workspace/workspace.ts`). |
| CodeIndex        | AST cache; function/class name registry; builds the `FunctionIndex`; owns the `ClassTable` (class info, inheritance).      |
| Interpreter      | AST walker; drives builtin dispatch and JIT triggers.                                                                      |
| Resolution       | What a name means in a scope (variable, local/private/package function, class member, builtin); target to function body.   |
| Runtime          | Value constructors, global/persistent stores, plot accumulator, output router.                                             |
| JIT              | Type-specializes hot code to JS (`--opt 1`) and to C (`--opt 2`).                                                          |
| Builtin registry | `IBuiltin` lookup by name; provides type resolution and JIT emission.                                                      |
| Ops layer        | Dispatches tensor kernels to the native addon or JS fallbacks.                                                             |
| Stdlib bundle    | `.m` files loaded into every run's workspace.                                                                              |
| FileTable        | MATLAB fid semantics (fopen modes, positions, EOF, encodings, write-back) over the adapter's byte primitives.              |
| Platform adapter | Implements file I/O, system info, and output/plot callbacks for one environment.                                           |

## File I/O

All stream semantics live in `numbl-core/fileTable.ts`. A `FileIOAdapter` supplies byte primitives (`readFile`, `writeFile`, `existsPath`, `resolvePath`, and optionally random access through `handles`: `openHandle`/`readAt`/`writeAt`/`size`/`close`) plus the operations behind individual builtins (dir, copyfile, zip, ...). All fids on the same path share one `FileStore` (`numbl-core/fileStore.ts`), so a reader sees another fid's writes at once, as in MATLAB. With `handles` (Node) the store is a `HandleStore`: reads go through a 64 KB cache and writes through a 64 KB buffer, so memory does not grow with file size. Without them (the browser VFS) it is a `MemoryStore` holding the whole file. Buffered writes reach the host on `fclose`, before reads, before any other adapter call (the runtime sees the adapter through `fileIOView`, which flushes first), and when `executeCode` returns. There is one table per adapter object, so fids stay open across runs that reuse the adapter (the CLI REPL). Adapters that predate this and implement only the old `fopen`/`freadBytes`/... methods still work: the table emulates `readFile`/`writeFile` with them.

## Data flow

- **Source → AST → RuntimeValue** is the critical path.
- **Side channels**: output text, plot instructions, and diagnostics go out through callbacks on `ExecOptions`. Input (e.g., `input()`) comes back through a platform-provided callback — in the browser worker this is bridged through a `SharedArrayBuffer` so it can be synchronous.
- **JIT compilation events** are reported through a separate callback, so tooling can surface what was specialized and why.

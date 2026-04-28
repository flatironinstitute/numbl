# Architecture

## Layers

Numbl is organized as a platform-agnostic language core surrounded by thin platform adapters.

- **numbl-core** — lexer, parser, lowering, interpreter, JIT, runtime, built-ins, and the stdlib bundle. Pure TypeScript with no Node- or browser-specific imports. Every platform drives it through the same entry point.
- **Platform surfaces** — the CLI, the browser web app (running inside a Web Worker), and the optional HTTP execution server. Each provides a `FileIOAdapter` and `SystemAdapter` plus output callbacks, then calls the core's `executeCode` function.
- **Native addon (optional)** — a Node-API C++ module wrapping LAPACK/OpenBLAS, FFTW, and selected element-wise kernels. Numbl detects it at runtime and falls back to pure-JS implementations when absent.

## Execution pipeline

1. **Lex** — source text to tokens.
2. **Parse** — tokens to an AST.
3. **Lower** — `LoweringContext` indexes workspace files and resolves function names, classes, and search paths. It caches parsed ASTs and stores per-file metadata (directives, external access).
4. **Interpret** — the `Interpreter` walks the AST against a `Runtime` and an `Environment` stack. Control flow is signalled by typed exceptions (break, continue, return).
5. **JIT (optional)** — hot user functions, hot loops, and feasible top-level scripts are specialized. The JIT lowers AST to a typed IR, then emits JS (`--opt 1`). Under `--opt 2`, named C-JIT optimizers compete with the JS-JIT executors via the executor registry. Runtime type mismatches bail back to the interpreter.
6. **Runtime values** — results are `RuntimeValue`s (numbers, tensors, strings, structs, cells, class instances, etc.). Output, plot instructions, and workspace changes flow back through callbacks on `ExecOptions`.

## Component map

| Component        | Responsibility                                                                   |
| ---------------- | -------------------------------------------------------------------------------- |
| Lexer            | Tokenize `.m` source, handling MATLAB's whitespace- and context-sensitive rules. |
| Parser           | Produce the AST (statements and expressions).                                    |
| LoweringContext  | Workspace/function/class resolution; AST cache; directive metadata.              |
| Interpreter      | AST walker; drives builtin dispatch and JIT triggers.                            |
| Runtime          | Value constructors, global/persistent stores, plot accumulator, output router.   |
| JIT              | Type-specializes hot code to JS (and to C under `--opt 2`).                      |
| Builtin registry | `IBuiltin` lookup by name; provides type resolution and JIT emission.            |
| Ops layer        | Dispatches tensor kernels to the native addon or JS fallbacks.                   |
| Stdlib bundle    | `.m` files loaded into every run's workspace.                                    |
| Platform adapter | Implements file I/O, system info, and output/plot callbacks for one environment. |

## Data flow

- **Source → AST → RuntimeValue** is the critical path.
- **Side channels**: output text, plot instructions, and diagnostics go out through callbacks on `ExecOptions`. Input (e.g., `input()`) comes back through a platform-provided callback — in the browser worker this is bridged through a `SharedArrayBuffer` so it can be synchronous.
- **JIT compilation events** are reported through a separate callback, so tooling can surface what was specialized and why.

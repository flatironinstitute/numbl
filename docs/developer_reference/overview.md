# Numbl Overview

Numbl is a numerical computing environment compatible with MATLAB syntax, written in TypeScript. It runs in Node (CLI, REPL, execution server) and in the browser (web IDE, embeddable worker). A program is a set of `.m` files that are parsed, interpreted, and optionally JIT-specialized to JavaScript (optionally with inline compiled C kernels).

## Execution pipeline

```
source .m ──► Lexer ──► Parser ──► AST
                                    │
                                    ▼
                           LoweringContext
                     (workspace / function resolver)
                                    │
                                    ▼
                            Interpreter
                      (AST walker + Runtime)
                                    │
                   hot path?        │
              ┌─────────────────────┘
              ▼
           JS-JIT
          (opt 1)
              │   (under opt 2: C-JIT executors compete
              │    with the JS-JIT set via the executor
              │    registry; both are registered)
              │
              ▼
           RuntimeValue
```

A single entry point (the `executeCode` function) accepts source, options, and adapters (file I/O, system, output callbacks) and drives this pipeline. Every surface — CLI, web worker, execution server — goes through it.

## Topic guide

- [architecture.md](architecture.md) — the pipeline and how components connect.
- [compiler/lexer-parser.md](compiler/lexer-parser.md) — tokenization and parsing.
- [compiler/interpreter.md](compiler/interpreter.md) — the AST walker and workspace resolution.
- [compiler/type-system.md](compiler/type-system.md) — `JitType`, unification, type refinement.
- [executors.md](executors.md) — pluggable executor registry that unifies the interpreter and JIT dispatch hooks.
- [jit/overview.md](jit/overview.md) — when the JIT runs, opt levels, bailouts.
- [jit/ir-codegen.md](jit/ir-codegen.md) — JIT IR and the JS backend.
- [jit/fusion.md](jit/fusion.md) — element-wise fusion.
- [jit/cjit.md](jit/cjit.md) — C-JIT (`--opt 2`): the C backend and how it shares the JS-JIT compiler.
- [runtime/values-and-tensors.md](runtime/values-and-tensors.md) — `RuntimeValue`, tensors, memory layout.
- [runtime/refcount.md](runtime/refcount.md) — refcount-driven pool reclamation, slot ownership, transient scopes.
- [runtime/native-addon.md](runtime/native-addon.md) — LAPACK/FFTW bindings and JS fallbacks.
- [builtins.md](builtins.md) — the `IBuiltin` registry, resolution, JIT emission.
- [stdlib.md](stdlib.md) — `.m`-defined standard library and bundle generation.
- [cli.md](cli.md) — CLI commands, options, environment variables.
- [web-app.md](web-app.md) — browser IDE, worker, VFS, blocking input.
- [browser-embedding.md](browser-embedding.md) — `numbl/browser` managed sessions and the raw embedding primitives.
- [server.md](server.md) — optional remote execution service.
- [plotting.md](plotting.md) — graphics instructions and the plot viewer.
- [uihtml.md](uihtml.md) — interactive HTML figures and the two-way data/event bridge to the interpreter.
- [testing.md](testing.md) — unit and integration test suites.
- [diagnostics-and-extensions.md](diagnostics-and-extensions.md) — error reporting, external-access directives, JS user functions.

When adding a new topic file, link it from this list.

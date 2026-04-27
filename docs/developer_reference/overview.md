# Numbl Overview

Numbl is a MATLAB-compatible numerical computing environment written in TypeScript. It runs in Node (CLI, REPL, execution server) and in the browser (web IDE, embeddable worker). A program is a set of `.m` files that are parsed, interpreted, and optionally JIT-specialized to JavaScript (optionally with inline compiled C kernels).

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
        (opt 1 or e1)
              │   (under opt e1: splice in compiled C
              │    kernels for fusible chains and
              │    pure-scalar user functions)
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
- [jit/ir-codegen.md](jit/ir-codegen.md) — JIT IR and the JS backend / inline C kernels.
- [jit/fusion.md](jit/fusion.md) — element-wise fusion.
- [jit/e1-kernels.md](jit/e1-kernels.md) — e1 kernel pipeline (chain kernels + scalar-function kernels).
- [jit/e2-kernels.md](jit/e2-kernels.md) — e2 per-assign C-kernel pipeline (interpreter outer, no JS-JIT).
- [runtime/values-and-tensors.md](runtime/values-and-tensors.md) — `RuntimeValue`, tensors, memory layout.
- [runtime/native-addon.md](runtime/native-addon.md) — LAPACK/FFTW bindings and JS fallbacks.
- [builtins.md](builtins.md) — the `IBuiltin` registry, resolution, JIT emission.
- [stdlib.md](stdlib.md) — `.m`-defined standard library and bundle generation.
- [cli.md](cli.md) — CLI commands, options, environment variables.
- [web-app.md](web-app.md) — browser IDE, worker, VFS, blocking input.
- [server.md](server.md) — optional remote execution service.
- [plotting.md](plotting.md) — graphics instructions and the plot viewer.
- [testing.md](testing.md) — unit and integration test suites.
- [diagnostics-and-extensions.md](diagnostics-and-extensions.md) — error reporting, external-access directives, JS user functions.

When adding a new topic file, link it from this list.

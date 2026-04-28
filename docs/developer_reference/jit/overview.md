# JIT Overview

The JIT sits on top of the interpreter. It type-specializes hot code paths to JavaScript (run in V8). The interpreter remains authoritative: anything the JIT cannot handle falls back to the interpreter transparently.

## Trigger points

Three entry points decide whether to JIT:

- **User function call** — the first call of a user-defined function with stable argument types.
- **Hot loop** — a `for` or `while` body after the interpreter sees enough iterations.
- **Top-level script** — a whole script body when the JIT judges it feasible.

All three share the same lowering pipeline and IR. Specializations are cached keyed on argument type signatures; a different signature compiles a new variant.

## Optimization levels (`--opt`)

- `--opt 0` — interpreter only. No JIT.
- `--opt 1` — **JS-JIT** (default). Lowers AST → JIT IR → JavaScript source, materialized via `new Function(...)`. Fast to compile, runs in-process.
- `--opt 2` — JS-JIT plus C-JIT optimizers (Node only). A registry of named C-codegen optimizers compete with the JS-JIT executors for each dispatch; the lowest-cost candidate wins. (Implementation in progress; see [executors.md](../executors.md).)

## Bailouts

When a specialized function runs, each operation implicitly assumes the types from compile time. If a runtime value is wrong (an integer turns into NaN, a real array receives a complex assignment, a struct field changes shape), a bail exception unwinds the JIT call and the interpreter continues from the call site. The cached compilation is invalidated so a fresh type-gathering pass can produce a new variant.

A separate safety pass classifies constructs that have observable side effects the JIT cannot faithfully reproduce (interactive input, certain display statements at the top level, etc.) and suppresses JIT for scripts that contain them unsuppressed.

## Debugging JIT output

`--dump-js <file>` writes the generated JavaScript source to disk for inspection. `--verbose` adds compilation events to stderr.

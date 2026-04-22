# JIT Overview

The JIT sits on top of the interpreter. It type-specializes hot code paths to either JavaScript (run in V8) or C (compiled and loaded as a Node addon). The interpreter remains authoritative: anything the JIT cannot handle falls back to the interpreter transparently.

## Trigger points

Three entry points decide whether to JIT:

- **User function call** — the first call of a user-defined function with stable argument types.
- **Hot loop** — a `for` or `while` body after the interpreter sees enough iterations.
- **Top-level script** — a whole script body when the JIT judges it feasible.

All three share the same lowering pipeline and IR. Specializations are cached keyed on argument type signatures; a different signature compiles a new variant.

## Optimization levels (`--opt`)

- `--opt 0` — interpreter only. No JIT.
- `--opt 1` — **JS-JIT**. Lowers AST → JIT IR → JavaScript source, materialized via `new Function(...)`. Fast to compile, runs in-process.
- `--opt 2` — **C-JIT**. In addition to JS-JIT, attempts to emit C for IR that falls inside the C backend's supported subset. Compiles with `cc`, loads as a `.node` addon, calls through Node-API. Infeasible IR falls back to JS-JIT.

C-JIT feasibility is checked before any C emission; complex tensors, string ops, struct/cell manipulation, and many builtins are JS-only. A hybrid mode splices native callees into a JS outer wrapper when only part of a function is C-feasible.

## Bailouts

When a specialized function runs, each operation implicitly assumes the types from compile time. If a runtime value is wrong (an integer turns into NaN, a real array receives a complex assignment, a struct field changes shape), a bail exception unwinds the JIT call and the interpreter continues from the call site. The cached compilation is invalidated so a fresh type-gathering pass can produce a new variant.

A separate safety pass classifies constructs that have observable side effects the JIT cannot faithfully reproduce (interactive input, certain display statements at the top level, etc.) and suppresses JIT for scripts that contain them unsuppressed.

## Debugging JIT output

The CLI's `--dump-js <file>` and `--dump-c <file>` flags write the generated source to disk for inspection. `--verbose` adds compilation events to stderr. These are the two primary tools for diagnosing a JIT issue.

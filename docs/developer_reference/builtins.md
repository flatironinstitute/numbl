# Builtins

Built-in functions (numbl's `IBuiltin`s) are the implementations of MATLAB's non-language-level functions: arithmetic helpers, array construction, linear algebra, FFT, string ops, predicates, and so on. They live in a single in-process registry keyed by name.

## Two complementary paths

Each builtin is consulted in two ways:

1. **Interpreter path** — the interpreter calls `resolve(argTypes, nargout)`, receives a specialized `apply` function, and invokes it on the runtime values. This path always works.
2. **JIT path** — when the JIT is lowering a call site, it uses the same `resolve` to propagate output types. For JS codegen, it prefers the builtin's optional `jitEmit` (an inline JavaScript expression). If `jitEmit` is missing or returns `null` for the given argument types, the JIT emits a trampoline call into the runtime apply function (`$h.ib_<name>`). The C-JIT path (executors under `cJit/`) uses its own whitelist and codegen rather than a per-builtin C-emission hook.

This split lets a builtin have a sophisticated general implementation while providing a tight fast path for common cases.

## Interface sketch

- `name` — the function name as seen by user code.
- `help` — structured metadata (signatures, description) surfaced by the `help` command. Required for user-facing builtins.
- `resolve(argTypes, nargout)` — decides whether this builtin accepts the given argument types. Returns `{ outputTypes, apply }` on success or `null` to reject (falling back to the next lookup or erroring).
- `jitEmit(argCode, argTypes)` — optional; returns a JS expression string or `null` to skip.

## Registration helpers

Common patterns have shared helpers so a new builtin often needs only a few lines:

- unary element-wise math (real-input, real-result or complex-if-needed);
- unary real-result math (e.g., absolute value);
- binary scalar functions;
- the corresponding `jitEmit` helpers that produce the right inline JS.

More complex builtins (multiple outputs, shape-dependent behavior, custom dispatch) implement `resolve` directly.

## Type responsibilities

`resolve` is responsible for:

- rejecting types the builtin does not support (return `null`, letting the interpreter either error or try another path);
- returning accurate output `JitType`s — the JIT propagates these to callers and will miscompile if they are wrong;
- respecting integer, logical, complex, and shape refinements where possible, so downstream callers can be specialized too.

## Where "special" builtins live

A small set of names are handled by the interpreter itself rather than as `IBuiltin`s — I/O, display, plotting commands, and similar side-effecting constructs. These are maintained in a dedicated list so the interpreter knows to short-circuit the registry lookup. When in doubt whether a name is special, grep for it.

# Interpreter

The interpreter is an AST walker. Given a parsed program and a `LoweringContext`, it evaluates statements in order against a `Runtime` and produces `RuntimeValue`s.

## Key objects

- **Interpreter** — holds the `Runtime` and the active `LoweringContext`. Walks `Stmt` and `Expr` nodes.
- **Environment** — one per activation (script, function call, class method). Resolves variable reads and writes. Routes `global` and `persistent` declarations through shared stores on the runtime so their values outlive the call.
- **Runtime** — shared state across a single execution: global/persistent stores, output router, plot instruction accumulator, RNG state, JIT caches.
- **LoweringContext** — workspace resolver. Indexes `.m` files by function name, maintains per-file and per-class sub-contexts, caches parsed ASTs, and stores external-access directive metadata. A shared `WorkspaceRegistry` is carried by reference across derived contexts so a single workspace view is visible everywhere.

## Control flow

Control-flow transfers (break, continue, return) are signalled by throwing typed signal objects that the enclosing loop or function body catches. Regular runtime errors flow as normal exceptions and end up in the diagnostics layer.

## Builtin dispatch

When the interpreter encounters a call expression:

1. Look up the name in the local workspace (user-defined function).
2. Otherwise, look it up in the `IBuiltin` registry.
3. Otherwise, look it up in the stdlib-bundled functions.
4. Otherwise, error.

Builtin calls go through the builtin's `resolve(argTypes, nargout)` to produce a specialized `apply` function, which the interpreter invokes. See [builtins.md](../builtins.md).

## JIT integration

The interpreter decides when to hand control to the JIT:

- **User function calls** are trial-compiled on first hot-enough call.
- **`for` / `while` loops** are candidates when iteration counts look worthwhile.
- **Top-level scripts** are compiled whole when feasible.

If the JIT succeeds, the specialized function or loop runs in place of the interpreted version. A type mismatch at runtime throws a bail signal, the interpreter resumes, and the JIT cache is invalidated.

## What the interpreter does not do

- It does not type-specialize. Every operation goes through generic `RuntimeValue` dispatch unless a JIT path takes over.
- It does not do parse-time name resolution for user functions; the `LoweringContext` does that on demand.
- It does not manage platform I/O directly — all file, stdout/stderr, and plot output flows through callbacks on `ExecOptions`.

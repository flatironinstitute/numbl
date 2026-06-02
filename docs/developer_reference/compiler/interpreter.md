# Interpreter

The interpreter is an AST walker. Given a parsed program and a `LoweringContext`, it evaluates statements in order against a `Runtime` and produces `RuntimeValue`s.

## Key objects

- **Interpreter** — holds the `Runtime` and the active `LoweringContext`. Walks `Stmt` and `Expr` nodes.
- **Environment** — one per activation (script, function call, class method). Resolves variable reads and writes. Routes `global` and `persistent` declarations through shared stores on the runtime so their values outlive the call.
- **Runtime** — shared state across a single execution: global/persistent stores, output router, plot instruction accumulator, JIT caches. (RNG state is held in module-level variables in [`helpers/prng.ts`](../../../src/numbl-core/helpers/prng.ts), not on the `Runtime`.)
- **LoweringContext** — workspace resolver. Indexes `.m` files by function name, maintains per-file and per-class sub-contexts, caches parsed ASTs, and stores external-access directive metadata. A shared `WorkspaceRegistry` is carried by reference across derived contexts so a single workspace view is visible everywhere.

## Control flow

Control-flow transfers (break, continue, return) are signalled by typed signal objects (`BreakSignal` / `ContinueSignal` / `ReturnSignal`, none of which extend `Error`) that `execStmt` **returns** (its type is `ControlSignal | null`); the enclosing loop or function body inspects the return value with `instanceof` and propagates it upward. They are not thrown. Regular runtime errors, by contrast, flow as normal exceptions and end up in the diagnostics layer.

## Builtin dispatch

When the interpreter encounters a call expression, name resolution (`resolveFunctionImpl` in [`functionResolve.ts`](../../../src/numbl-core/functionResolve.ts)) tries, in order:

1. The local workspace — user-defined functions. **The stdlib bundle is part of this tier**: bundled stdlib `.m` files are registered as workspace functions, so a stdlib function (and any user `.m` of the same name) is found here, _before_ the builtin registry. This means a workspace/stdlib `.m` shadows a builtin of the same name.
2. JS user functions and workspace classes.
3. The `IBuiltin` registry (and the runtime's built-in table).
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

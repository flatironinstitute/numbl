# Interpreter

The interpreter is an AST walker. Given a parsed program and a `CodeIndex`, it evaluates statements in order against a `Runtime` and produces `RuntimeValue`s.

## Key objects

- **Interpreter** — holds the `Runtime` and the run's `CodeIndex`. Walks `Stmt` and `Expr` nodes.
- **Environment** — one per activation (script, function call, class method). Resolves variable reads and writes. Routes `global` and `persistent` declarations through shared stores on the runtime so their values outlive the call.
- **Runtime** — shared state across a single execution: global/persistent stores, output router, plot instruction accumulator, JIT caches. (RNG state is held in module-level variables in [`helpers/prng.ts`](../../../src/numbl-core/helpers/prng.ts), not on the `Runtime`.)
- **CodeIndex** — owned by the `Workspace` (the visible file set, kept in sync with addpath/rmpath/cd). Caches parsed ASTs, maps workspace files to function names (`+pkg` namespaces, `private/` scopes, `.numbl.js` functions), holds the main file's local functions, and builds the `FunctionIndex` that `resolveFunction` consults. The interpreter and the JIT share the same instance.
- **ClassTable** — part of the `CodeIndex`: `ClassInfo` for workspace and main-file classes, plus inheritance queries (lineage, ancestors, defining class, handle-ness).

## Control flow

Control-flow transfers (break, continue, return) are signalled by typed signal objects (`BreakSignal` / `ContinueSignal` / `ReturnSignal`, none of which extend `Error`) that `execStmt` **returns** (its type is `ControlSignal | null`); the enclosing loop or function body inspects the return value with `instanceof` and propagates it upward. They are not thrown. Regular runtime errors, by contrast, flow as normal exceptions and end up in the diagnostics layer.

## Name resolution and calls

A call expression or bare name is resolved by `resolveName` (see [function-resolution.md](function-resolution.md)) from the interpreter's current `Scope`: the running file, class and method, and the current `Environment`. The result says what the name is (variable, constant, nested, local, private or workspace function, class method or constructor, package function, builtin, ...), and `callResolved` runs it. A user function's `FunctionDef` comes from `functionBodyFor`, which also gives the file and class context to run it in. A workspace or stdlib `.m` shadows a builtin of the same name.

Builtin calls go through the builtin's `resolve(argTypes, nargout)` to produce a specialized `apply` function, which the interpreter invokes. See [builtins.md](../builtins.md).

## JIT integration

The interpreter decides when to hand control to the JIT:

- **User function calls** are trial-compiled on first hot-enough call.
- **`for` / `while` loops** are candidates when iteration counts look worthwhile.
- **Top-level scripts** are compiled whole when feasible.

If the JIT succeeds, the specialized function or loop runs in place of the interpreted version. A type mismatch at runtime throws a bail signal, the interpreter resumes, and the JIT cache is invalidated.

## What the interpreter does not do

- It does not type-specialize. Every operation goes through generic `RuntimeValue` dispatch unless a JIT path takes over.
- It does not do parse-time name resolution for user functions; calls are resolved on demand (and cached per function index).
- It does not manage platform I/O directly — all file, stdout/stderr, and plot output flows through callbacks on `ExecOptions`.

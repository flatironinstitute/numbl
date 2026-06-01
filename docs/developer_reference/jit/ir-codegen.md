# JIT IR and Codegen

The JIT compiler is the in-tree subsystem under `src/numbl-core/jit`. It lowers the parser AST to a typed intermediate representation, then emits target code — JavaScript (`--opt 1`) or C (`--opt 2`) — from the same IR.

## IR

`jit/lowering/ir.ts` defines the IR: discriminated-union expression and statement nodes, each carrying its resolved `Type` (`jit/lowering/types.ts`). Expression kinds include numeric / imaginary / string literals, variables, binary/unary ops, calls (user-function specialization, builtin, runtime helper), tensor build, index load / slice, member load, and cell / struct / handle nodes. Statement kinds include assign, indexed / sliced / member / cell stores, `if` / `while` / `for`, multi-assign, and control flow. Types are fixed at lowering time — codegen never re-decides what an operation means.

## Lowering

`jit/lowering/lower.ts` (the `Lowerer`) walks the AST top-down, tracking variable types per scope and resolving callees through the workspace (`jit/workspace/workspace.ts`, which delegates to numbl's `resolveFunction`). `specialize.ts` produces one `IRFunc` per `(function, arg-type signature, nargout)` and caches it in `Lowerer.specializations`, so repeated calls with the same signature reuse the lowering. A construct the IR can't represent throws `UnsupportedConstruct` / `TypeError` (`jit/lowering/errors.ts`); the executor catches it and declines to the interpreter.

## Backends

Both backends walk the same IR program — `compileSpec` → `jit/codegen/emitJs.ts` for JS; `compileSpecC` → `jit/codegen/emit.ts` for C:

- **JS** emits an ES module string returning `($h) => specFn`, materialized with `new Function(...)`. `$h` carries the host hooks (output sink, plot dispatch).
- **C** emits C source compiled to a `.so` with `cc` and called via koffi; multi-output specs use out-pointer params (see `executors/jit/typeAdapterC.ts`).

Builtins emit through the JIT compiler's own registry (`jit/builtins`): each def supplies `emitJs` / `emitC`. **Runtime snippets** (`jit/builtins/runtime/*.h` C bodies + `*.js` siblings) are generated into `snippets.gen.ts` by `scripts/build_runtime_snippets.ts` and inlined into the emitted module **lazily** — only snippets activated by the emitted code are included.

See [executors.md](../executors.md) for how the executors invoke `compileSpec` / `compileSpecC` and marshal values across the boundary, and [cjit.md](cjit.md) for the C backend specifics.

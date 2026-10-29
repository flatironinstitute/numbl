# JIT Overview

The JIT sits on top of the interpreter. It type-specializes code to JavaScript (run in V8) or, at `--opt 2`, to C compiled to a native `.so` and called via koffi. The interpreter remains authoritative: anything the JIT cannot handle falls back to the interpreter transparently.

The JIT compiler is an in-tree, self-contained subsystem under `src/numbl-core/jit` (lowering → IR → JS/C codegen, builtins, runtime snippets, workspace) with **no external dependency**. Its per-spec entry points, `compileSpec` for JS ([`jit/compileSpec.ts`](../../../src/numbl-core/jit/compileSpec.ts)) and `compileSpecC` for C ([`jit/compileSpecC.ts`](../../../src/numbl-core/jit/compileSpecC.ts)), both re-exported from `jit/index.ts`, are called by the per-shape executors under `src/numbl-core/executors/jit/`; see [executors.md](../executors.md). Both run one pipeline, `compileWith` ([`jit/compile.ts`](../../../src/numbl-core/jit/compile.ts)), with their `CompileTarget`: it specializes the entry, collects the specs it transitively calls (`reachableCallees`), runs the IR passes the target asks for (`definiteAssignment`: the definite-assignment check and assigned flags, C only; then, for both, the kernel pass with the target's `fusion` policy, which turns element-wise chains into `ElemwiseKernel` nodes, see [fusion.md](fusion.md)), and calls the target's emitter. The JIT began as a separate project (mtoc2) and was later vendored into numbl. For the IR, the backends, the runtime snippets and the steps for adding a builtin, see [ir-codegen.md](ir-codegen.md).

## Trigger points

The interpreter offers three kinds of unit to the executors, with no hotness or iteration-count threshold: a whole script body (once, at script start), each `for`/`while` loop it reaches, and each user-function call (see [executors.md](../executors.md)). All three share the same lowering pipeline and IR. Specializations are cached by input types (widened across calls, see [type-system.md](../compiler/type-system.md#executor-inputs)); a different signature compiles a new variant.

## Optimization levels (`--opt`)

- `--opt 0`: interpreter only. No JIT.
- `--opt 1`: **JS-JIT** (default). Lowers AST → JIT IR → JavaScript source, materialized via `new Function(...)`. Fast to compile, runs in-process.
- `--opt 2`: **C-JIT** (Node only). Lowers the same IR to C, compiles a `.so` with `cc`, and calls it via koffi. The C-JIT executors (`cjit-top-level`, `cjit-loop`, `cjit-call`) are tried before the JS-JIT set; C-JIT takes what it can marshal, JS-JIT picks up the rest. Requires `cc` + `koffi`; otherwise collapses to JS-JIT. See [executors.md](../executors.md) and [cjit.md](cjit.md).

## Static decline (vs. runtime bailout)

The compiler decides feasibility at **compile time**: it either lowers a specialization cleanly or throws `UnsupportedConstruct` / `JitTypeError`, which the executor catches and declines: dispatch then falls through to the interpreter. There is no per-operation runtime type-guard bailout. A runtime error raised by the emitted code goes through the raise channel: it becomes numbl's `RuntimeError` with the interpreter's message and identifier, or, when the JIT cannot reproduce the result exactly, the interpreter re-runs the unit to raise it (see [executors.md](../executors.md#runtime-errors-and-bails)).

The top-level executor also declines scripts whose display output the emitted code does not reproduce (unsuppressed assignments; see [executors.md](../executors.md#shapes-and-backends)).

Indexed **stores** are the exception to "no runtime bailout": an index past a tensor's current extent would grow it in MATLAB, which a spec compiled against a fixed shape cannot model, so the emitted store routes through a grow-aware bounds check (`jit_idx_*_grow`) that raises a bail to the interpreter at that point. Both scalar stores (`v(k) = x`) and the scalar slots of slice stores (`A(k, :) = row`) use it, which is what lets a store with a loop-variable index compile at all: the index is never statically known inside a loop.

### Decline diagnostics

`Lowerer.specializations` is shared by every compile in the session, from both backends, but an emitted module contains only the entry spec and the specs it reaches, so a spec one compile left behind (say one the C backend emits but the JS emitter rejects) cannot make an unrelated compile fail. A failed compile also rolls back whatever it added to the map.

Set `NUMBL_JIT_LOG=1` to print every decline (executor, source position, message) to stderr, and every runtime error that re-runs on the interpreter (`[jit-rerun]`). `%!numbl:assert_jit` reports only the most recent one; the env var shows them all, which is what you want when a chain of calls declines for several different reasons.

## Debugging JIT output

`--dump-js <file>` writes the generated JavaScript source to disk for inspection; `--dump-c <file>` writes the generated C source. Each flag captures only its backend's output, so at `--opt 2` (where C-JIT and JS-JIT both run) you can pass both: C kernels go to the `--dump-c` file and the JS-JIT fallbacks go to the `--dump-js` file.

Each section is written the moment it compiles, so the dump reflects everything generated up to a failure.

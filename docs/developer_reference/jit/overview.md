# JIT Overview

The JIT sits on top of the interpreter. It type-specializes hot code paths to JavaScript (run in V8) or, at `--opt 2`, to C compiled to a native `.so` and called via koffi. The interpreter remains authoritative: anything the JIT cannot handle falls back to the interpreter transparently.

The JIT compiler is an in-tree, self-contained subsystem under `src/numbl-core/jit` (lowering → IR → JS/C codegen, builtins, runtime snippets, workspace) with **no external dependency**. Its per-spec entry points — `compileSpec` for JS ([`jit/compileSpec.ts`](../../../src/numbl-core/jit/compileSpec.ts)) and `compileSpecC` for C ([`jit/compileSpecC.ts`](../../../src/numbl-core/jit/compileSpecC.ts)), both re-exported from `jit/index.ts` — are called by the per-shape executors under `src/numbl-core/executors/jit/`; see [executors.md](../executors.md).

## Trigger points

Three entry points decide whether to JIT:

- **User function call** — the first call of a user-defined function with stable argument types.
- **Hot loop** — a `for` or `while` body after the interpreter sees enough iterations.
- **Top-level script** — a whole script body when the JIT judges it feasible.

All three share the same lowering pipeline and IR. Specializations are cached keyed on argument type signatures; a different signature compiles a new variant.

## Optimization levels (`--opt`)

- `--opt 0` — interpreter only. No JIT.
- `--opt 1` — **JS-JIT** (default). Lowers AST → JIT IR → JavaScript source, materialized via `new Function(...)`. Fast to compile, runs in-process.
- `--opt 2` — **C-JIT** (Node only). Lowers the same IR to C, compiles a `.so` with `cc`, and calls it via koffi. The C-JIT executors (`cjit-top-level`, `cjit-loop`, `cjit-call`) compete with the JS-JIT set via the cost model; C-JIT wins where it can marshal the types, JS-JIT picks up the rest. Requires `cc` + `koffi`; otherwise collapses to JS-JIT. See [executors.md](../executors.md) and [cjit.md](cjit.md).

## Static decline (vs. runtime bailout)

The compiler decides feasibility at **compile time**: it either lowers a specialization cleanly or throws `UnsupportedConstruct` / `JitTypeError`, which the executor catches and declines — dispatch then falls through to the interpreter. There is no per-operation runtime type-guard bailout; once a spec compiles it runs to completion. (The executor registry still supports a `{ bail }` path for hard runtime errors in emitted code, but the JIT executors report `bailRisk: false`.)

A separate safety pass classifies constructs with observable side effects the JIT cannot faithfully reproduce (interactive input, certain display statements at the top level, etc.) and suppresses JIT for scripts that contain them unsuppressed.

Indexed **stores** are the exception to "no runtime bailout": an index past a tensor's current extent would grow it in MATLAB, which a spec compiled against a fixed shape cannot model, so the emitted store routes through a grow-aware bounds check (`mtoc2_idx_*_grow`) that bails to the interpreter at that point. Both scalar stores (`v(k) = x`) and the scalar slots of slice stores (`A(k, :) = row`) use it, which is what lets a store with a loop-variable index compile at all — the index is never statically known inside a loop.

### Decline diagnostics

A failed `compileSpec` rolls back whatever it added to `Lowerer.specializations`. The emitted module contains **every** spec in that map, so a spec that lowers cleanly but throws during emit would otherwise be re-emitted — and rethrow — on every later compile in the session, silently stopping unrelated functions from being JIT'd.

Set `NUMBL_JIT_LOG=1` to print every decline (executor, source position, message) to stderr. `%!numbl:assert_jit` reports only the most recent one; the env var shows them all, which is what you want when a chain of calls declines for several different reasons.

## Debugging JIT output

`--dump-js <file>` writes the generated JavaScript source to disk for inspection; `--dump-c <file>` writes the generated C source. Each flag captures only its backend's output, so at `--opt 2` (where C-JIT and JS-JIT both run) you can pass both: C kernels go to the `--dump-c` file and the JS-JIT fallbacks go to the `--dump-js` file. `--verbose` adds compilation events to stderr.

Each section is written the moment it compiles, so the dump reflects everything generated up to a failure — including the case where compiled C aborts the process via `exit(1)` (e.g. an out-of-bounds index or `error(...)`), which terminates Node from inside the FFI call before any end-of-run cleanup could run.

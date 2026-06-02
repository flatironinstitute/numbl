# C-JIT

Orientation for `--opt 2` (Node only).

## What C-JIT is

C-JIT shares the **same compiler** as JS-JIT — same AST → lowering → IR pipeline (`src/numbl-core/jit`) — but emits C instead of JavaScript via `compileSpecC` (vs `compileSpec`). The C source is compiled to a shared object with `cc` and called through koffi. It handles the same three shapes as JS-JIT:

- **`cjit-top-level`** — whole (suppressed) script body.
- **`cjit-loop`** — outermost `for` / `while` body.
- **`cjit-call`** — user-function call.

At `--opt 2` both the C-JIT and JS-JIT executors are registered and compete via the dispatcher's cost model: C-JIT proposes only where it can marshal the argument types across the koffi ABI, and its lower per-call/run cost wins there; everything else falls through to JS-JIT, then the interpreter.

## Type marshaling boundary

What C-JIT can accept is gated by `executors/jit/typeAdapter.ts` (numbl `JitType` → compiler `Type`), `typeAdapterC.ts` (compiler `Type` → C decl, via `compilerTypeToCDecl`), and `valueAdapterC.ts` (`RuntimeValue` ↔ C ABI via koffi). Scalars (real double, logical) pass as `double`; tensors pass as the `mtoc2_tensor_t` struct (real/imag pointers + dims). Types the adapter can't marshal cause the executor to decline — the spec still lowers, but the call routes to JS-JIT/interpreter instead.

## Native compile/load

The `cc`-invoke + koffi `dlopen` step lives in `executors/jit/compileC.node.ts` (Node only; uses `node:fs`/`node:child_process`). It is wired in behind a browser-safe stub (`compileC.ts`) via `registerNodeCompileC()` from `cli.ts` at bootstrap. Compiled `.so`s are cached on disk keyed by a hash of the flags + C source. Without `cc` or `koffi`, the C-JIT executors decline and `--opt 2` behaves like `--opt 1`.

## When it helps

C-JIT beats JS-JIT for code where dispatch is statically eliminated, leaving raw arithmetic on contiguous memory — scalar numeric loops and dense element-wise tensor work. Outside that regime, V8's inline caches and escape analysis make JS-JIT competitive, and the cost model lets JS-JIT win. The compile + `dlopen` cost is paid once per `(function, arg-type signature, nargout)` and amortized across calls.

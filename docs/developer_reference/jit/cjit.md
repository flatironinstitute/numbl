# C-JIT

Orientation for `--opt 2` (Node only).

## What C-JIT is

C-JIT shares the **same compiler** as JS-JIT (the same AST → lowering → IR pipeline in `src/numbl-core/jit`) but emits C instead of JavaScript via `compileSpecC` (vs `compileSpec`). The C source is compiled to a shared object with `cc` and called through koffi. It handles the same three shapes as JS-JIT:

- **`cjit-top-level`**: whole (suppressed) script body.
- **`cjit-loop`**: outermost `for` / `while` body.
- **`cjit-call`**: user-function call.

At `--opt 2` both the C-JIT and JS-JIT executors are registered, C-JIT first. C-JIT proposes only where it can marshal the argument types across the koffi ABI and is then always preferred; everything else falls through to JS-JIT, then the interpreter. The two share the shape executors and differ only in their `JitBackend` (`executors/jit/cBackend.ts` vs `jsBackend.ts`).

## Type marshaling boundary

What C-JIT can accept is gated by the input types (`typeOfValue` in executor classification, runtime value → compiler `Type`), `typeAdapterC.ts` (compiler `Type` → C decl, via `compilerTypeToCDecl`), and `valueAdapterC.ts` (`RuntimeValue` ↔ C ABI via koffi). Scalars (real double, logical) pass as `double`; tensors pass as the `jit_tensor_t` struct (real/imag pointers + dims). Types the adapter can't marshal cause the executor to decline: the spec still lowers, but the call routes to JS-JIT/interpreter instead. The callee owns its tensor params, so input data is copied into buffers from the `.so`'s exported `jit_malloc`; returned tensors are copied out and released with `jit_free`. The exception is a tensor parameter the entry only reads (see [Borrowed inputs](../executors.md#borrowed-inputs)): when it has at least 1024 elements, the host passes a pointer to its own buffer, and the emitter leaves that parameter out of the entry's frees.

A loop or top-level wrapper may leave an output unassigned (the loop variable and body locals of a zero-trip `for`). C has no unassigned state, so the compile pipeline (for a target with `definiteAssignment`) adds a flag output for each output that is not definitely assigned (`lowering/assignedFlags.ts`), and the host skips writing back an output whose flag is still 0.

## Runtime errors: the raise channel

Every runtime error or bail in compiled C goes through `jit_raise(kind, identifier, message)` (`runtime/system/raise.h`, emitted first in every `.so`). Nothing calls `exit` or `abort`.

- **Guard.** The host does not call the entry spec directly but an exported wrapper, `<spec>__host` (`emitHostEntry` in `codegen/emit.ts`), which clears the status, arms a `setjmp` guard, and calls the spec. The spec is `static __attribute__((noinline))`, so the `setjmp` (which makes the compiler spill values live across it) stays out of the hot body, and a recursive spec does not re-arm the guard. The guard costs one `setjmp` per call, which is small next to the koffi call itself.
- **Raise.** `jit_raise` records the kind (`JIT_RAISE_BAIL`, `JIT_RAISE_ERROR` or `JIT_RAISE_CANCEL`), the identifier, the message (NULL for an error whose wording the JIT leaves to the interpreter) and the current statement location, then `longjmp`s to the guard, which returns a zeroed value. The C backend reads the status after each call (`bindRaise`: `jit_raise_kind` / `_id` / `_msg` / `_loc`) and throws the normalized `JitRaise`; see [executors.md](../executors.md#runtime-errors-and-bails) for how it is mapped. Helpers: `jit_raise_error`, `jit_raise_errorf`, `jit_raise_dims_mismatch` (exact interpreter wording), `jit_raise_error_deferred` (wording left to the interpreter), `jit_raise_bail` / `jit_raise_bailf`.
- **Location.** Emitted code stores the current source statement in `jit__loc` at the start of each statement (`emitBody`), as the interpreter sets `rt.$line`. The value packs a file index and an offset (`locCode` / `decodeLoc` in `codegen/runtime.ts`); the compile result carries the file table (`locFiles`). The cost is one store per statement.
- **Memory.** A raise abandons the frames between it and the guard together with the tensors they own. Every allocation therefore goes through a tracked allocator: `raise.h` defines `malloc` / `calloc` / `realloc` / `free` as macros for versions that link each block (16-byte header) into one live list, and the guard frees every live block (`jit__free_all`). Between calls the list holds only returned output buffers, which the host frees with `jit_free`; input buffers the host allocates with `jit_malloc` belong to the callee and are freed by the guard on a raise. A borrowed input buffer is not in the list, so the guard leaves it alone. A raise therefore leaks nothing.

- **Cancel.** `jit__cancel` points at the host's cancel flag, a `SharedArrayBuffer` word the IDE's main thread sets on Stop (or at a constant zero). Emitted code polls it with `JIT_POLL_CANCEL()` at the top of every loop iteration (every 1024 iterations in a strip-mined constant-time loop) and at the entry of functions that call specs: a relaxed atomic load and a never-taken branch to `jit_raise_cancel`, which raises `JIT_RAISE_CANCEL` through the guard like any other raise, so nothing leaks. The host installs the flag with the exported `jit_set_cancel` (`bindCancel` in `valueAdapterC.ts`) before a call whenever it changed; koffi passes a pointer to the buffer's backing store, which never moves, and the host keeps the view alive. See [executors.md](../executors.md#cancellation).

Because errors do not end the process, C-JIT also runs inside `try`.

Program output reaches the host through a koffi callback while the call runs, so it streams (see [executors.md](../executors.md#output-streaming)). That callback may run user code (`onOutput`); a `.so`'s file-scope raise and allocator state makes it non-reentrant, so the backend bails a call into a `.so` that is already running. Large matrix products also call back into the host, through the LAPACK provider stack, so they match the interpreter bit for bit (see [native-addon.md](../runtime/native-addon.md#matrix-products-across---opt-levels)).

## Native compile/load

The `cc`-invoke + koffi `dlopen` step lives in `executors/jit/compileC.node.ts` (Node only; uses `node:fs`/`node:child_process`). `cBackend.ts` calls it directly; the backend reaches a run through its host (`host.cjit`, set from `nodeCJitBackend()` in `cjit.node.ts` by `initNodePlatform` in `src/node-platform.ts`), and `compileC.ts` keeps only the types. Compiled `.so`s are cached on disk (`~/.cache/numbl/c-jit`, or `$NUMBL_CJIT_CACHE`) keyed by a hash of the flags, the koffi declaration and the C source without its comments. The emitter names a module's specs by their position in call order from the entry (`_jit_fn<k>_<name>`, see `codegen/cSource.ts`) rather than by the mangled `<name>__<hash of argument types>`, so a widened spec whose code is the same as the exact-shape spec it replaced (only the type comments differ) loads the cached object instead of running `cc` again. Without `cc` or `koffi`, the C-JIT executors decline and `--opt 2` behaves like `--opt 1`.

## When it helps

C-JIT beats JS-JIT for code where dispatch is statically eliminated, leaving raw arithmetic on contiguous memory: scalar numeric loops and dense element-wise tensor work. Outside that regime, V8's inline caches and escape analysis make JS-JIT competitive. Note that the registry does not weigh this: whenever C-JIT accepts a unit it is used, so a tiny function called many times from interpreted code pays the koffi marshaling cost on every call. The compile + `dlopen` cost is paid once per `(function, arg-type signature, nargout)` and amortized across calls.

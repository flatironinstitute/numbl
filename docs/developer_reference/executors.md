# Executor Registry

The interpreter delegates three kinds of unit to a registry of _executors_: a whole script body, a `for`/`while` loop, and a user-function call. Each executor handles one of these _shapes_ with one JIT backend (JS-JIT at `--opt 1`; C-JIT, then JS-JIT, at `--opt 2`). The AST-walking interpreter is the registry's hardcoded last-resort fallback; it is not a registered executor.

## Concepts

- **Shape**: the kind of unit, one of `top-level`, `loop`, or `call`. The registry lowers a unit once (`executors/lowering.ts`) into a `LoweredStmt` tagged with its shape and hands it to the executors registered for that shape.
- **Executor**: one strategy for one shape. Implements the `Executor` interface below.
- **Registry**: holds an ordered list of executors per shape, dispatches at runtime, and owns the compile cache.
- **Backend**: what differs between JS-JIT and C-JIT (`executors/jit/backend.ts`). The three shape executors are written once against a `JitBackend` and instantiated per backend.
- **Cache key**: per-executor projection of the proposal data into a stable string. Drops volatile bits (e.g. exact scalar values) so repeated runs of the same code reuse compiled artifacts.

## Executor interface

```ts
interface Executor<K extends ShapeKind, D, C> {
  readonly name: string; // e.g. "cjit-loop"
  readonly kind: K; // "top-level" | "loop" | "call"

  // Cheap; runs on every dispatch. Returns data for compile/run, or null
  // to decline.
  propose(lowered: LoweredOf<K>, interp: Interpreter): D | null;

  cacheKey(data: D): string;

  // Called only on cache miss. Cached under (executor.name, owner,
  // cacheKey), where owner is the head Stmt (top-level, loop) or the
  // FunctionDef (call).
  compile(data: D, interp: Interpreter): C;

  // { result } on success (the return value for a call; undefined
  // otherwise), null to bail, or RERUN for a runtime error the
  // interpreter must reproduce. A throw is the unit's error.
  run(compiled: C, data: D, interp: Interpreter): RunResult;

  // Whether the run's output may stream (see Output streaming).
  streamsOutput?(compiled: C): boolean;
}

type RunResult = { result: unknown } | null | typeof RERUN;
```

## Dispatch

Each shape has its own entry point; all three share one loop (`Registry.runFirst`):

```
runFirst(executors, lowered, owner):
  for executor in executors:          // registration order
    data = executor.propose(lowered)
    if data is null: continue
    key = executor.cacheKey(data)
    compiled = cache.get(executor, owner, key)
    if compiled is BAILED: continue
    if compiled is missing: compiled = cache.set(..., executor.compile(data))
    result = executor.run(compiled, data)   // output captured or streamed
                                            // (a throw commits it, then propagates)
    if result is null:                      // bail
      discard output, plots, PRNG draws
      cache.markBailed(executor, owner, key)
      continue
    if result is RERUN:                     // runtime error, see Runtime errors
      discard output, plots, PRNG draws
      return null                           // no bail cached
    commit output; return result
  return null                               // caller falls back to the interpreter
```

- `tryRunWholeScope(siblings, interp)`: called once at script start with the script body. On null, the interpreter runs the body statement by statement.
- `dispatch(siblings, i, interp)`: per statement. A `for`/`while` stmt is lowered and offered to the loop executors; any other stmt (and a loop no executor took) runs on the interpreter.
- `dispatchCall(fn, args, nargout, interp)`: from `callBody` (`interpreter/callConvention.ts`). On null the interpreter's call path runs.

There is no cost model: the order is fixed by registration (`executors/plugins.ts`), C-JIT before JS-JIT. A C executor declines what it cannot marshal across the koffi ABI, so JS-JIT picks that up, and the interpreter takes whatever both decline.

A compiler rejection (`UnsupportedConstruct` / `JitTypeError`) makes `compile` return null; `run` then bails, and the bail is cached, so the unit is not recompiled on every later dispatch. Runtime bails are cached the same way; runtime errors are not (see [Runtime errors](#runtime-errors-and-bails)). Side effects of a run that bails or re-runs are rolled back so the interpreter's re-run does not duplicate them: its output is discarded (or, for a streaming run, skipped on the re-run; see [Output streaming](#output-streaming)), plot instructions and plot state are restored (`Runtime.discardOutputCapture`), and the PRNG is rewound (`snapshotRng` / `restoreRng`). A run that throws (an error or a cancel) keeps the output it produced before, as the interpreter would.

## Lowering

`tryLowerTopLevel`, `tryLowerLoop`, and `tryLowerCall` classify the unit (inputs, outputs, input types, cache key) via `classification.ts`. Input types are the JIT compiler's own `Type`, read off the runtime values by `typeOfValue` (a scalar's sign but not its value, a tensor's shape; `Unknown` for values no backend takes), and the cache key is built from their `canonicalizeType` strings; see [type-system.md](compiler/type-system.md#executor-inputs). Lowering makes no codegen-feasibility decisions; those live in each executor's `propose`.

`WideningState` records the most recent input-type signature per owner (and per nargout for calls). Classification widens each input against it with `widenArgType`, so a unit invoked with shifting input types converges to a single specialization rather than thrashing the compile cache. An input that should not widen (a scalar after a tensor, say) keeps its own type and gets its own specialization.

The loop shape has one extra step. A capture-free function-handle _input_ (defined before the loop, e.g. `f = @(t) …;` or `f = @name;`) would otherwise be an opaque `function_handle` the JIT can't type. Classification (`classifyLoop`) detects these and, instead of taking them as runtime inputs, emits them as `ConstHandle`s that the loop executor inlines as in-scope `<name> = @…` assignments prepended to the synthetic body, reducing the boundary case to the supported in-scope handle case on both backends. The inliner (`handleInline.ts`) reuses the handle's recorded defining AST and env (stored on `RuntimeFunction`) and only fires when capture-free and same-file. It also declines if any free body name (a name that was a function or undefined at definition, not a captured variable) collides with a name that will be in scope at the relocation site (the loop's inputs, assigned locals, or `for` variable); otherwise re-lowering would re-resolve that name to the loop variable, silently turning `@(t) sq(t)` from a function call into an array index.

## Shapes and backends

The shape executors (`executors/jit/topLevelExecutor.ts`, `loopExecutor.ts`, `callExecutor.ts`) are factories, `make<Shape>Executor(backend)`, that produce `<prefix>-top-level`, `<prefix>-loop`, and `<prefix>-call`, where the prefix is `jit` (JS) or `cjit` (C).

- **top-level** wraps the script body in a synthetic function (params = variables read from the env, outputs = variables assigned), and writes the outputs back to the env. It declines scripts with a top-level `return` and scripts with unsuppressed assignments, whose "x = 10" display the emitted code does not reproduce. The C backend applies a stricter display gate (unsuppressed bare expressions decline too) unless the script carries `%!numbl:assert_jit c`.
- **loop** wraps a single `for`/`while` stmt the same way; outputs are filtered to those live after the loop.
- **call** compiles the user function itself, specialized on argument types and nargout, and returns its outputs. It declines `nargout === 0`, since the interpreter still binds the first output to `ans`.

A `JitBackend` supplies:

| Member        | JS (`jsBackend.ts`)                            | C (`cBackend.ts`)                                                        |
| ------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| `available`   | always                                         | `interp.nativeBridge.koffi` present                                      |
| `accepts`     | any input type but `Unknown`                   | input types with a C declaration (`compilerTypeToCDecl`)                 |
| `compile`     | `compileSpec` → `new Function` → spec function | `compileSpecC` → `cc -shared` → koffi `dlopen`                           |
| `invoke`      | `numblToJit` / `jitToNumbl` around the JS call | `marshalInputs` / out-pointer buffers / `unmarshal*` around the FFI call |
| raise channel | tagged `JitRaise` thrown by the emitted code   | `longjmp` to the host-entry guard; status read with `bindRaise`          |

Shared policy lives in `backend.ts`: `compileUnit` (the single place compiler rejections are caught and recorded for `NUMBL_JIT_LOG` and `%!numbl:assert_jit`), `declinedByAssertC` (at `--opt 2` the JS backend declines units marked `%!numbl:assert_jit c`, so they either C-JIT or reach the interpreter, which raises), and `insideCompiledLoop` (the loop-depth gate below).

## The loop-depth gate

The loop and call executors decline while `interp.loopDepth > 0`: an enclosing loop is the natural JIT entry point, since compiling it captures everything nested inside, and bidding per-iteration underneath it would pay propose and cache-lookup overhead for no gain.

That reasoning holds only while the enclosing loop is going to be compiled. When it is _declined_, the interpreter walks its body statement by statement and nothing above will capture what is nested inside, so the gate lifts. `Registry.interpretFallback` bumps `Interpreter.jitDeclinedLoopDepth` around the interpreter's execution of a declined loop stmt, and the executors gate on `loopDepth > 0 && jitDeclinedLoopDepth === 0`. A nested loop or an expensive callee under a declined loop then gets its own chance; declines are cached, so a nested unit that also declines costs one cache lookup per iteration rather than a repeated compile.

## Runtime errors and bails

The JIT rejects statically: the compiler either lowers a spec cleanly or throws at `compile()` time. At run time, compiled code reports everything through one _raise channel_, `jit_raise(kind, identifier, message)`: emitted JS throws a tagged `JitRaise` (`runtime/system/raise.js`), and compiled C `longjmp`s to its host-entry guard and leaves a status (`runtime/system/raise.h`, see [cjit.md](jit/cjit.md)). Both backends normalize it to `executors/jit/raise.ts`'s `JitRaise`, carrying the source statement that raised it (emitted code records the current statement as the interpreter sets `rt.$line`), and the shape executors map it with `handleRaise`:

- **bail** (a JIT limitation, e.g. an indexed store past a tensor's extent, which would grow it): warn through `interp.onJitBail`, return null. The bail is cached and the interpreter re-runs the unit.
- **error** (a MATLAB runtime error): when the JIT reproduces the interpreter's result exactly, throw numbl's `RuntimeError` with the interpreter's message and identifier, at the raising statement's line (and, for the call shape, with the function's call frame). That requires the wording (the message is not null), a raising statement in the unit's own code (not a callee, whose frames the JIT does not track, and not a recursive function), and unobservable intermediate state: the call shape discards the function's workspace, while a loop or script body inside a `try` would leave variables the catch block can read but the compiled unit never wrote back. Otherwise return `RERUN`: the interpreter re-runs the unit to raise the error itself. Neither path caches a bail, so one error does not de-JIT the unit.

- **cancel** (the host's cancel flag was set; see [Cancellation](#cancellation)): throw numbl's `CancellationError`, as the interpreter's `checkCancel` would. The run ends, its output so far stays, and no bail is cached.

Anything else thrown by emitted JS is a JIT bug and propagates as an internal error (a V8 stack overflow excepted: it re-runs, so the interpreter reports its own recursion limit).

## Cancellation

The IDE's Stop sets word 0 of a `SharedArrayBuffer` (`ExecOptions.cancelSAB`, viewed as `Runtime.cancelFlag`) from the main thread while the worker runs synchronously. The interpreter polls it between statements and at calls (`Runtime.checkCancel`), and compiled code polls it too, so Stop also ends a JIT'd loop:

- **Where.** Every loop iteration, and the entry of every function that calls a spec (so recursion polls as well). A loop whose iterations do constant work (scalar statements only, no nested loop or user call; `jit/codegen/cancelPoll.ts`) is strip-mined instead: an outer loop polls around chunks of 1024 iterations that do not, since even one extra test per iteration costs 10% or more in a one-statement loop. The work between polls is thus bounded, except for a single long builtin call.
- **JS.** The module's `$cancel` (`$h.cancel`: the runtime's flag, or a private zero word) is read through a function-local alias with a plain typed-array load; V8 does not eliminate such loads across iterations, so the other thread's store is seen. A loop tests the flag in its condition and raises after it exits, because a raise call inside the loop body, even one never taken, slows a tight V8 loop severalfold.
- **C.** `JIT_POLL_CANCEL()`: a relaxed atomic load through `jit__cancel` and a never-taken branch to `jit_raise_cancel`, which raises through the raise channel, so the guard frees the call's allocations. The host installs the flag with the `.so`'s `jit_set_cancel` before a call when it changed; koffi passes a pointer to the `SharedArrayBuffer`'s backing store, which does not move.

On scalar loops of one to three statements the cost is within the run-to-run noise (about 5%) at `--opt 1` and `--opt 2`. The exception is a `while` loop whose body is a single scalar update (`while k < n, k = k + 1; end`, about one cycle per iteration), which runs about 40% slower at `--opt 1`: the chunk counter that strip-mining adds is a large share of such an iteration. Only a host with a cancel buffer can cancel: the IDE worker and embedders that pass `cancelSAB`. The CLI has none; Ctrl-C ends the process, compiled code included.

## Output streaming

A JIT run's output is captured so that a bail or `RERUN` leaves none behind. Output could stream directly only for a run that cannot end in a bail or `RERUN` once it has produced output. Bails come from indexed stores that may grow an array (every store with a computed index), allocation-overflow and out-of-memory checks in the C runtime, and a few builtin limitations; `RERUN` comes from every error whose wording the JIT leaves to the interpreter, every error raised in a callee or a recursive function, every error in a loop or script unit inside `try`, and a JS stack overflow. A unit free of all of these is scalar code that stores into no array by index, which excludes most long loops worth watching.

Instead, a run streams (`Executor.streamsOutput`) and a discard is made safe on the re-run. `Runtime.beginOutputCapture(true)` emits the run's output as it is produced and records it; `discardOutputCapture` rewinds a cursor to where the run began, and while the interpreter's re-run produces output up to what was already emitted, `Runtime.output` skips it as long as it matches. Nested attempts rewind the same cursor. This is exact when the re-run reproduces the discarded run's output byte for byte up to the point of the discard: the re-run starts from the same variables and PRNG state, and the JIT reproduces the interpreter's output (the `jit_parity` gate). It does not hold for output that depends on the clock, so a unit whose module uses `toc` (`outputIsReproducible`) is captured as before. Should the re-run's output differ (a parity bug), the rest is emitted from the first differing character, the discarded run's text past it stays visible, and `NUMBL_JIT_LOG` reports `[jit-replay]`. Plot instructions are not streamed and are still rolled back.

Streaming means user output callbacks (`onOutput`) run while compiled code is on the stack, during a C call through the `.so`'s output callback. A `.so` is not re-entrant (its raise guard and allocator list are file-scope), so a call into a `.so` that is already running bails.

## Mode-driven registration

`registerExecutorsForOpt(registry, opt)` in `executors/plugins.ts` is the single switch from an `--opt` level to the registered executors.

| `--opt` | Executors, in order                                                                      |
| ------- | ---------------------------------------------------------------------------------------- |
| 0       | none                                                                                     |
| 1       | jit-top-level, jit-loop, jit-call                                                        |
| 2       | cjit-top-level, cjit-loop, cjit-call, then jit-top-level, jit-loop, jit-call (Node only) |

The C backend is not imported by `plugins.ts`. `initNodePlatform` (`src/node-platform.ts`, used by the CLI and exported from `numbl/node`) calls `registerNodeCompileC()` (`executors/jit/compileC.node.ts`), which installs the `cc` + koffi compile/load implementation behind the browser-safe stub in `compileC.ts` and registers `cBackend` with `registerCJitBackend`. The browser worker bundle never imports `compileC.node.ts`, so the Node modules and the C compile graph stay out of the web build; there, and whenever `koffi` is unavailable, `--opt 2` behaves like `--opt 1`.

## Files

The executor registry and the per-shape executors live under `executors/`; the JIT compiler they call (lowering → IR → JS/C codegen, builtins, runtime snippets) is the self-contained subsystem under [`src/numbl-core/jit`](jit/overview.md).

```
executors/
  types.ts          Executor / RunResult / ShapeKind
  registry.ts       Registry: tryRunWholeScope, dispatch, dispatchCall
  cache.ts          ExecutorCache (WeakMap with BAILED sentinel)
  lowering.ts       tryLowerTopLevel / tryLowerLoop / tryLowerCall + WideningState
  classification.ts top-level / loop / call classification (backend-independent)
  inputTypes.ts     runtime value → compiler Type, widening, cache-key strings
  handleInline.ts   inlines capture-free function-handle inputs into loop bodies
  plugins.ts        registerExecutorsForOpt (the --opt → executors switch)
  jit/
    backend.ts           JitBackend interface + shared shape helpers
    topLevelExecutor.ts  makeTopLevelExecutor(backend)
    loopExecutor.ts      makeLoopExecutor(backend)
    callExecutor.ts      makeCallExecutor(backend)
    jsBackend.ts         JS backend (compileSpec + new Function)
    cBackend.ts          C backend (compileSpecC + cc + koffi; Node only)
    session.ts           per-context Workspace + Lowerer (the spec cache)
    valueAdapter.ts      RuntimeValue ↔ emitted-JS value shape
    typeAdapterC.ts      compiler Type → C decl (`compilerTypeToCDecl`)
    valueAdapterC.ts     RuntimeValue ↔ C ABI marshaling (koffi)
    hostHelpers.ts       the `$h` object the emitted JS receives
    compileC.ts          browser-safe compile/load stub
    compileC.node.ts     cc invoke + koffi dlopen; registers cBackend (Node only)
```

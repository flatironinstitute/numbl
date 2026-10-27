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
  // otherwise), or null to bail.
  run(compiled: C, data: D, interp: Interpreter): RunResult;
}

type RunResult = { result: unknown } | null;
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
    result = executor.run(compiled, data)   // output captured
    if result is null:                      // bail
      discard captured output; cache.markBailed(executor, owner, key)
      continue
    commit captured output; return result
  return null                               // caller falls back to the interpreter
```

- `tryRunWholeScope(siblings, interp)`: called once at script start with the script body. On null, the interpreter runs the body statement by statement.
- `dispatch(siblings, i, interp)`: per statement. A `for`/`while` stmt is lowered and offered to the loop executors; any other stmt (and a loop no executor took) runs on the interpreter.
- `dispatchCall(fn, args, nargout, interp)`: from `callUserFunction`. On null the interpreter's call path runs.

There is no cost model: the order is fixed by registration (`executors/plugins.ts`), C-JIT before JS-JIT. A C executor declines what it cannot marshal across the koffi ABI, so JS-JIT picks that up, and the interpreter takes whatever both decline.

A compiler rejection (`UnsupportedConstruct` / `JitTypeError`) makes `compile` return null; `run` then bails, and the bail is cached, so the unit is not recompiled on every later dispatch. Runtime bails are cached the same way. Output produced during a run that bails is discarded so the interpreter's re-run does not duplicate it.

## Lowering

`tryLowerTopLevel`, `tryLowerLoop`, and `tryLowerCall` classify the unit (inputs, outputs, input types, cache key) via `classification.ts`. Lowering makes no codegen-feasibility decisions; those live in each executor's `propose`.

`WideningState` records the most recent input-type signature per owner (and per nargout for calls). Classification widens against it, so a unit invoked with shifting input types converges to a single specialization rather than thrashing the compile cache.

The loop shape has one extra step. A capture-free function-handle _input_ (defined before the loop, e.g. `f = @(t) …;` or `f = @name;`) would otherwise be an opaque `function_handle` the JIT can't type. Classification (`classifyLoop`) detects these and, instead of taking them as runtime inputs, emits them as `ConstHandle`s that the loop executor inlines as in-scope `<name> = @…` assignments prepended to the synthetic body, reducing the boundary case to the supported in-scope handle case on both backends. The inliner (`handleInline.ts`) reuses the handle's recorded defining AST and env (stored on `RuntimeFunction`) and only fires when capture-free and same-file. It also declines if any free body name (a name that was a function or undefined at definition, not a captured variable) collides with a name that will be in scope at the relocation site (the loop's inputs, assigned locals, or `for` variable); otherwise re-lowering would re-resolve that name to the loop variable, silently turning `@(t) sq(t)` from a function call into an array index.

## Shapes and backends

The shape executors (`executors/jit/topLevelExecutor.ts`, `loopExecutor.ts`, `callExecutor.ts`) are factories, `make<Shape>Executor(backend)`, that produce `<prefix>-top-level`, `<prefix>-loop`, and `<prefix>-call`, where the prefix is `jit` (JS) or `cjit` (C).

- **top-level** wraps the script body in a synthetic function (params = variables read from the env, outputs = variables assigned), and writes the outputs back to the env. It declines scripts with a top-level `return` and scripts with unsuppressed assignments, whose "x = 10" display the emitted code does not reproduce. The C backend applies a stricter display gate (unsuppressed bare expressions decline too) unless the script carries `%!numbl:assert_jit c`.
- **loop** wraps a single `for`/`while` stmt the same way; outputs are filtered to those live after the loop.
- **call** compiles the user function itself, specialized on argument types and nargout, and returns its outputs. It declines `nargout === 0`, since the interpreter still binds the first output to `ans`.

A `JitBackend` supplies:

| Member           | JS (`jsBackend.ts`)                            | C (`cBackend.ts`)                                                        |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| `available`      | always                                         | `interp.nativeBridge.koffi` present                                      |
| `mapType`        | `jitTypeToCompilerType`                        | same, and the type must have a C declaration (`compilerTypeToCDecl`)     |
| `compile`        | `compileSpec` → `new Function` → spec function | `compileSpecC` → `cc -shared` → koffi `dlopen`                           |
| `invoke`         | `numblToJit` / `jitToNumbl` around the JS call | `marshalInputs` / out-pointer buffers / `unmarshal*` around the FFI call |
| grow-bail signal | sentinel exception from the store helper       | flag set in the `.so` (`bindGrowBail`)                                   |

Shared policy lives in `backend.ts`: `compileUnit` (the single place compiler rejections are caught and recorded for `NUMBL_JIT_LOG` and `%!numbl:assert_jit`), `declinedByAssertC` (at `--opt 2` the JS backend declines units marked `%!numbl:assert_jit c`, so they either C-JIT or reach the interpreter, which raises), and `insideCompiledLoop` (the loop-depth gate below).

## The loop-depth gate

The loop and call executors decline while `interp.loopDepth > 0`: an enclosing loop is the natural JIT entry point, since compiling it captures everything nested inside, and bidding per-iteration underneath it would pay propose and cache-lookup overhead for no gain.

That reasoning holds only while the enclosing loop is going to be compiled. When it is _declined_, the interpreter walks its body statement by statement and nothing above will capture what is nested inside, so the gate lifts. `Registry.interpretFallback` bumps `Interpreter.jitDeclinedLoopDepth` around the interpreter's execution of a declined loop stmt, and the executors gate on `loopDepth > 0 && jitDeclinedLoopDepth === 0`. A nested loop or an expensive callee under a declined loop then gets its own chance; declines are cached, so a nested unit that also declines costs one cache lookup per iteration rather than a repeated compile.

## Bailouts

The JIT rejects statically: the compiler either lowers a spec cleanly or throws at `compile()` time. A compiled spec bails at runtime in two cases. The first is an indexed store past a tensor's current extent, which would grow it (the grow-bail); this also warns through `interp.onJitBail`. The second is any other runtime error raised by the emitted code, which bails silently so the interpreter reproduces numbl's canonical error. In both cases the interpreter re-runs the whole unit.

## Mode-driven registration

`registerExecutorsForOpt(registry, opt)` in `executors/plugins.ts` is the single switch from an `--opt` level to the registered executors.

| `--opt` | Executors, in order                                                                      |
| ------- | ---------------------------------------------------------------------------------------- |
| 0       | none                                                                                     |
| 1       | jit-top-level, jit-loop, jit-call                                                        |
| 2       | cjit-top-level, cjit-loop, cjit-call, then jit-top-level, jit-loop, jit-call (Node only) |

The C backend is not imported by `plugins.ts`. `cli.ts` calls `registerNodeCompileC()` (`executors/jit/compileC.node.ts`) at bootstrap, which installs the `cc` + koffi compile/load implementation behind the browser-safe stub in `compileC.ts` and registers `cBackend` with `registerCJitBackend`. The browser worker bundle never imports `compileC.node.ts`, so the Node modules and the C compile graph stay out of the web build; there, and whenever `koffi` is unavailable, `--opt 2` behaves like `--opt 1`.

## Files

The executor registry and the per-shape executors live under `executors/`; the JIT compiler they call (lowering → IR → JS/C codegen, builtins, runtime snippets) is the self-contained subsystem under [`src/numbl-core/jit`](jit/overview.md).

```
executors/
  types.ts          Executor / RunResult / ShapeKind
  registry.ts       Registry: tryRunWholeScope, dispatch, dispatchCall
  cache.ts          ExecutorCache (WeakMap with BAILED sentinel)
  lowering.ts       tryLowerTopLevel / tryLowerLoop / tryLowerCall + WideningState
  classification.ts top-level / loop / call classification (backend-independent)
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
    typeAdapter.ts       numbl JitType → compiler Type
    valueAdapter.ts      RuntimeValue ↔ emitted-JS value shape
    typeAdapterC.ts      compiler Type → C decl (`compilerTypeToCDecl`)
    valueAdapterC.ts     RuntimeValue ↔ C ABI marshaling (koffi)
    hostHelpers.ts       the `$h` object the emitted JS receives
    compileC.ts          browser-safe compile/load stub
    compileC.node.ts     cc invoke + koffi dlopen; registers cBackend (Node only)
```

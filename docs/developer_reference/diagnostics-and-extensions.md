# Diagnostics and Extensions

## Diagnostics

Errors that escape the interpreter (syntax errors and runtime errors) are converted to structured `DiagnosticInfo` objects by `diagnoseErrors` before being reported. Each diagnostic carries:

- error class (`errorType`: syntax / runtime / unknown) and the MATLAB `identifier`, if any;
- source location (file, line);
- a source snippet with a pointer caret;
- the call stack when it is a runtime error.

This conversion lives in one diagnostics layer (`src/numbl-core/diagnostics/`, which also has `formatDiagnostic`) that both the CLI and the web worker consume. Any new error-producing path should throw a typed error the diagnostics layer already understands rather than format messages ad hoc.

### Caught errors (MException)

`catch ME` binds an MException object, made from the `RuntimeError` by `errorToMException` ([`runtime/mexception.ts`](../../src/numbl-core/runtime/mexception.ts)). An MException is a value-class `RuntimeClassInstance` with className `MException` (or a user subclass) and the properties identifier, message, cause, stack and Correction; there is no classdef behind it. Its methods (`throw`, `rethrow`, `throwAsCaller`, `getReport`, `addCause`, ...) are builtins taking the exception first ([`builtins/mexception.ts`](../../src/numbl-core/interpreter/builtins/mexception.ts)), which a method call on the object reaches because class-method dispatch falls back to builtins for a class without a classdef. A `throw` puts the exception on the `RuntimeError` (`exception`), so the catch gets the same object back (subclass, causes, extra properties) with the stack of the throw site; `rethrow` also sets `keepStack` so the original stack survives. Errors raised by `error` and builtins carry only an identifier and message, and the catch builds the exception and its stack from the error's call stack.

## JS user functions

Workspace files with a `.numbl.js` extension define functions in JavaScript; see [src/docs/js-functions.md](../../src/docs/js-functions.md) for the user-facing API. Each file calls `register` with a function `(args, nargout) => result`, an object `{ apply, help }`, or the older `{ resolve }` form, and the loader (`builtinFromSpec` in `jsUserFunctions.ts`) wraps it as an `IBuiltin`. They are loaded at startup and participate in function resolution like workspace functions. This is the extension point for users who want native-ish performance or access to external APIs without patching numbl itself.

In resolution, `.numbl.js` functions sit just after workspace `.m` functions (including the bundled stdlib) and before class constructors and builtins, so a `.m` of the same name shadows a `.numbl.js` function, which in turn shadows a builtin; see [function-resolution.md](compiler/function-resolution.md#bare-names-resolvename).

The loader (`jsUserFunctions.ts`) runs each file's body via `new Function(...)`, injecting these globals: `RTV`, `RuntimeError`, `Float64Array`, `register`, `wasm`, `native`, `importJS`, `callHandle`, `toNumber`, `gpu`. A `// wasm: <name>` / `// native: <name>` directive at the top binds a WASM module / native shared library.

### Function-handle callbacks into WASM

A `.numbl.js` apply can receive a numbl function handle (it arrives as a `function_handle` argument, a `RuntimeFunction`) and invoke it two ways:

- **Directly from JS** via `callHandle(handle, args, nargout?)`, which binds to the active runtime through `getCurrentRuntime()` and forwards to `Runtime.index` (the same call path `arrayfun` uses). `args` may be runtime values or raw JS scalars.
- **From inside WASM** via a per-instance callback registry. WASM cannot hold a `RuntimeFunction`, so the handle stays in JS and WASM gets an integer id: `wasm.callbacks.add(fn)` registers a JS thunk and returns an id, `wasm.callbacks.remove(id)` drops it (id lifetime = one outer call). The WASM module declares `extern double numbl_cb_d(int id, double x)` and calls it to invoke the handle; the host wires `env.numbl_cb_d` in `instantiateWasm` to look up the thunk and call it. The import is always provided (a module only resolves imports it declares), so no directive is needed. A missing id throws, propagating out through the WASM call.

Worked example: [`myquad.numbl.js`](../../numbl_test_scripts/functions/js_user_functions/myquad.numbl.js) + [`myquad.c`](../../numbl_test_scripts/functions/js_user_functions/myquad.c) integrate a handle by the midpoint rule (with a pure-JS fallback when the WASM module is absent). Tests: `numbl_test_scripts/functions/js_user_functions/test_js_callback.m` (full numbl chain via the fallback) and the `wasm function-handle callbacks` block in `coverage-extras.test.ts` (full chain through a hand-assembled WASM module). Only the scalar `numbl_cb_d` signature exists; vector variants would marshal through linear memory.

### WebGPU from `.numbl.js` (browser)

In the browser workers (IDE and embedding session), `.numbl.js` files get a `gpu` global (`null` elsewhere, or without WebGPU or cross-origin isolation). Builtins are synchronous but WebGPU is not, so a helper worker spawned at worker boot owns the GPU device, and `gpu.call` blocks on a `SharedArrayBuffer` with `Atomics.wait` until the helper replies (a worker created during a run would not start until the run returned to the event loop, hence the spawn at boot). `gpu.load(name, factory)` installs a self-contained factory in the helper; its methods take `(device, args)` and their state (GPU buffers, pipelines) persists across calls and runs until the source changes, when an optional `dispose()` is called. `gpu.call(name, method, args)` returns the method's result; arguments and results may be numbers, strings, booleans, null, arrays, plain objects and typed arrays, and an exception in the helper becomes a numbl error. See [`src/gpuHelper.ts`](../../src/gpuHelper.ts). A round trip costs a few milliseconds, so methods should do coarse-grained work.

## Op-code synchronization

The ops layer's op codes are declared on the TypeScript side (`src/numbl-core/ops/opCodes.ts`) and mirrored in the native addon's C headers. A unit test (`op-codes-sync.test.ts`) asks the loaded addon for its op-code table (`numbl_dump_op_codes()`) and compares it against the TS enums; a mismatch fails the test rather than silently executing the wrong kernel. When adding a new op, update both sides and re-run this test.

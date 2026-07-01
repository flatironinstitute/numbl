# Diagnostics and Extensions

## Diagnostics

Errors that escape the interpreter — syntax errors, semantic errors, runtime errors — are converted to structured `DiagnosticInfo` objects before being reported. Each diagnostic carries:

- error class (`errorType`: syntax / semantic / runtime / unknown);
- source location (file, line);
- a source snippet with a pointer caret;
- the call stack when it is a runtime error.

This conversion lives in a single diagnostics layer that both the CLI error formatter and the web worker consume. Any new error-producing path should throw a typed error the diagnostics layer already understands — do not format messages ad hoc.

## External-access directives

A per-file annotation that marks which workspace variables are externally observable. The web worker uses these to decide which variables to serialize back to the main thread after a run (most workspace state does not need to cross the wire). The `LoweringContext` collects and stores directive metadata from each parsed file; consumers read it through the context.

## JS user functions

Workspace files with a `.numbl.js` extension can register custom `IBuiltin`s written in TypeScript or JavaScript. They are loaded at startup and participate in builtin dispatch like any other registered builtin. This is the extension point for users who want native-ish performance or access to external APIs without patching numbl itself.

Resolution order for a free function name (after the in-scope checks for nested functions, local subfunctions, imports, private functions, and class-method dispatch on a class-instance arg) is: `.m` workspace functions → `.numbl.js` user functions → workspace class constructors → registered `IBuiltin`s. The stdlib `.m` files are bundled into the workspace at startup, so they live in the same tier as user `.m` files (and shadow same-named builtins). A user `.m` or `.numbl.js` with the same name as a builtin therefore shadows it.

The loader (`jsUserFunctions.ts`) runs each file's body via `new Function(...)`, injecting these globals: `RTV`, `RuntimeError`, `Float64Array`, `register`, `wasm`, `native`, `importJS`, `callHandle`, `toNumber`. A `// wasm: <name>` / `// native: <name>` directive at the top binds a WASM module / native shared library.

### Function-handle callbacks into WASM

A `.numbl.js` apply can receive a numbl function handle (it arrives as a `function_handle` argument — a `RuntimeFunction`) and invoke it two ways:

- **Directly from JS** via `callHandle(handle, args, nargout?)`, which binds to the active runtime through `getCurrentRuntime()` and forwards to `Runtime.index` (the same call path `arrayfun` uses). `args` may be runtime values or raw JS scalars.
- **From inside WASM** via a per-instance callback registry. WASM cannot hold a `RuntimeFunction`, so the handle stays in JS and WASM gets an integer id: `wasm.callbacks.add(fn)` registers a JS thunk and returns an id, `wasm.callbacks.remove(id)` drops it (id lifetime = one outer call). The WASM module declares `extern double numbl_cb_d(int id, double x)` and calls it to invoke the handle; the host wires `env.numbl_cb_d` in `instantiateWasm` to look up the thunk and call it. The import is always provided — a module only resolves imports it declares — so no directive is needed. A missing id throws, propagating out through the WASM call.

Worked example: [`myquad.numbl.js`](../../numbl_test_scripts/functions/js_user_functions/myquad.numbl.js) + [`myquad.c`](../../numbl_test_scripts/functions/js_user_functions/myquad.c) integrate a handle by the midpoint rule (with a pure-JS fallback when the WASM module is absent). Tests: `test_js_callback.m` (full numbl chain via the fallback) and the `wasm function-handle callbacks` block in `coverage-extras.test.ts` (full chain through a hand-assembled WASM module). Only the scalar `numbl_cb_d` signature exists today; vector variants (`numbl_cb_v` for vector→scalar, `numbl_cb_vv` for vector→vector, marshaling through linear memory) are the natural extension.

## Op-code synchronization

The ops layer's op codes are declared on the TypeScript side and mirrored in the native addon's C headers. A unit test asks the loaded addon for its op-code table and compares it against the TS enum; a mismatch fails the test rather than silently executing the wrong kernel. When adding a new op, update both sides and re-run this test.

## Input channel

In the browser worker, `input(...)` is implemented through a `SharedArrayBuffer` + `Atomics` bridge so it can block synchronously while the main thread gathers the response. See [web-app.md](web-app.md) for the broader worker architecture.

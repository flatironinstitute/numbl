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

## Op-code synchronization

The ops layer's op codes are declared on the TypeScript side and mirrored in the native addon's C headers. A unit test asks the loaded addon for its op-code table and compares it against the TS enum; a mismatch fails the test rather than silently executing the wrong kernel. When adding a new op, update both sides and re-run this test.

## Input channel

In the browser worker, `input(...)` is implemented through a `SharedArrayBuffer` + `Atomics` bridge so it can block synchronously while the main thread gathers the response. See [web-app.md](web-app.md) for the broader worker architecture.

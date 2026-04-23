/**
 * Runtime-overridable OpenMP availability flag for the e1 codegen path.
 *
 * `scalarFnKernel.ts` is transitively reachable from the JS-JIT module
 * graph that Vite bundles for the web REPL, but `c/compile.ts` is
 * Node-only (child_process, fs, ...). Importing `cJitOpenmpAvailable`
 * directly from `compile.ts` would drag all of that into the browser
 * bundle. Instead we default to `false` here and let Node-only
 * `e1/install.ts` override the getter at install time — the same
 * pattern used for the `compileKernel` stub in `jitHelpers.ts`.
 */

let _getter: () => boolean = () => false;

export function setOpenmpAvailableGetter(fn: () => boolean): void {
  _getter = fn;
}

export function isOpenmpAvailable(): boolean {
  return _getter();
}

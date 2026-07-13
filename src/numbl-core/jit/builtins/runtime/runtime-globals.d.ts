// Ambient declarations for host hooks that runtime .js snippets
// reference as free variables. The host (CLI, browser, codegen-
// emitted `run($h)`) assigns these on `globalThis` before any
// snippet function runs.
//
// Adding a hook: declare it here, assign it in every host's
// bootstrap, and let snippets reference it by bare name.

declare global {
  /** Append text to stdout. No implicit newline. */
  var $write: (s: string) => void;

  /** Optional accelerated real matrix multiply, installed by the host when
   *  a WASM LAPACK bridge is loaded (browser worker). Given column-major
   *  A (m×k) and B (k×n), returns column-major C = A*B, or a falsy value to
   *  decline (the caller then runs its own JS loop). */
  var $matmulAccel:
    | ((
        a: Float64Array,
        m: number,
        k: number,
        b: Float64Array,
        n: number
      ) => Float64Array | null | undefined)
    | undefined;
}

export {};

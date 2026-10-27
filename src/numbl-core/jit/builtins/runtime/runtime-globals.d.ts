// Ambient declarations for host hooks that runtime .js snippets
// reference as free variables. The host (CLI, browser, codegen-
// emitted `run($h)`) assigns these on `globalThis` before any
// snippet function runs.
//
// Adding a hook: declare it here, assign it in every host's
// bootstrap, and let snippets reference it by bare name.

declare global {
  /** Optional host element-wise kernels for real tensors (the numbl JIT
   *  host installs the interpreter's native-or-TS kernels). `op` is the
   *  builtin name; `out` has the input's length. `unary` returns false
   *  for an op it doesn't cover; `minmax` skips NaN. */
  var $elemwise:
    | {
        binary(
          op: string,
          a: Float64Array,
          b: Float64Array,
          out: Float64Array
        ): void;
        scalarBinary(
          op: string,
          s: number,
          a: Float64Array,
          scalarOnLeft: boolean,
          out: Float64Array
        ): void;
        unary(op: string, a: Float64Array, out: Float64Array): boolean;
        minmax(op: "min" | "max", a: Float64Array): number;
      }
    | undefined;

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

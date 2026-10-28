// Ambient declarations for host hooks that runtime .js snippets
// reference as free variables. `$write`, `$warning`, `$plotDispatch`, `$rand` and
// `$elemwise` are module-scoped bindings in the emitted JS module,
// assigned from `$h` by its factory (see `codegen/emitJs.ts`), so each
// compiled module talks to its own host. `$matmulAccel` is a genuine
// global installed by the browser worker's WASM LAPACK bridge.
//
// Adding a per-host hook: declare it here, add it to the `let` and the
// factory in `emitJs.ts`, and let snippets reference it by bare name.

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

  /** Show warning `msg` unless the warning state turns off `identifier`
   *  ("" for a warning without one). */
  var $warning: (identifier: string, msg: string) => void;

  /** Optional host plot hook; without it plot calls are written to
   *  `$write` as records. */
  var $plotDispatch: ((name: string, args: unknown[]) => void) | undefined;

  /** One uniform double in [0, 1) from the host PRNG. */
  var $rand: () => number;

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

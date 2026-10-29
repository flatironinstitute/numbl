// Ambient declarations for host hooks that runtime .js snippets
// reference as free variables. `$write`, `$warning`, `$plotDispatch`,
// `$rand`, `$elemwise` and `$numeric` are module-scoped bindings in the
// emitted JS module, assigned from `$h` by its factory (see
// `codegen/emitJs.ts`), so each compiled module talks to its own host.
//
// Adding a per-host hook: declare it here, add it to the `let` and the
// factory in `emitJs.ts`, and let snippets reference it by bare name.

declare global {
  /** Optional host element-wise kernels for real tensors (the numbl JIT
   *  host installs the interpreter's native-or-TS kernels). `op` is the
   *  builtin name; `out` has the input's length. `unary` returns false
   *  for an op it doesn't cover; `reduce` reduces all of a (non-empty)
   *  array in sequential order (`min` / `max` skip NaN). */
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
        reduce(op: "sum" | "prod" | "min" | "max", a: Float64Array): number;
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

  /** Optional host matrix multiply (the numbl JIT host routes it through
   *  the interpreter's LAPACK provider stack). Column-major in and out.
   *  Returns null for a small product, which the caller computes with its
   *  own loop. */
  var $numeric:
    | {
        matmul(
          a: Float64Array,
          m: number,
          k: number,
          b: Float64Array,
          n: number
        ): Float64Array | null;
        matmulComplex(
          aRe: Float64Array,
          aIm: Float64Array | undefined,
          m: number,
          k: number,
          bRe: Float64Array,
          bIm: Float64Array | undefined,
          n: number
        ): { re: Float64Array; im: Float64Array } | null;
      }
    | undefined;
}

export {};

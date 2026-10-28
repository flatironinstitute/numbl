# Builtins

Built-in functions (numbl's `IBuiltin`s) are the implementations of MATLAB's non-language-level functions: arithmetic helpers, array construction, linear algebra, FFT, string ops, predicates, and so on. They live in a single in-process registry keyed by name (`interpreter/builtins/types.ts`).

## Interpreter and JIT

The interpreter calls a builtin on runtime values. The JIT compiler (`src/numbl-core/jit`) has its **own** builtin registry (`jit/builtins`), separate from the interpreter's table. Each JIT builtin carries a type-transfer rule plus `emitJs` / `emitC` hooks that synthesize inline JS / C for the lowered call; the same set drives both the `--opt 1` (JS) and `--opt 2` (C) backends. The JIT never reads interpreter builtin metadata. A source builtin with no JIT counterpart simply isn't lowered: the call declines and runs on the interpreter. This split lets a builtin have a general implementation while providing a tight fast path for common cases.

## Defining a builtin

Builtins are defined with `builtin(spec)` (`interpreter/builtins/builtin.ts`):

- `name` — the function name as seen by user code.
- `help` — signatures and a description, surfaced by `help`. Required for user-facing builtins.
- `nargin` — accepted input count: a number (exactly that many) or `[min, max]` (`max` may be `Infinity`). Default: any.
- `nargout` — accepted output count: a number `n` (up to `n`, i.e. `[0, n]`) or `[min, max]`. Default 1. `nargout: 0` marks a void builtin: its result is discarded, `ans` is not set, and requesting an output errors.
- `mfile` — set when MATLAB implements the function as an M-file; this only changes the "too many" error identifiers (below).
- `apply(args, nargout, ctx)` — the implementation, on runtime values. It returns one value, or an array when `nargout > 1`. `ctx.name` is the builtin's name and `ctx.rt` the active `Runtime` (output, file I/O, figures, search path).

There are no argument-type rules. `apply` sees whatever values the caller passed and must reject what it cannot handle; `args.ts` provides coercion helpers that raise MATLAB's errors instead of silently dropping data (`numericArg`, `realNumericArg`, `realScalarArg`, `boolToNumber`, `builtinError`). For unary element-wise math, `types.ts` provides ready-made applies: `unaryElemwise`, `unaryRealResult`, and `unaryPredicate`.

Arity is checked before `apply` runs, using MATLAB's messages and identifiers: "Not enough input arguments." (`MATLAB:minrhs`), "Too many input arguments." (`MATLAB:maxrhs`, or `MATLAB:TooManyInputs` with `mfile`), "Too many output arguments." (`MATLAB:maxlhs`, or `MATLAB:TooManyOutputs` with `mfile`). `nargin(@name)` reports the maximum input count, or -1 when it is unbounded (MATLAB reports -1 for nearly all variadic builtins, e.g. `zeros`, `numel`, `fprintf`).

`builtin()` produces an ordinary `IBuiltin` with a `direct` entry (the arity metadata and a `call` that checks arity and runs `apply`). All callers go through `callIBuiltin` (`types.ts`), which uses `direct` when present.

## Legacy form

Builtins not yet converted implement `resolve(argTypes, nargout)`, usually through `defineBuiltin({ name, cases: [{ match, apply }] })`. `callIBuiltin` infers a dispatch type for each argument, and the first case whose `match` accepts them runs; if none does, the call fails with "Builtin 'name' does not support these argument types". The returned `outputTypes` are only used to detect void builtins (an empty list); nothing else reads them. `nargin(@name)` for a legacy builtin is estimated by probing `resolve` with up to 10 arguments of a few kinds (the largest accepted count, or -1 when every count is accepted). New code should use `builtin()`.

## Builtins that need the runtime or the caller's scope

Builtins that need runtime state (output, file I/O, plot instructions, figures, the search path) are ordinary `IBuiltin`s registered with `registerSpecial` / `registerSpecialVoid` (`runtime/registerSpecial.ts`); each call receives the active `Runtime` from the current-runtime stack. They live in `runtime/specialBuiltins.ts` and, for plotting and graphics ops, `plotting/builtins.ts`. Every caller (interpreter, JIT dispatch, `builtin(...)`, method dispatch) invokes builtins through `invokeBuiltin` in `runtime/runtimeDispatch.ts`.

A few names need the calling scope's variables (`eval`, `evalin`, `assignin`, `input`, `clear`, `exist`, `nargin`, ...). These are interpreter specials in `interpreter/interpreterSpecialBuiltins.ts`, which the interpreter checks before normal function resolution.

## Sparse direct solvers

Sparse matrices are stored in CSC form (`RuntimeSparseMatrix`). Element-wise and matrix arithmetic live in `helpers/sparse-arithmetic.ts`; the direct solvers live in `helpers/sparse-direct.ts`: Kronecker product, a cached structural analysis (`triangularInfo`: triangularity plus diagonal positions), forward/back substitution, an up-looking Cholesky with an elimination-tree symbolic phase, a left-looking LU with threshold partial pivoting (Gilbert-Peierls), symmetric/column permutations, and two orderings (exact minimum degree, reverse Cuthill-McKee). The `sparseSolve` driver picks triangular substitution, permuted Cholesky, or LU, and returns `null` when the dense path should be used (complex, non-square, or nearly full input).

Entry points: `mLeftDiv` / `mDiv` in `helpers/arithmetic.ts` (and the `LeftDiv` case in `runtime/runtimeOperators.ts`) route sparse left-hand sides to `sparseSolve`; `chol`, `lu`, `kron`, `inv`, `norm`, `istriu`, `istril`, `isbanded` in `builtins/linear-algebra.ts` dispatch on sparse input; `builtins/sparse-solvers.ts` registers `symrcm`, `amd`, `colamd`, `sprand`, `sprandn`, `sprandsym`, `isIllConditioned`, and the `decomposition` object (a `class_instance` whose factors sit in underscore-prefixed fields; the `mldivide` / `mrdivide` operator builtins in `misc.ts` call `tryDecompositionDivide` before the ordinary operator). Helper code emits MATLAB-style warnings (for example the singular-matrix warning) through `helpers/warn.ts`, whose sink the `Runtime` points at its output channel.

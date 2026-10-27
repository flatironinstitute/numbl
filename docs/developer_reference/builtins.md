# Builtins

Built-in functions (numbl's `IBuiltin`s) are the implementations of MATLAB's non-language-level functions: arithmetic helpers, array construction, linear algebra, FFT, string ops, predicates, and so on. They live in a single in-process registry keyed by name.

## Two complementary paths

Each builtin is consulted in two ways:

1. **Interpreter path** — the interpreter calls `resolve(argTypes, nargout)`, receives a specialized `apply` function, and invokes it on the runtime values. This path always works.
2. **JIT path** — the JIT compiler (`src/numbl-core/jit`) has its **own** builtin registry (`jit/builtins`), separate from the interpreter's `IBuiltin` table. Each JIT builtin carries a type-transfer rule plus `emitJs` / `emitC` hooks that synthesize inline JS / C for the lowered call; the same set drives both the `--opt 1` (JS) and `--opt 2` (C) backends. A source builtin with no JIT counterpart simply isn't lowered — the call declines and runs on the interpreter.

This split lets a builtin have a sophisticated general implementation while providing a tight fast path for common cases.

## Interface sketch

- `name` — the function name as seen by user code.
- `help` — structured metadata (signatures, description) surfaced by the `help` command. Required for user-facing builtins.
- `resolve(argTypes, nargout)` — decides whether this builtin accepts the given argument types. Returns `{ outputTypes, apply }` on success or `null` to reject (falling back to the next lookup or erroring).

(There is no JIT-emission hook on `IBuiltin`. The JIT compiler's builtins are a separate set under `src/numbl-core/jit/builtins/defs/`, each with its own `emitJs` / `emitC` hooks.)

## Registration helpers

Common patterns have shared helpers so a new builtin often needs only a few lines:

- unary element-wise math (real-input, real-result or complex-if-needed);
- unary real-result math (e.g., absolute value);
- binary scalar functions.

More complex builtins (multiple outputs, shape-dependent behavior, custom dispatch) implement `resolve` directly. To make a builtin JIT-compilable, add a matching definition (with `emitJs`/`emitC`) to the JIT compiler's builtin set under `src/numbl-core/jit/builtins/defs/`.

## Type responsibilities

`resolve` is responsible for:

- rejecting types the builtin does not support (return `null`, letting the interpreter either error or try another path);
- returning accurate output `JitType`s — the JIT propagates these to callers and will miscompile if they are wrong;
- respecting integer, logical, complex, and shape refinements where possible, so downstream callers can be specialized too.

## Builtins that need the runtime or the caller's scope

Builtins that need runtime state (output, file I/O, plot instructions, figures, the search path) are ordinary `IBuiltin`s registered with `registerSpecial` / `registerSpecialVoid` (`runtime/registerSpecial.ts`); each call receives the active `Runtime` from the current-runtime stack. They live in `runtime/specialBuiltins.ts` and, for plotting, `runtime/plotBuiltins.ts`. Every caller (interpreter, JIT dispatch, `builtin(...)`, method dispatch) invokes builtins through `invokeBuiltin` in `runtime/runtimeDispatch.ts`.

A few names need the calling scope's variables (`eval`, `evalin`, `assignin`, `input`, `clear`, `exist`, `nargin`, ...). These are interpreter specials in `interpreter/interpreterSpecialBuiltins.ts`, which the interpreter checks before normal function resolution.

## Sparse direct solvers

Sparse matrices are stored in CSC form (`RuntimeSparseMatrix`). Element-wise and matrix arithmetic live in `helpers/sparse-arithmetic.ts`; the direct solvers live in `helpers/sparse-direct.ts`: Kronecker product, a cached structural analysis (`triangularInfo`: triangularity plus diagonal positions), forward/back substitution, an up-looking Cholesky with an elimination-tree symbolic phase, a left-looking LU with threshold partial pivoting (Gilbert-Peierls), symmetric/column permutations, and two orderings (exact minimum degree, reverse Cuthill-McKee). The `sparseSolve` driver picks triangular substitution, permuted Cholesky, or LU, and returns `null` when the dense path should be used (complex, non-square, or nearly full input).

Entry points: `mLeftDiv` / `mDiv` in `helpers/arithmetic.ts` (and the `LeftDiv` case in `runtime/runtimeOperators.ts`) route sparse left-hand sides to `sparseSolve`; `chol`, `lu`, `kron`, `inv`, `norm`, `istriu`, `istril`, `isbanded` in `builtins/linear-algebra.ts` dispatch on sparse input; `builtins/sparse-solvers.ts` registers `symrcm`, `amd`, `colamd`, `sprand`, `sprandn`, `sprandsym`, `isIllConditioned`, and the `decomposition` object (a `class_instance` whose factors sit in underscore-prefixed fields; the `mldivide` / `mrdivide` operator builtins in `misc.ts` call `tryDecompositionDivide` before the ordinary operator). Helper code emits MATLAB-style warnings (for example the singular-matrix warning) through `helpers/warn.ts`, whose sink the `Runtime` points at its output channel.

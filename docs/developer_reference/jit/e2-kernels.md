# e2 kernels

`--opt e2` runs the AST interpreter exactly as `--opt 0` does, but
hooks every `Assign` statement AND every user-function call: when the
interpreter is about to execute one, the e2 driver checks whether it
heads a fusible chain / forms a recognized reduction pattern / is a
pure-scalar function — and if so — compiles a single-purpose C
function for that work, dispatches via koffi, and binds the resulting
buffer(s) and/or scalar(s) back into the env.

There is no JS-JIT under `--opt e2`. The base optimization level is
clamped to 0, so `tryJitFor` / `tryJitWhile` / `tryJitTopLevel` never
fire. The only JIT in play is on-demand C compilation triggered per
chain head or per scalar-function call.

## What gets fused

The e2 driver recognizes five kernel shapes, in order of preference:

1. **Chain + trailing reduction** — a run of consecutive same-or-
   different-LHS suppressed Assigns followed by an `acc = [acc OP]
reduce(<expr>)` Assign. The chain runs in one per-element loop;
   any LHS not referenced outside the chain (using full function-
   scope liveness) becomes a stack-local with no buffer materialized;
   the reduction is folded into the same loop as a scalar accumulator.
2. **Standalone reduction** — a single Assign of the form
   `acc = [acc OP] reduce(<elemwise expr>)` over env tensors, with no
   preceding chain. One pass, no intermediate tensor materialized.
3. **Multi-LHS chain (no reduction)** — a run of consecutive
   suppressed Assigns where every LHS that escapes (referenced
   elsewhere in the function or is a function output) gets its own
   `out_<name>` buffer. Chain-locals get stack-locals with no buffer.
4. **Multi-reduction over the same tensor** — a single scalar Assign
   whose RHS contains 2+ reduction calls (`sum`, `prod`, `max`, `min`,
   `mean`) over the same tensor variable. One C kernel computes all
   reductions in a single pass; the JS side substitutes the kernel's
   scalar outputs back into the residual expression and lets the
   interpreter evaluate it.
5. **Whole-function scalar kernel** — triggered from
   `callUserFunction`, not from an `Assign` hook. When the callee's
   args are all scalar `number`/`boolean`, its declared outputs are
   all scalar, and the body passes `checkCFeasibility`, the entire
   function body compiles to a single C kernel (`jit_<fnname>`). No
   per-iteration interpreter overhead inside the function body. Used
   by e.g. `benchmarks/scalar_bench.m`'s `run_bench(N, M)` to match
   e1's whole-function scalar path without routing through the
   JS-JIT outer.

If none of these patterns match, e2 silently falls through to the
regular interpreter.

## Pipeline

```
AST Assign (head of potential chain)
    │
    ▼
classifyAssignChain  ── opaque subtrees evaluated by interpreter ──► extraBindings
    │ candidate stmts                                                   ▲
    ▼                                                                   │
incremental lower stmt-by-stmt (truncate on first non-tensor result)    │
    │ acceptedAssigns + JitExpr specs                                   │
    ▼                                                                   │
matchTrailingReduction(siblings[afterChain])                            │
    │ optional reduction info                                           │
    ▼                                                                   │
liveness (using interp._currentScopeBody / _currentScopeExports)        │
    │ chain-local vs escape per LHS                                     │
    ▼                                                                   │
emit chain or reduction kernel                                          │
    │                                                                   │
    ▼                                                                   │
compileAndLoad (cc -O2 -shared -fPIC + koffi.load) ─► chain cache ──────┤
    │                                                                   │
    ▼                                                                   │
allocate output Float64Array(n) per escape LHS (reuse if unique-ref)    │
allocate Float64Array(1) for reduction accumulator (if any)             │
    │                                                                   │
    ▼                                                                   │
call kernel ──► bind escape LHSs to env ──► combine reduction with old acc
    │
    ▼
interp._e2ChainAdvance := (consumed - 1) ──► execStmts skips ahead
```

When the chain detector bails, the driver falls through to the
multi-reduction detector, which has its own driver and cache.

## Files

[executors/e2/](../../../src/numbl-core/executors/e2/):

| File                                                                                    | Role                                                                                               |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [classify.ts](../../../src/numbl-core/executors/e2/classify.ts)                         | AST `Expr` walker; multi-LHS chain detection; trailing-reduction matcher                           |
| [astToJitExpr.ts](../../../src/numbl-core/executors/e2/astToJitExpr.ts)                 | Whitelist-only AST → `JitExpr` lowerer with runtime types from env                                 |
| [chainKernelEmit.ts](../../../src/numbl-core/executors/e2/chainKernelEmit.ts)           | Multi-LHS chain C kernel (escape outputs, chain-locals, optional `in_<lhs>` inputs)                |
| [reductionKernelEmit.ts](../../../src/numbl-core/executors/e2/reductionKernelEmit.ts)   | Chain + trailing reduction (or standalone reduction) C kernel — extra `out_acc` slot               |
| [emitShared.ts](../../../src/numbl-core/executors/e2/emitShared.ts)                     | Shared helpers used by both emitters (name mangling, param/koffi list builders, FusedTarget, etc.) |
| [multiReductionDriver.ts](../../../src/numbl-core/executors/e2/multiReductionDriver.ts) | Detects N≥2 reductions over the same tensor; reuses e1's `multiReductionKernel`                    |
| [scalarFnDriver.ts](../../../src/numbl-core/executors/e2/scalarFnDriver.ts)             | `tryE2ScalarFn` — whole-function scalar C kernel (hook in `callUserFunction`)                      |
| [liveness.ts](../../../src/numbl-core/executors/e2/liveness.ts)                         | `isNameReferencedOutsideStmts` — recursive scope-body scan with stmt-identity exclusion            |
| [assignKernel.ts](../../../src/numbl-core/executors/e2/assignKernel.ts)                 | `tryE2Assign` driver — dispatches between chain/reduction paths                                    |
| [cache.ts](../../../src/numbl-core/executors/e2/cache.ts)                               | `WeakMap<Stmt, Map<sig, CompiledFn                                                                 | BAILED>>` — chain & reduction kernels share one cache |
| [compileFn.ts](../../../src/numbl-core/executors/e2/compileFn.ts)                       | Browser-safe indirection for the compile driver                                                    |
| [install.ts](../../../src/numbl-core/executors/e2/install.ts)                           | Node-only side-effect import wiring `compileAndLoad`                                               |

## Kernel shapes

**Chain only** (`r = x + y; r = r .* y + 3.0;`):

```c
void e2c_<hash>(int64_t n, const double *in_x, const double *in_y, double *out_r) {
    #pragma omp simd
    for (int64_t i = 0; i < n; i++) {
        double r;
        r = (in_x[i] + in_y[i]);
        r = ((r * in_y[i]) + 3.0);
        out_r[i] = r;
    }
}
```

**Chain + trailing reduction** (`r2 = x .* y + 0.5; r2 = exp(-r2 .* r2); r2 = r2 .* x; chain_acc = chain_acc + sum(r2);`):

```c
void e2r_<hash>(int64_t n, const double *in_x, const double *in_y, double *out_acc) {
    double acc = 0.0;
    #pragma omp simd
    for (int64_t i = 0; i < n; i++) {
        double r2;
        r2 = ((in_x[i] * in_y[i]) + 0.5);
        r2 = exp(((-r2) * r2));
        r2 = (r2 * in_x[i]);
        acc += r2;
    }
    *out_acc = acc;
}
```

`r2` is purely chain-local — no buffer materialized. The JS side reads
`acc` from `Float64Array(1)` and combines it with the existing env value
of `chain_acc` per `accOp` (`+` here).

**Multi-LHS multi-output** (`c1 = x > 0; c2 = y < 0.5;` where both
escape because they're read by an outside stmt):

```c
void e2c_<hash>(int64_t n, const double *in_x, const double *in_y,
                double *out_c1, double *out_c2) {
    #pragma omp simd
    for (int64_t i = 0; i < n; i++) {
        double c1, c2;
        c1 = (((double)((in_x[i]) > (0.0))));
        c2 = (((double)((in_y[i]) < (0.5))));
        out_c1[i] = c1;
        out_c2[i] = c2;
    }
}
```

When the trailing reduction's target expression is `c1 .* c2` (so both
`c1` and `c2` are referenced ONLY by the chain and the reduction
stmt), the driver folds everything together — `c1` and `c2` become
chain-locals and the reduction kernel runs in one pass with no escape
buffers.

**Multi-reduction over same tensor** (`red_acc = red_acc + (sum(x) +
mean(x) + max(x) + min(x))`): the driver invokes
[multiReductionKernel.ts](../../../src/numbl-core/jit/multiReductionKernel.ts)
to emit one kernel that computes every reduction in a single pass,
then substitutes the kernel's scalar outputs back into the residual
expression and lets the interpreter evaluate the result.

**Whole-function scalar kernel** (e.g.
`benchmarks/scalar_bench.m`'s `run_bench(N, M)`): invoked from
`callUserFunction` before the interpreter runs the body. When all args
are scalar number/boolean and all outputs are scalar, we call
`lowerFunction` + `checkCFeasibility` + `generateC` — the same pipeline
`--opt 2` and `--opt e1` use — then dispatch via koffi:

```c
void jit_run_bench(double N, double M, double *__out_total) {
    double total, x, acc;
    total = 0.0;
    for (double i = 1.0; i <= N; i += 1.0) {
        x = i * 0.001;
        acc = 0.0;
        for (double k = 1.0; k <= M; k += 1.0) {
            acc = acc + sin(x * k) / (k * k);
        }
        total = total + acc;
    }
    *__out_total = total;
}
```

This path is how e2 matches e1's scalar-function performance without
routing through a JS-JIT wrapper. The cache is per-FunctionDef, keyed
by `nargout|argTypeSig`, with progressive widening so small variants
of the same call site coalesce.

## Liveness

The `_currentScopeBody` and `_currentScopeExports` interpreter fields
record the innermost enclosing function (or top-level script) body
and its exported names:

- For a function call: `_currentScopeBody = fn.body`,
  `_currentScopeExports = new Set(fn.outputs)`.
- For top-level scripts: `_currentScopeBody = ast.body`,
  `_currentScopeExports = null` (script-level: every name escapes
  because the caller can read `result.variableValues`).

A chain LHS escapes if either:

1. it's in `_currentScopeExports` (or that field is `null`), OR
2. it's textually referenced in `_currentScopeBody` outside the chain
   stmts (and the trailing-reduction stmt, if any). The recursive
   walker in [liveness.ts](../../../src/numbl-core/executors/e2/liveness.ts)
   excludes the chain stmts at every nesting level.

If neither, the LHS is purely chain-local: kept as a per-element
stack-local with no buffer materialization.

## Buffer reuse

When the LHS already holds a unique-reference Float64 tensor of the
right length, its data buffer is reused as the kernel's output (no
fresh allocation per call). This is safe for the elemwise + reduction
shapes because each iteration reads index `i` before writing index
`i`, so future iterations still see the original values at their
indices.

## Cache

Per-Stmt `WeakMap<Stmt, Map<sig, CompiledFn|BAILED>>`. The signature
includes input names + types, partition lists (regular tensor inputs,
in_lhs inputs, scalar inputs, escape outputs), chain length, and
reduction details (op + accumulate variant + binary op). Different
specializations for the same Stmt — e.g. one without and one with the
LHS in env — get separate cache entries.

The multi-reduction driver has its own cache keyed similarly.

## Threshold

`NUMBL_E2_MIN_ELEMS` (default 1000) — the largest tensor input must
have at least this many elements before we'll attempt to compile.
Below this, koffi marshalling overhead dominates and the interpreter
is faster.

## Parallelism (`--par`)

With `--par`, the chain and reduction emitters upgrade their per-
element loops from `#pragma omp simd` to `#pragma omp parallel for
simd [reduction(...)] if(n >= NUMBL_OMP_THRESHOLD)` when the per-
element body does non-trivial work. "Non-trivial" = at least one heavy
op (transcendentals, `pow`, `atan2`, `hypot`) per iteration — for
arithmetic-only bodies, thread-spawn overhead exceeds the memory-
bandwidth-bound compute, so the pragma would slow things down. The
heuristic is shared with e1 (see
[heavyOps.ts](../../../src/numbl-core/jit/heavyOps.ts)).

Reduction kernels need an OpenMP-expressible reduction clause
(sum/mean/prod/max/min); `any`/`all` use if-update patterns instead
and stay serial-SIMD under `--par`. The cache signature includes
`par=0|1` so the serial and parallel specializations are kept
separate.

# Element-Wise Fusion

Fusion collapses an element-wise tensor assignment into a single inline per-element loop instead of a chain of runtime-helper calls, each of which would allocate an intermediate tensor. The goal is fewer allocations, fewer passes over memory, and better cache behavior.

**Fusion is a C-backend (`--opt 2`) optimization only.** The JS backend never fuses: it always routes tensor ops through the per-op runtime helpers (`jit_tensor_<op>_tt` and friends in `tensor_elemwise_real.js`), which hand tensors of 1024 or more elements to the interpreter's element-wise kernels (the `$elemwise` host hook: native addon when loaded) and loop in JS below that. The pass lives in the JIT compiler's C codegen (`src/numbl-core/jit/codegen/emitTensorFused.ts`).

## What is fused

`emitTensorFused` handles a single `Assign` whose right-hand side is a _pure element-wise_ expression, built only from numeric literals, variables, `Binary`/`Unary` ops, and calls to builtins marked `elementwise` (whose `emitC` also works on scalar arguments), and whose multi-element operands all share the target's shape. It emits one C loop that computes each output slot directly, allocating only the result tensor.

`emitTensorFused` fuses one `Assign`; it does not merge a sequence of statements. Since lowering puts each tensor sub-expression in its own ANF temp, a chain like `_t1 = a * b; c = _t1 + d` would fuse as two loops with an intermediate tensor. To avoid that, `compileSpecC` runs an IR-to-IR inline pass (`codegen/inlinePass.ts`) on the program just before emit, which folds the chain into a single `c = (a * b) + d`.

## Inline pass

The pass substitutes a producer `Assign`'s RHS into its consumer and drops the producer when all of the following hold:

- the producer is itself fusable (`isFusableAssign`);
- its variable is read exactly once in the whole function, nested bodies included, and is not a function output (function-wide counting is what keeps a def inside a loop body that is also read after the loop);
- the consumer is the next statement in the same body that reads it, is an `Assign`, and reads it as a `Binary`/`Unary`/`Call` operand; statements in between do not write the variable or any operand of the producer and are not control flow (the consumer itself may overwrite an operand, as in `v = v + 0.01 .* (a .* v)`, since the RHS is evaluated before the store);
- the consumer is still fusable after substitution, so a nested tensor expression never reaches the helper-call path, where its intermediates would not be freed.

Chains of any depth collapse in one forward sweep per body. The pass returns a new program and never mutates its input: the `Lowerer`'s specialization cache is shared with the JS backend, which keeps the ANF form. JS does not run the pass because it does not fuse; nested tensor expressions there would still allocate one array per operation.

When a fused assign needs a runtime shape guard (dynamic extents) and the guard fails, the fallback evaluates the nested RHS through the broadcasting helpers with each interior tensor in a block-local temp that is freed after the assign (`emitFusedFallback` in `emitStmt.ts`).

On a `y = a.*b + c.*d - e` loop over 2M-element vectors, inlining reduces the per-statement cost at `--opt 2` by roughly 2.5×, mainly by removing the allocation and memory traffic of the intermediate tensors. The `ew_arith` case of `npm run bench` times the same statement.

## How it works

The fused emitter walks the RHS expression and emits, for each output element, the scalar C expression for that slot (builtins contribute their `emitC` output on scalar arguments). The loop iterates over the flat element count of the freshly allocated result tensor and writes `_r.real[i] = <expr>` (plus the imaginary part for complex results).

There are two shape cases:

- **Statically-equal shapes**: emit the loop directly.
- **Same static dim _pattern_ but dynamic extents** (e.g. `ones(1,n) + ones(1,m)`): guard the loop with a runtime shape-equality probe (`jit_fused_shape_eq`); on mismatch it falls back to the broadcasting runtime helper.

If any multi-element operand has a statically-different shape from the target, the assignment falls back to the runtime-helper broadcast path (`jit_tensor_<op>_bcast_tt(...)`) rather than fusing.

## What breaks fusion

- A right-hand side that isn't purely element-wise: indexing, `reshape`, matrix multiply, **reductions** (`sum`, `prod`, `max`, `min`, …), or any builtin call not marked `elementwise`.
- Statically-different operand shapes (broadcasting outside the shared fast path: handled by the runtime helper instead).
- Across statements: an intermediate that is read more than once, is a function output, or whose chain is interrupted by control flow or a write to one of its operands stays materialized (the inline pass leaves it in place).
- The JS backend (`--opt 1`), which never fuses.

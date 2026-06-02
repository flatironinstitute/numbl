# Element-Wise Fusion

Fusion collapses an element-wise tensor assignment into a single inline per-element loop instead of a chain of runtime-helper calls, each of which would allocate an intermediate tensor. The goal is fewer allocations, fewer passes over memory, and better cache behavior.

**Fusion is a C-backend (`--opt 2`) optimization only.** The JS backend never fuses — it always routes tensor ops through the shape-checking `tt_kernel` runtime helpers. The pass lives in the JIT compiler's C codegen (`src/numbl-core/jit/codegen/emitTensorFused.ts`).

## What is fused

`emitTensorFused` handles a single `Assign` whose right-hand side is a _pure element-wise_ expression — built only from numeric literals, variables, `Binary`/`Unary` ops, and element-wise builtin calls that provide a per-slot C hook (`perSlotC`) — and whose multi-element operands all share the target's shape. It emits one C loop that computes each output slot directly, allocating only the result tensor.

Multi-statement chains are collapsed earlier, by the `--inline-temps` pass (`codegen/inlinePass.ts`, with `liveness.ts`): it folds a chain like `_t1 = a * b; c = _t1 + d` into a single `c = (a * b) + d` assignment, which the fused emitter then renders as one loop with no intermediate tensor allocation. `emitTensorFused` itself fuses one already-folded `Assign`; it does not merge a sequence of statements on its own.

## How it works

The fused emitter walks the RHS expression and emits, for each output element, the scalar C expression for that slot (builtins contribute their `perSlotC` snippet). The loop iterates over the flat element count of the freshly allocated result tensor and writes `_r.real[i] = <expr>` (plus the imaginary part for complex results).

There are two shape cases:

- **Statically-equal shapes** — emit the loop directly.
- **Same static dim _pattern_ but dynamic extents** (e.g. `ones(1,n) + ones(1,m)`) — guard the loop with a runtime shape-equality probe (`mtoc2_fused_shape_eq`); on mismatch it falls back to the broadcasting runtime helper.

If any multi-element operand has a statically-different shape from the target, the assignment falls back to the runtime-helper broadcast path (`mtoc2_tensor_<op>_bcast_tt(...)`) rather than fusing.

## What breaks fusion

- A right-hand side that isn't purely element-wise: indexing, `reshape`, matrix multiply, **reductions** (`sum`, `prod`, `max`, `min`, …), or any builtin call without a `perSlotC` hook.
- Statically-different operand shapes (broadcasting outside the shared fast path — handled by the runtime helper instead).
- The JS backend (`--opt 1`), which never fuses.

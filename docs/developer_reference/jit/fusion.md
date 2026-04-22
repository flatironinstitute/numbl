# Element-Wise Fusion

Fusion collapses a run of element-wise tensor assignments into a single per-element loop. The goal is fewer allocations, fewer passes over memory, and better cache behavior.

## What is fused

A consecutive sequence of IR assignments where:

- each right-hand side is an element-wise combination of previous intermediates and tensor inputs of compatible shape;
- intermediates are not observed outside the chain;
- the trailing assignment writes a sink that is allowed to be produced per-element.

A trailing reduction (`sum`, `prod`, `max`, `min`, etc.) can be absorbed into the fused loop as an accumulator, removing the need to materialize the last intermediate at all.

## How it works

A single analysis pass inspects the IR and produces a fusion plan: which statements merge, which variables remain live, which intermediates become scalar temporaries inside the loop. Both the JS and C backends consume the same plan:

- **JS** emits a plain `for (let i = 0; i < n; i++)` loop with typed-array access and scalar temporaries.
- **C** emits a C loop with pointer math; inner operations are either inlined or dispatched through the ops-library op codes.

If the fused form cannot be emitted (complex tensors in a backend that doesn't handle them, an op without a usable `jitEmit`/`jitEmitC`), fusion is skipped for that chain and the individual assignments are emitted.

## What breaks fusion

- Non-element-wise operations in the middle of the chain (indexing, reductions other than the trailing one, reshape).
- Shape mismatches (broadcasting rules outside the shared fast path).
- Side-effecting or opaque builtins.
- Any output being observed between statements in the chain.

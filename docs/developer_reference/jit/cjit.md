# C-JIT

Orientation for `--opt e3`.

## What C-JIT is for

C-JIT only beats JS-JIT in one regime: code where dispatch can be **statically eliminated entirely**, leaving raw arithmetic on contiguous memory. Outside that regime V8's inline caches and escape analysis are doing real work that a naive C version cannot match.

So the slice is narrow on purpose: scalar numeric loops, dense element-wise tensor ops, and short chains of those. Anything richer — struct fields, cell arrays, function-handle calls, dynamic dispatch — belongs to JS-JIT or the interpreter. **C-JIT's value is depth on a narrow slice, not breadth.**

## Trigger contexts

Three executors compete on each dispatch, each consuming a different shape:

- **`c-jit-loop`** — whole `for` / `while` body, scalar (real or complex via pair-of-doubles).
- **`c-jit-fuse`** — single element-wise tensor `Assign`.
- **`c-jit-chain`** — run of ≥2 adjacent element-wise Assigns, fused into one kernel.

All three share a common substrate: the same builtin set, the same elementwise AST classifier, the same codegen helpers. New shapes that don't fit one of these contexts need a shared-substrate change, not a fourth executor with parallel whitelists.

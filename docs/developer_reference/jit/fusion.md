# Element-Wise Fusion

Fusion collapses an element-wise tensor assignment into a single per-element loop instead of a chain of runtime-helper calls, each of which would allocate an intermediate tensor and make its own pass over memory. Both backends fuse. The decision is made once, on the IR, by a shared pass that produces an explicit `ElemwiseKernel` node; each backend only renders that node.

## The kernel pass

`compileWith` (`jit/compile.ts`) runs `kernelPass` (`jit/lowering/elemwiseKernels.ts`) on the program just before emit, with the target's `FusionPolicy`. The pass returns a new program and never mutates its input: the `Lowerer`'s specialization cache is shared by both backends.

Lowering puts each tensor sub-expression in its own ANF temp, so `y = a.*b + c` arrives as `_t1 = a .* b; y = _t1 + c`. The pass works in two steps per statement list:

1. **Inline.** A single-use producer is substituted into its consumer, so the chain becomes `y = (a .* b) + c`. The producer must itself be an element-wise kernel assignment; its variable must be read exactly once in the whole function (nested bodies included) and not be an output; the consumer must be the next statement in the same body that reads it, read it as a `Binary` / `Unary` / `Call` operand, and still be a kernel assignment after substitution; statements in between must not write the variable or any operand of the producer and must not be control flow. The consumer itself may overwrite an operand, as in `v = v + 0.01 .* (a .* v)`, since the RHS is evaluated before the store. Chains of any depth collapse in one forward sweep.
2. **Build.** Each remaining kernel assignment gets an `ElemwiseKernel` RHS.

ANF stays in lowering because a consumer that does not fuse (a matrix product, an indexed store) needs its tensor operands in named temps that codegen can free; the pass undoes it only where it fuses.

## The `ElemwiseKernel` node

An `ElemwiseKernel` (`jit/lowering/ir.ts`) appears only as an `Assign` RHS and holds:

- `inputs`: the distinct multi-element operands, in first-appearance order. `inputs[0]` is the shape source: the output has its runtime shape.
- `body`: the per-element expression. Every node is scalar-typed, and each read of a tensor operand is a `KernelElem` (element `i` of input `k`). This is the form the builtins' scalar `emitC` / `emitJs` hooks expect, so a backend renders the body with the same hooks it uses for scalar code.
- `guard`: set when there are two or more distinct operands and some dimension is not exact. Their static dimension patterns match, but their runtime extents may not (`ones(1,n) + ones(1,m)`), so the emitted loop checks the shapes first and otherwise evaluates `tree` through the broadcasting helpers, which broadcast compatible shapes and raise "Matrix dimensions must agree" on incompatible ones.
- `reduce` (optional): `sum`, `prod` or `mean` over all elements. The kernel's value is then a scalar, accumulated in linear order from `0` (or `1` for `prod`), the order of the `jit_<op>_all` helpers, so the result is bit-identical to reducing the materialized tensor.
- `tree`: the expression the kernel replaces. IR walkers (`walk.ts`) visit it, so liveness, the definite-assignment reads and the cancel-poll classification see the same variables as before the rewrite.

## What fuses

A kernel assignment is either

- **element-wise**: a real, multi-element result whose RHS consists only of numeric literals, variables, and `Binary` / `Unary` / `Call` nodes over builtins marked `elementwise` (`mtimes` / `mrdivide` only with a scalar side; a tensor-valued call only when its tensor arguments have its own static shape), with every multi-element operand sharing the result's static dimension pattern and no complex tensor anywhere; or
- **reducing**: `sum(x)`, `prod(x)` or `mean(x)` with a real scalar result (a reduction over all elements, which the transfer function picks for a vector or `'all'`-like shape) where `x` is a non-trivial element-wise tree of the above form. A reduction of a bare variable stays a helper call.

The `FusionPolicy` narrows this per target:

| Target | Builtins in a kernel                                                                                                                                     | Types                         |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| C      | every element-wise builtin                                                                                                                               | any the C emitter handles     |
| JS     | `plus`, `minus`, `times`, `rdivide`, `mtimes` / `mrdivide` (scalar side), `uminus`, comparisons, `abs`, `floor`, `ceil`, `fix`, `sqrt`, and the reducers | real double or logical values |

The JS set is restricted to operations whose per-element result IEEE arithmetic fixes exactly. The JS per-op helpers hand tensors of 1024 or more elements to the interpreter's kernels (the `$elemwise` host hook: the native addon when loaded), whose `exp`, `sin`, `pow` and so on come from libm and need not agree with V8's `Math` functions in the last bit. A transcendental call therefore stays a helper call in its own temp at `--opt 1`, and the kernel starts above it: `exp(a) .* b + 1` becomes `_t1 = exp(a)` plus one kernel over `_t1` and `b`.

## Rendering

- **C** (`codegen/emitKernelC.ts`): allocate the result with the shape source's dims, then one `JIT_OMP_PARFOR_N` loop writing `_r.real[_jit_i] = <body>`, with tensor reads as `<input>.real[_jit_i]`. A reducing kernel accumulates into a `double` in a sequential loop. A guarded kernel's fallback hoists each tensor-valued interior node of `tree` into a block-local temp freed after the store (`emitKernelFallback` in `emitStmt.ts`).
- **JS** (`codegen/emitKernelJs.ts`): bind each input's `data` array once, allocate the result (tagged logical for a logical result), and run one loop. The guard calls `jit_fused_shape_eq`; the fallback is `tree` through the per-op helpers. Kernel locals use the `_jit_e` prefix, which no other emitted name uses.

The C output for element-wise kernels is the same as before the kernel IR existed (the `jit_parity` corpus emits identical C).

## What breaks fusion

- An RHS that is not element-wise: indexing, `reshape`, matrix multiply, a reduction other than a trailing full `sum` / `prod` / `mean` (a reduction along one dimension of a matrix, `max`, `any`, ...), or any builtin call not marked `elementwise`.
- Statically different operand shapes (broadcasting: handled by the runtime helper instead). Tensor comparisons currently type their result with unknown dimensions, so a comparison fuses only with operands whose dimensions are unknown too.
- Complex tensors (both targets), and at `--opt 1` the builtins outside the JS set.
- Across statements: an intermediate that is read more than once, is a function output, or whose chain is interrupted by control flow or a write to one of its operands stays materialized.

## Performance

On `y = a.*b + c.*d - e` over 2M-element vectors, fusing at `--opt 2` reduces the per-statement cost by roughly 2.5x, mainly by removing the allocation and memory traffic of the intermediate tensors. At `--opt 1` the gain is smaller: V8 does not vectorize the loop and keeps a bounds check per array, so a fused JS loop runs at roughly 2 to 3 ns per element, against the host's SIMD kernels per operation. The `ew_*` and `sum_*` cases of `npm run bench` time these statements.

Element-wise statements that no other JIT shape reaches (in a script or function body that was not compiled, or in a declined loop) can be compiled on their own; see the `stmt` shape in [executors.md](../executors.md#the-statement-shape).

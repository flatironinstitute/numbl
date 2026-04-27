# e1 kernels

Under `--opt e1`, the outer function still runs as JS-JIT. On top of that
the JIT splices compiled C kernels at two well-defined points:

1. **Fusible tensor chains** inside a JS-JIT body — e.g. a run of
   consecutive element-wise assigns. The chain becomes a single C kernel
   loaded via koffi; the surrounding JS dispatches to it when `N` exceeds
   a size threshold (otherwise falls back to the plain JS fused loop).
2. **Pure-scalar user functions** — the entire function body becomes a
   C kernel; the JS wrapper marshals scalar params in and reads scalar
   outputs back out. See [scalarFnKernel.ts](../../../src/numbl-core/jit/e1/scalarFnKernel.ts).

The C backend infrastructure under [jit/c/](../../../src/numbl-core/jit/c/)
supplies the shared emitter (`generateC` in `assemble.ts`), the feasibility
gate (`feasibility.ts`), and the cc / koffi driver (`compile.ts`). The e1
path gates its callers to the scalar-only envelope that infrastructure
supports cleanly; chain kernels are emitted by e1's own walkers
([kernelEmit.ts](../../../src/numbl-core/jit/e1/kernelEmit.ts) for real,
[complexKernelEmit.ts](../../../src/numbl-core/jit/e1/complexKernelEmit.ts)
for complex).

## Files

[jit/e1/](../../../src/numbl-core/jit/e1/):

| File                                                                        | Role                                                                |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [kernelEmit.ts](../../../src/numbl-core/jit/e1/kernelEmit.ts)               | Real-tensor chain kernel emitter (per-element body → C `for` loop)  |
| [complexKernelEmit.ts](../../../src/numbl-core/jit/e1/complexKernelEmit.ts) | Complex-tensor chain kernel emitter (paired re/im buffers)          |
| [scalarFnKernel.ts](../../../src/numbl-core/jit/e1/scalarFnKernel.ts)       | Whole-function scalar kernel emitter — delegates to `c/assemble.ts` |
| [install.ts](../../../src/numbl-core/jit/e1/install.ts)                     | Wires `$h.compileKernel` to `compileAndLoad` via koffi (Node-only)  |
| [openmpFlag.ts](../../../src/numbl-core/jit/e1/openmpFlag.ts)               | Browser-safe getter for `isOpenmpAvailable()`                       |
| [hash.ts](../../../src/numbl-core/jit/e1/hash.ts)                           | FNV-1a 64-bit string hash used for content-addressed kernel keys    |

[jit/c/](../../../src/numbl-core/jit/c/) (shared C-emitter infrastructure
used by e1's scalar-kernel path):

| File                                                           | Role                                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [feasibility.ts](../../../src/numbl-core/jit/c/feasibility.ts) | Walks IR and returns `{ok}` / `{ok:false, reason}`                       |
| [classify.ts](../../../src/numbl-core/jit/c/classify.ts)       | `analyzeTensorUsage` — single pass producing the `TensorMeta` table      |
| [context.ts](../../../src/numbl-core/jit/c/context.ts)         | Shared `EmitCtx`, mangle / scratch helpers, op-code maps                 |
| [abi.ts](../../../src/numbl-core/jit/c/abi.ts)                 | ABI slot schema used by the JS wrapper                                   |
| [assemble.ts](../../../src/numbl-core/jit/c/assemble.ts)       | Top-level `generateC` — joins callee emissions + outer into one C source |
| [prelude.ts](../../../src/numbl-core/jit/c/prelude.ts)         | Function prelude construction                                            |
| [epilogue.ts](../../../src/numbl-core/jit/c/epilogue.ts)       | Function epilogue construction                                           |
| [compile.ts](../../../src/numbl-core/jit/c/compile.ts)         | `cc` driver + koffi loader + content-addressed cache                     |
| [visit.ts](../../../src/numbl-core/jit/c/visit.ts)             | Shared `walkExprNodes` / `walkStmts` / `walkStmtExprs` traversal helpers |

Per-statement / per-expression C emission lives in [jit/c/emit/](../../../src/numbl-core/jit/c/emit/):

| File                                                                         | Role                                                                                      |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [emit/index.ts](../../../src/numbl-core/jit/c/emit/index.ts)                 | Barrel — exports `emitStmts`, `shapeExprsFor`                                             |
| [emit/stmt.ts](../../../src/numbl-core/jit/c/emit/stmt.ts)                   | Top-level stmt dispatch + fusion hook + `withPendingStmts`                                |
| [emit/scalar.ts](../../../src/numbl-core/jit/c/emit/scalar.ts)               | Real scalar expr (`emitExpr`, Binary/Unary/Call/Index/Truthiness)                         |
| [emit/complexScalar.ts](../../../src/numbl-core/jit/c/emit/complexScalar.ts) | Complex scalar expr (`emitComplex`, Smith div, materialize)                               |
| [emit/tensor.ts](../../../src/numbl-core/jit/c/emit/tensor.ts)               | Tensor expr → scratch (real + complex), shape inference, buffer sizing, range-slice reads |
| [emit/assign.ts](../../../src/numbl-core/jit/c/emit/assign.ts)               | Tensor assign forms (real + complex + fresh-alloc + reduction-of-expr)                    |
| [emit/userCall.ts](../../../src/numbl-core/jit/c/emit/userCall.ts)           | UserCall emission (scalar + tensor-return via dynamic ABI)                                |
| [emit/fused.ts](../../../src/numbl-core/jit/c/emit/fused.ts)                 | Fused per-element loop (real + complex paths)                                             |
| [emit/helpers.ts](../../../src/numbl-core/jit/c/emit/helpers.ts)             | Small shared helpers + type guards                                                        |

## External entry points

The CLI installs the Node-only kernel compiler via side-effect import of
[e1/install.ts](../../../src/numbl-core/jit/e1/install.ts). From there,
[executors/jsJit/jitCall.ts](../../../src/numbl-core/executors/jsJit/jitCall.ts) (scalar-fn path) and
[jit/js/jsFusedCodegen.ts](../../../src/numbl-core/jit/js/jsFusedCodegen.ts)
(chain path) do the splicing during JS codegen.

## Parallelism

`--par` emits `#pragma omp parallel for simd` on fusible chain kernels
whose per-element body has at least one heavy op (transcendentals like
`exp`, `sin`, ...). Arithmetic-only chains stick to `#pragma omp simd`
because thread-spawn overhead exceeds the memory-bandwidth-bound cost.
The threshold where the parallel-for actually fires is `NUMBL_OMP_THRESHOLD`
(default 100 000 elements).

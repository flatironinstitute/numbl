# C-JIT

The C-JIT backend (`--opt 2`) emits a C translation unit for each hot
function, compiles it via `cc` into a shared object, loads it through
koffi, and calls it from the interpreter via a JS wrapper.

It sits on top of the same lowered IR the JS-JIT consumes and falls
back to JS-JIT whenever an IR construct isn't in the supported subset.

## Pipeline

```
lowered IR (JitStmt[] / JitExpr)
        │
        ▼
feasibility.ts  ──────────────►  fall back to JS-JIT on a miss
        │
        ▼
classify.ts       — single pass builds the TensorMeta table every
                    downstream stage reads from
        │
        ▼
assemble.ts       — top-level orchestration: collects reachable
                    UserCall callees, emits each as a static C fn,
                    then emits the outer; joins headers + all fn defs
                    into one C source string
        │
        ├── prelude.ts    — param-output seeds, unshare copies,
        │                   local tensor decls, complex scalar inits,
        │                   scratch slot declarations
        │
        ├── emit/         — per-stmt / per-expr C emission (see below)
        │
        ├── epilogue.ts   — output-slot writes, scratch frees,
        │                   local / unshared tensor frees
        │
        └── abi.ts        — build the ABI slot schedule used by
                            both the C signature and the JS wrapper
        │
        ▼
compile.ts        — shell out to cc, content-addressed cache under
                    ~/.cache/numbl/c-jit/, load via koffi
        │
        ▼
install.ts        — JS wrapper that marshals calls per the ABI slot
                    schedule; registered as the CJitBackend via
                    registry.ts
```

## Files

Subsystem level (one file each):

| File                                                           | Role                                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [registry.ts](../../../src/numbl-core/jit/c/registry.ts)       | `CJitBackend` interface + module-level registration slot                 |
| [feasibility.ts](../../../src/numbl-core/jit/c/feasibility.ts) | Walks the IR and returns `{ok}` / `{ok:false, reason}`                   |
| [classify.ts](../../../src/numbl-core/jit/c/classify.ts)       | `analyzeTensorUsage` — single pass producing the `TensorMeta` table      |
| [context.ts](../../../src/numbl-core/jit/c/context.ts)         | Shared `EmitCtx`, mangle / scratch helpers, op-code maps                 |
| [abi.ts](../../../src/numbl-core/jit/c/abi.ts)                 | ABI slot schema shared with the JS wrapper                               |
| [assemble.ts](../../../src/numbl-core/jit/c/assemble.ts)       | Top-level `generateC` — joins callee emissions + outer into one C source |
| [prelude.ts](../../../src/numbl-core/jit/c/prelude.ts)         | Function prelude construction                                            |
| [epilogue.ts](../../../src/numbl-core/jit/c/epilogue.ts)       | Function epilogue construction                                           |
| [compile.ts](../../../src/numbl-core/jit/c/compile.ts)         | `cc` driver + koffi loader + content-addressed cache                     |
| [install.ts](../../../src/numbl-core/jit/c/install.ts)         | JS wrapper for invoking compiled functions                               |
| [hybrid.ts](../../../src/numbl-core/jit/c/hybrid.ts)           | JS-outer + C-callee / extracted-loop compilation                         |
| [parityError.ts](../../../src/numbl-core/jit/c/parityError.ts) | Error class for `--check-c-jit-parity`                                   |
| [visit.ts](../../../src/numbl-core/jit/c/visit.ts)             | Shared `walkExprNodes` / `walkStmts` / `walkStmtExprs` traversal helpers |

Per-statement / per-expression emission lives in [emit/](../../../src/numbl-core/jit/c/emit/):

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

Callers outside `jit/c/` only see three names:

- `getCJitBackend()` from [registry.ts](../../../src/numbl-core/jit/c/registry.ts)
- `compileHybridCallees`, `compileHybridLoops` from [hybrid.ts](../../../src/numbl-core/jit/c/hybrid.ts)
- `CJitParityError`, `formatCJitParityMessage` from [parityError.ts](../../../src/numbl-core/jit/c/parityError.ts)

The CLI installs the Node-only backend via side-effect import of
[install.ts](../../../src/numbl-core/jit/c/install.ts).

## Adding a new supported construct

1. Extend [feasibility.ts](../../../src/numbl-core/jit/c/feasibility.ts) to accept the new shape.
2. Extend the matching emitter under [emit/](../../../src/numbl-core/jit/c/emit/).
3. If a new JitExpr / JitStmt tag is involved, update [visit.ts](../../../src/numbl-core/jit/c/visit.ts).
4. For new tensor ops, update the op-code helpers in [context.ts](../../../src/numbl-core/jit/c/context.ts) (`TENSOR_BIN_OP`, `getTensorUnaryOp`, etc.).
5. Run `npm run test:scripts:c-jit` — the script-level tests stamp every
   feasible path.

Real and complex tensor paths in [emit/tensor.ts](../../../src/numbl-core/jit/c/emit/tensor.ts) and [emit/assign.ts](../../../src/numbl-core/jit/c/emit/assign.ts) are structurally parallel; adding a tensor op usually means one edit in each.

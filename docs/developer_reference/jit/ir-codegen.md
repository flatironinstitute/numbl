# JIT IR and Codegen

## IR

The JIT lowers the parser AST to a typed intermediate representation before emitting target code. Expressions and statements are tagged discriminated unions.

- **Expressions (`JitExpr`)** — number literal, variable, binary/unary op, generic call, user call, index read, range-slice read, tensor literal, vertical-concat growth, member read, struct-array member read, function-handle call, user-dispatch call.
- **Statements (`JitStmt`)** — assign, assign-via-index (full and specialized column / 3-D-page / range forms), assign-member, if, for, while, multi-assign, break/continue/return, set-location (for diagnostics).

Every IR node carries its resolved `JitType`. This is the contract between lowering and codegen: types are fixed at this point, so codegen never re-decides what an operation means.

## Lowering

Lowering walks the parser AST top-down, tracking variable types in a per-scope environment. For each call it invokes the callee's `IBuiltin.resolve` to get output types and an `apply` function. Loop heads join types from the entry and back-edge; a widening triggers recompilation.

If any construct is unsupported — an uncommon builtin, an unusual control flow pattern, a type the JIT doesn't represent — lowering aborts and the interpreter keeps control.

## JS backend

Emits an ES function as a string, then materializes it with `new Function(args, body)`. Key ingredients:

- A **helpers object** (conventionally `$h`) holds all runtime functions the generated code calls into: tensor allocation, indexed read/write, binary ops, complex arithmetic, copy-on-write helpers, builtin trampolines (`$h.ib_<name>`). The helpers object is fresh per compilation.
- **Inline emission** via `IBuiltin.jitEmit` produces a literal expression string for fast scalar/real-tensor paths. When absent or unsuitable, codegen falls back to the `$h.ib_<name>` trampoline.
- **Bail expressions** throw a typed bail object that the JIT caller unwinds.

## C backend

Emits a single C translation unit containing one function per compiled IR function plus any required helpers. Feasibility is checked first: the C backend supports real scalars and a subset of real tensor ops, enough to cover most numerical inner loops. Non-feasible IR nodes are rejected and the compilation falls back to JS.

- `IBuiltin.jitEmitC` (when defined) emits the C expression for a builtin call.
- For tensor ops with an `jitCapabilities` annotation, the C backend issues the corresponding op-code call into the native ops library — the same kernels used by the interpreter's ops layer.
- The compile step shells out to a C compiler (configurable via `NUMBL_CC` and `NUMBL_CFLAGS`) and loads the resulting `.node` module through Node-API. A hybrid mode wraps native callees in a JS function when the outer IR is JS-only.

## Fusion

Element-wise assignment runs are detected and collapsed into a single per-element loop by a shared analysis. Both backends consume the same fusion plan and emit their own fused form. See [fusion.md](fusion.md).

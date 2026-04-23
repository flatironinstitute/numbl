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

## C kernels under `--opt e1`

Under `--opt e1`, the JS-JIT outer can splice in compiled C kernels at two
boundaries: fusible tensor chains (inside a JS function body) and pure-scalar
user functions (the whole body becomes a C kernel). Kernel source is generated
by the e1 emitters and the shared `generateC` in `jit/c/assemble.ts`, then
compiled via `cc` (configurable via `NUMBL_CC` / `NUMBL_CFLAGS`) and loaded
through koffi. The JS wrapper dispatches calls inline. See
[e1-kernels.md](e1-kernels.md).

- `IBuiltin.jitEmitC` (when defined) emits the C expression for a builtin call inside a kernel body.
- For tensor ops with a `jitCapabilities` annotation, the kernel issues the corresponding op-code call into the native ops library — the same kernels used by the interpreter's ops layer.

## Fusion

Element-wise assignment runs are detected and collapsed into a single per-element loop by a shared analysis. The JS-JIT and the e1 kernel emitters consume the same fusion plan and emit their own fused form. See [fusion.md](fusion.md).

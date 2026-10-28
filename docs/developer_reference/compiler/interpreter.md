# Interpreter

The interpreter is an AST walker. Given a parsed program and a `CodeIndex`, it evaluates statements in order against a `Runtime` and produces `RuntimeValue`s.

## Key objects

- **Interpreter** (`interpreter.ts`): the interpreter's state (the running call frame, the `Runtime`, the run's `CodeIndex`, the executor registry, JIT bookkeeping such as `loopDepth`) and its entry points (`run`, `execStmt`, `installRuntimeCallbacks`).
- **CallFrame** (`callFrame.ts`): one activation, that is the base workspace, a function call or an anonymous-function call. It holds the activation's `Environment`, the file, class and method whose code runs, the frame it was called from (`caller`), and the call-site names of its arguments (`inputNames`, for `inputname`). A frame is the `Scope` names resolve from.
- **Environment**: the variables of an activation. Resolves variable reads and writes, and routes `global` and `persistent` declarations through shared stores on the runtime so their values outlive the call.
- **Runtime**: shared state across a single execution: global/persistent stores, output router, plot instruction accumulator, JIT caches. (RNG state is held in module-level variables in [`helpers/prng.ts`](../../../src/numbl-core/helpers/prng.ts), not on the `Runtime`.)
- **CodeIndex**: owned by the `Workspace` (the visible file set, kept in sync with addpath/rmpath/cd). Caches parsed ASTs, maps workspace files to function names (`+pkg` namespaces, `private/` scopes, `.numbl.js` functions), holds the main file's local functions, and builds the `FunctionIndex` that `resolveFunction` consults. The interpreter and the JIT share the same instance.
- **ClassTable**: part of the `CodeIndex`: `ClassInfo` for workspace and main-file classes (with per-property and per-method attributes), plus inheritance and member queries (lineage, ancestors, defining class, handle-ness, `findProperty`/`findMethod`, access). See [classes.md](classes.md).

## Code layout

The `Interpreter` class holds state only. The evaluation code is in modules of plain functions that take the interpreter as their first argument, under `src/numbl-core/interpreter/`:

| File                 | Contents                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `execStatements.ts`  | statements, control flow, `for`/`while` loops, `%!numbl:assert_jit`                        |
| `evalExpressions.ts` | expressions, operators, call expressions, member access, tensor and cell literals          |
| `indexing.ts`        | indexing reads, `end`, assignment targets (`a(i).f{j} = v`), the same-class subsref bypass |
| `handles.ts`         | anonymous functions and `@name` handles                                                    |
| `calls.ts`           | calling what a name resolves to (user functions, builtins, `.numbl.js`, scripts)           |
| `callConvention.ts`  | binding inputs, running a body in a new frame, collecting outputs; nested functions        |
| `oop.ts`             | instantiation, constructors, superclass calls, class methods, enumerations, Constant props |
| `metaClass.ts`       | `?Name` and `metaclass`: meta.class objects                                                |
| `argumentsBlock.ts`  | `arguments` block defaults and name-value struct parameters                                |
| `callerScope.ts`     | the calling scope that builtins declared with `scope: true` see, eval of text              |

## Call frames

`interp.frame` is the running frame. `callBody` (and `callNestedFunction`) build a new frame for a call, and `runFunctionBody` makes it the running frame for the body and restores the caller's on exit. An anonymous function call likewise runs in a frame of its own (in `handles.ts`), whose environment holds the parameters on top of the captured snapshot and whose caller is the frame that invoked it.

Code that belongs to the running activation but comes from another file runs in a frame from `frame.at(site)`, which shares the environment, caller and input names and only changes the file, class and method: a script called by name, `run(file)`, a `.numbl.js` function, and a `@name` handle (which runs at the site where it was created). `evalin` runs text in `frame.inWorkspaceOf(target)`: the target's variables and input names, with the current site and caller, so that, as in MATLAB, a nested `evalin('caller')` reaches the same caller.

A function handle keeps a detached frame (`frame.detach()`: the site and environment, without the caller chain or input names), not the running frame.

`inputname` reads the frame's `inputNames`. A call expression `f(x, y)` sets the calling frame's `callSiteNames` for the duration of the call, and the callee's frame takes them as its `inputNames`. A function that a builtin calls back during that call (`cellfun(h, c)`) sees the builtin call's names, as in MATLAB.

## Control flow

Control-flow transfers (break, continue, return) are signalled by typed signal objects (`BreakSignal` / `ContinueSignal` / `ReturnSignal`, none of which extend `Error`) that `execStmt` **returns** (its type is `ControlSignal | null`); the enclosing loop or function body inspects the return value with `instanceof` and propagates it upward. They are not thrown. Regular runtime errors, by contrast, flow as normal exceptions and end up in the diagnostics layer.

## Name resolution and calls

A call expression or bare name is resolved by `resolveName` (see [function-resolution.md](function-resolution.md)) from the running frame: its file, class and method, and its `Environment`. The result says what the name is (variable, constant, nested, local, private or workspace function, class method or constructor, package function, builtin, ...), and `callResolved` runs it. A user function's `FunctionDef` comes from `functionBodyFor`, which also gives the file and class context to run it in. A workspace or stdlib `.m` shadows a builtin of the same name.

Builtin calls go through `invokeBuiltin`, which checks the builtin's arity and runs its `apply`. See [builtins.md](../builtins.md). Builtins that need the caller's frame (`eval`, `evalin`, `exist`, `nargin`, `inputname`, ...) declare `scope: true` and read it as `ctx.scope`, a `CallerScope` (`callerScope.ts`) whose frame-derived fields read `interp.frame` at the time of the call. The interpreter hands it to them through `rt.callerScope`, set in `installRuntimeCallbacks`, so they behave the same whether called directly, through a handle, `feval` or `builtin(...)`.

A call used as a statement requests no outputs (nargout 0), also when the callee is reached through indexing (`c{i}()`, `s.f()`). The first output, if the function assigned it (or the first element of `varargout`), becomes `ans`; a function that returns nothing leaves `ans` unset.

## JIT integration

The interpreter decides when to hand control to the JIT:

- **User function calls** are trial-compiled on first hot-enough call.
- **`for` / `while` loops** are candidates when iteration counts look worthwhile.
- **Top-level scripts** are compiled whole when feasible.

If the JIT succeeds, the specialized function or loop runs in place of the interpreted version. A type mismatch at runtime throws a bail signal, the interpreter resumes, and the JIT cache is invalidated. A call the JIT runs does not create a call frame.

## What the interpreter does not do

- It does not type-specialize. Every operation goes through generic `RuntimeValue` dispatch unless a JIT path takes over.
- It does not do parse-time name resolution for user functions; calls are resolved on demand (and cached per function index).
- It does not manage platform I/O directly — all file, stdout/stderr, and plot output flows through callbacks on `ExecOptions`.

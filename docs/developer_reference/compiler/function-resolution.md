# Function resolution

A name in MATLAB code can mean many things depending on where it appears: a variable, a nested or local function, a private function, a method of an object argument, a package function reached through an `import`, a class, a builtin. The module `src/numbl-core/resolution/` answers that question in one place. Every consumer (calls, dotted names, function handles, calls by name from the runtime, `which`, `exist`, `nargin('f')`) goes through it, so they agree.

## Scope

A `Scope` (`scope.ts`) is where a name is looked up from: the `CodeIndex`, the running file, the class and method whose code is running (if any), and the `Environment` holding the variables and nested functions in scope. Its `imports` are the file's `import` statements (numbl tracks imports per file, while MATLAB scopes them per function). The interpreter's running `CallFrame` is a `Scope` (see [interpreter.md](interpreter.md)); a function handle keeps the scope it was created in.

## Bare names: `resolveName`

`resolveName(name, scope, opts)` returns a `NameResolution`, a discriminated union whose kinds are `variable`, `constant`, `nested`, `builtin`, `mainLocal`, `fileLocal`, `private`, `classMethod`, `staticMethod`, `classConstructor`, `packageFunction`, `jsUserFunction`, `workspaceFunction`, and `notFound`. Callable kinds carry a `ResolvedTarget` that `interpretTarget` runs.

The precedence is MATLAB's:

1. variable;
2. nested function;
3. explicit import (`import pkg.fn`);
4. local function of the calling file (the main file, a function file, a classdef file and its `@Class` method files, or a private file);
5. wildcard import (`import pkg.*`);
6. private function of the calling file's folder;
7. method of the dominant object argument (`InferiorClasses` breaks ties);
8. workspace function, then `.numbl.js` function, then class constructor;
9. constant (`pi`, with no arguments), builtin.

Tiers 3 to 9 are `resolveFunction` (`resolveFunction.ts`), which works on the `FunctionIndex` and caches its result per index, call site and object-argument classes. A workspace `.m`, including the bundled stdlib, shadows a builtin, and `.m` wins over `.numbl.js` in the same tier. This includes the builtins that work on the calling scope (`eval`, `exist`, `nargin`, ...): they are ordinary builtins, so a user `exist.m` on the path shadows `exist`, much as a folder earlier on MATLAB's path shadows a MATLAB builtin (MATLAB warns when this happens; numbl does not).

Options select the view a consumer needs. `args` supplies the call's arguments (object arguments select methods; no arguments allows a constant). `variables: false` skips variables, as calls do once the caller has ruled them out. `nested: false` skips nested functions: MATLAB's string-named calls (`feval('f')`) do not see them, while `@f`, `nargin('f')`, `exist` and `which` do.

## Dotted names: `resolveQualified`

`resolveQualified(parts, scope)` handles `pkg.fn`, `pkg.sub.fn`, `pkg.Class`, `Class.staticMethod`, `Class.EnumMember` and `Class.empty`. The name is tried with its first part replaced by an explicit import of that name, then under each wildcard-import namespace, then as written, so `import pkg.*` makes `Class.method` and `sub.fn` reach `pkg.Class.method` and `pkg.sub.fn`. Besides the kinds above it returns `enumMember`, `classConstant` for a Constant property (its value is read without constructing the class, see [classes.md](classes.md)), and `classMember` for `Class.name` where `name` is none of these (an instance method): a call `Class.name(args)` treats it as a method, while a bare `Class.name` evaluates `Class` and reads the member. A variable named like the first part makes the whole name a variable access.

`resolveName` delegates dotted names here, so a call by name such as `feval('pkg.fn')` works too.

## Function bodies

`functionBodyFor(code, target)` (`functionBody.ts`) maps a target to the `FunctionDef` it runs and the file, class and method context to run it in. `constructorBodyFor` and `oldStyleConstructorBodyFor` do the same for class constructors. Bodies are built once per function index, so every call of a function (including private functions and `@Class` method files) sees the same `FunctionDef`, which the JIT's per-function caches key on. `definingFile` gives the file behind a resolution, for `which`.

`resolveClassMethod` (`classMethod.ts`) looks up a method by class name for calls on an object (`obj.method`, property accessors, `subsref`/`subsasgn` overloads), behind the runtime's `resolveClassMethod` hook.

## Consumers

- **Interpreter calls**: `callFunction` (`interpreter/calls.ts`; call expressions and bare names) resolves and then `callResolved`.
- **Dotted names**: `evalMember`, the `MethodCall` path and `makeFuncHandle` use `resolveQualified`.
- **Runtime dispatch by name**: `rt.dispatch` (a handle without a closure, `cellfun` with a function name, operator overloads, `onCleanup`) asks the interpreter's `rt.resolveNamedCall` hook, which resolves from the running code's scope.
- **Introspection**: `which`, `exist` and `nargin('name')` resolve in the caller's scope.
- **JIT**: `jit/workspace/workspace.ts` calls `resolveFunction` directly with call sites that carry only the file, and maps the target to its own shapes.

## MATLAB behavior worth knowing

- `exist` reports nested, local and private functions visible from the caller as 2, a class file as 2 (8 with `'class'`), a package class as 8, and a package function or a class method as 0.
- MATLAB's `cellfun` accepts only a few legacy names (`'isempty'`, `'length'`, ...) as strings, and `arrayfun` none. numbl accepts any function name there and resolves it like `feval`.
- Known divergences: MATLAB rejects an explicit import that has the name of a nested function, where numbl lets the nested function win; and under a wildcard import, MATLAB's `feval('f')` prefers a private `f` over the imported one, where numbl resolves as for a direct call.

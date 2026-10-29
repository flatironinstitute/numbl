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
7. method of the dominant object argument (`InferiorClasses` breaks ties), then a method of a TypeScript-implemented class (see below);
8. workspace function, then `.numbl.js` function, then class constructor;
9. constant (`pi`, with no arguments), builtin.

Tiers 3 to 9 are `resolveFunction` (`resolveFunction.ts`), which works on the `FunctionIndex` and caches its result per index, call site and object-argument classes. A workspace `.m`, including the bundled stdlib, shadows a builtin, and `.m` wins over `.numbl.js` in the same tier. This includes the builtins that work on the calling scope (`eval`, `exist`, `nargin`, ...): they are ordinary builtins, so a user `exist.m` on the path shadows `exist`, much as a folder earlier on MATLAB's path shadows a MATLAB builtin (MATLAB warns when this happens; numbl does not).

Options select the view a consumer needs. `args` supplies the call's arguments (object arguments select methods; no arguments allows a constant). `variables: false` skips variables, as calls do once the caller has ruled them out. `nested: false` skips nested functions: MATLAB's string-named calls (`feval('f')`) do not see them, while `@f`, `nargin('f')`, `exist` and `which` do.

## Object arguments and operators

Which class a value dispatches to is one rule, `dispatchClassOf(v)` (`runtime/kinds.ts`): a classdef object or object array dispatches to its class; any other object kind (`isObject` in its kind descriptor) dispatches to its class only when that class has methods registered from TypeScript; values of the built-in types (double, char, logical, cell, struct, function handle) dispatch to nothing. `resolutionArgType` passes the dispatch class to `resolveFunction`, and tier 7 works on it.

Tier 7 first collects the classdef classes among the arguments that define the method and picks the dominant one (the first argument's class, unless a later class lists it in `InferiorClasses`). Only when no classdef argument has the method does it look in the TypeScript class method table, taking the first argument whose class registered the method; the target is `{ kind: "builtinMethod", className, methodName }`, which `interpretTarget` runs through `invokeClassMethod`. So a classdef object dominates a TypeScript-implemented one whatever the argument order. In MATLAB, classes such as `datetime` are classdef classes, and between two classes with no `InferiorClasses` relationship the leftmost wins; numbl differs only when both operands define the method and the TypeScript object comes first.

Operators use the same resolution. `BINARY_OPERATOR_METHODS` and `UNARY_OPERATOR_METHODS` (`runtime/runtimeOperators.ts`) map each operator to its method (`+` to `plus`, `~` to `not`, `'` to `ctranspose`, ...). When an operand has a dispatch class, the `Runtime` operator (`binop`, `uplus`, `uminus`, `not`, `transpose`, `ctranspose`) calls `rt.dispatch(method, ...)`, which resolves the name with the operands as arguments; with no method the name reaches the builtin of that name, which runs the built-in operation. The built-in operations reject operands they do not take with MATLAB's error (`MATLAB:math:mustBeNumericCharOrLogical`, "Invalid data type. Argument must be numeric, char, or logical."): cells, structs, function handles, strings for `-`, `~` and `&`, and objects whose class does not define the operator's method. Transposing a cell, struct or object rearranges it; transposing a function handle raises `MATLAB:transpose:unsupportedType`.

### Registering methods of a TypeScript-implemented class

A class implemented in TypeScript has no classdef, so its methods come from a table in `runtime/builtinRegistry.ts`, keyed by class name. A builtin module defines each method with `classMethod(className, spec)` (`interpreter/builtins/builtin.ts`), where `spec` is a builtin spec whose `name` is the method name:

```ts
classMethod("duration", {
  name: "plus",
  nargin: 2,
  apply: args => durationPlus(args[0], args[1]),
});
```

The method receives the objects as ordinary arguments (either operand may be the object), and `a + b`, `plus(a, b)` and `a.plus(b)` all reach it. `methods` and `ismethod` list the registered methods. Like a builtin, a method gets its other arguments under its `accepts` policy (a sparse argument arrives dense unless it declares `accepts: { sparse: true }`; see [builtins.md](../builtins.md#argument-kinds-sparse-and-other-non-dense-values)); since resolution picks the method first, the method's policy is the one that applies. `datetime`, `duration` (`plus`, `minus` and the comparisons) and `decomposition` (`mldivide`, `mrdivide`) are registered this way. A new value type that should overload operators and functions:

1. stores its values as a kind whose descriptor has `isObject: true` and whose `className` is the class name (a `class_instance` with that class name, as `datetime` does, or a new kind; see [values-and-tensors.md](../runtime/values-and-tensors.md#adding-a-value-kind));
2. registers its methods with `classMethod`, including the operator methods it supports; operators it does not define raise MATLAB's error.

The JIT does not compile calls that resolve to such a method (it declines the unit, which runs on the interpreter).

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

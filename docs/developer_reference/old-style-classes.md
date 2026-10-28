# Old-style (pre-`classdef`) classes

numbl supports MATLAB's pre-`classdef` class system ("class folders") so that packages such as GeoPDEs (Octave/old-MATLAB origin, six `@ClassName` folders and no `classdef` files) run. The behaviors below were derived by probing MATLAB R2025b and Octave 10, which agree on all of them; regression tests live in `numbl_test_scripts/classes/oldstyle_{polynom,subsref,objarray}/`, `at_folder_no_classdef/` and `old_style_superiorto/`.

Old-style classes run only on the interpreter: they have no `classdef` AST, so the JIT skips them (`jit/workspace/workspace.ts` ignores `ClassInfo.isOldStyle`).

## What an old-style class is

A class is a folder `@ClassName` on the path with no `classdef` file:

- `@ClassName/ClassName.m` is the constructor, an ordinary function that builds a struct and returns `obj = class(s, 'ClassName')`.
- Every other `.m` file in the folder is a method, an ordinary function taking an instance as one of its arguments.
- `@ClassName/private/` holds helpers callable only from the class's methods.

There is no property or method declaration block: the field set is whatever the constructor puts in the struct, and the method set is the folder's files. Old-style objects are value objects (`RuntimeClassInstance` with `isHandleClass = false`) with the usual copy-on-write behavior.

## Implementation

- **`class(s, 'Name')`**: the two-argument form of the `class` builtin (`interpreter/builtins/introspection.ts`) builds a value-class instance from the struct's fields. The inheritance form `class(s, 'Name', parent, ...)` is not supported and errors. `superiorto`/`inferiorto` are accepted as no-ops.
- **Registration**: `workspace/codeIndex.ts` registers a class folder without a `classdef` file through `makeOldStyleClassInfo` (`workspace/classInfo.ts`), a `ClassInfo` with `isOldStyle: true` whose constructor is `@Name/Name.m` and whose methods are the other files. The constructor is not exposed as a free method (`workspace/functionIndex.ts`).
- **Construction**: `instantiateClass` (`interpreter/oop.ts`) calls the constructor function directly with the user's arguments (`oldStyleConstructorBodyFor` in `resolution/functionBody.ts`) rather than pre-creating an instance as for `classdef`.
- **Dispatch, `subsref`/`subsasgn` overloads, the in-method bypass and operator overloading** reuse the `classdef` machinery: `resolution/classMethod.ts`, `runtime/runtimeIndexing.ts` and `runtime/runtimeMemberAccess.ts` (which build the `S` struct array), and `bypassesOverload` in `interpreter/indexing.ts`.
- **Object arrays** are `RuntimeClassInstanceArray` (`RTV.classInstanceArray`); indexed assignment grows them (`runtime/runtimeIndexing.ts`).

## MATLAB semantics

For `p = polynom([1 2 3])` built with `class(struct('c', [1 2 3]), 'polynom')`:

| query                   | result                                                         |
| ----------------------- | -------------------------------------------------------------- |
| `class(p)`              | `'polynom'`                                                    |
| `isa(p,'polynom')`      | `true` (`isa(p,'double')` and `isa(p,'numeric')` are `false`)  |
| `isobject(p)`           | `true`                                                         |
| `isstruct(p)`           | `false`                                                        |
| `numel(p)`, `isscalar`  | `1`, `true`                                                    |
| `fieldnames(p)`         | `{'c'}` (works outside methods)                                |
| `struct(p)`             | scalar struct with field `c` (works outside methods)           |
| `properties('polynom')` | errors (`No class 'polynom'`); `properties` is `classdef`-only |

**Field access.** Inside a method of the class, `obj.field` reads and writes the underlying struct directly and does not call an overloaded `subsref`/`subsasgn`, so a delegating overload cannot recurse. Outside the methods, `obj.field` errors ("Access to an object's fields is only permitted within its methods.") unless the class overloads `subsref` (read) or `subsasgn` (write), in which case `obj.field`, `obj(i)` and `obj{i}` dispatch to them with `S` a struct array of `{type, subs}` entries, one per chained operation (`type` is `'.'`, `'()'` or `'{}'`; `subs` is a field name or a cell of indices). GeoPDEs ships exactly such delegating overloads in every class:

```matlab
function value = subsref(obj, S)
  value = builtin('subsref', obj, S);
end
function obj = subsasgn(obj, S, value)
  obj = builtin('subsasgn', obj, S, value);
end
```

`builtin('subsref', ...)` and `builtin('subsasgn', ...)` apply the default indexing and bypass the class's overloads.

**Method dispatch.** Methods are called in function-call form, `method(obj, args...)`; `obj.method(...)` is field access, the opposite of `classdef`. Dispatch selects the folder of the leftmost argument that is a class instance (`combine(3, a)` runs `@aa/combine.m`), and arguments bind positionally as written. `@folder` methods are not globally callable: `helper(3)` with `helper` only in `@aa/` errors. `feval('method', obj, ...)` dispatches the same way. Operators dispatch to methods with reserved names (`plus`, `mtimes`, `eq`, `uminus`, ...), `disp(p)` to `@class/disp.m`, and bare `p` to `@class/display.m`.

**Field-set invariant.** All instances of a class must have the same field names; MATLAB errors ("Cannot change the number of fields of class 'X' without first typing 'clear classes'"). Field values and their types may differ.

**Object arrays.** `[a a a]` is a 1×3 object array. `arr(2) = obj` with `arr` undefined, and `s.boundary(i) = obj` with `s.boundary` undefined, grow an object array; `arr(k).field = v` is a chained assignment. In MATLAB, growth from an explicit `[]` base fails (`x = []; x(i) = obj` and `subsasgn([], S, obj)` give "value of type 'X' is not convertible to 'double'").

## Deliberate divergences

These make numbl more permissive than MATLAB and only admit more programs:

- Field access outside methods is permitted when the class does not overload `subsref`: `getMember` returns the stored field rather than erroring. GeoPDEs makes fields public through its overloads anyway.
- The field-set invariant is not enforced.
- `arr(i) = obj` builds an object array even when `arr` is `[]` or the class defines a delegating `subsasgn`.

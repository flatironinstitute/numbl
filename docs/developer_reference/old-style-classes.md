# Spec: Old-style (pre-`classdef`) MATLAB classes

Status: **implemented** (interpreter path). This document specifies the
pre-`classdef` MATLAB OOP system ("class folders") that numbl supports so that
packages like **GeoPDEs** (Octave/old-MATLAB origin) run. It is the result of
empirically probing MATLAB R2025b and Octave 10 with small test classes; the
observed behaviors below are what the implementation reproduces.

Old-style classes run **exclusively on the interpreter** — they have no
`classdef` AST, so the JIT (`--opt 1`/`--opt 2`) skips them and the interpreter
handles construction, dispatch, and indexing.

Implemented across: the 2-arg `class(struct,'name')` constructor + `isobject`
(`interpreter/builtins/introspection.ts`); old-style `@folder` registration
(`lowering/classInfo.ts` `makeOldStyleClassInfo`, `lowering/loweringContext.ts`);
the construction path (`interpreter/interpreterFunctions.ts` `instantiateClass`);
object-array growth (`runtime/runtimeIndexing.ts` `indexStore`,
`runtime/constructors.ts` `RTV.classInstanceArray`); JIT skip
(`jit/workspace/workspace.ts`). Method dispatch, `subsref`/`subsasgn` overload
dispatch, the in-method bypass, and operator overloading reuse the existing
classdef machinery unchanged. Regression tests live in
`numbl_test_scripts/classes/oldstyle_{polynom,subsref,objarray}/` (verified in
both numbl and `matlab -batch`).

### Deliberate divergences from MATLAB (permissive, not bugs)

- **Field access outside methods is permitted.** numbl's `getMember` returns
  the stored field directly rather than erroring "Access to an object's fields
  is only permitted within its methods." GeoPDEs overloads `subsref` to make
  fields public anyway, so observable behavior matches; numbl is just more
  permissive (§4).
- **The field-name invariant (§3) is not enforced.** Instances of one class may
  carry different field sets without error.
- **Object-array growth from an empty/undefined slot is permissive (§7).**
  `arr(i) = obj` builds an object array even when the class defines a
  (delegating) `subsasgn` — modern MATLAB rejects the bare-variable form in that
  case (it dispatches to `subsasgn([],…)`), but the struct-field form GeoPDEs
  actually uses works identically in both. Accepting the bare form only admits
  more programs.

## 1. Motivation

GeoPDEs is built entirely on the old-style class system: 6 `@ClassName` folders
(`@msh_cartesian`, `@sp_scalar`, `@sp_vector`, `@msh_multipatch`,
`@sp_multipatch`, `@sp_multipatch_C1`), **zero `classdef` files**, and `obj =
class(struct, 'name')` constructors. numbl today only registers `@ClassName`
folders that contain a `classdef` file and explicitly _skips_ folder-classes
without one (`loweringContext.ts`, the `if (!group.classDefFile) continue;`
branch). The first wall is:

```
Builtin 'class' does not support these argument types: (struct, char)
  at @msh_cartesian/msh_cartesian.m:168 →  msh = class(msh, 'msh_cartesian');
```

## 2. What an old-style class _is_

A class is a **folder** named `@ClassName` on the path. Inside:

- `@ClassName/ClassName.m` — the **constructor**, an _ordinary function_ (not a
  `classdef`). It builds a plain struct of the object's data and calls
  `obj = class(structData, 'ClassName')` to tag it.
- `@ClassName/<method>.m` — **methods**, ordinary functions that take an object
  of the class as one of their arguments (conventionally the first).
- `@ClassName/private/` — private helper functions, callable only from the
  class's own methods (numbl already has the `privateFilesByDir` mechanism).

There is **no property/method declaration block**; the field set is whatever the
constructor puts in the struct, and the method set is whatever `.m` files are in
the folder.

## 3. The `class(s, 'name', ...)` constructor builtin

`class(structValue, 'ClassName')` produces a class instance whose data is
`structValue` and whose class is `'ClassName'`. (There is an optional trailing
form `class(s, 'Name', parent1, parent2, ...)` for old-style inheritance — **not
needed for GeoPDEs**, can be deferred / error cleanly.)

Empirically (MATLAB R2025b), for `p = polynom([1 2 3])` built via
`class(struct('c',[1 2 3]), 'polynom')`:

| query                   | result                                                            |
| ----------------------- | ----------------------------------------------------------------- |
| `class(p)`              | `'polynom'`                                                       |
| `isa(p,'polynom')`      | `true`                                                            |
| `isa(p,'double')`       | `false`                                                           |
| `isobject(p)`           | `true`                                                            |
| `isstruct(p)`           | `false`                                                           |
| `numel(p)`              | `1`                                                               |
| `isscalar(p)`           | `true`                                                            |
| `fieldnames(p)`         | `{'c'}` (works outside methods)                                   |
| `struct(p)`             | scalar struct with field `c` (works outside methods)              |
| `properties('polynom')` | **errors** (`No class 'polynom'`) — `properties` is classdef-only |

### Field-set invariant

All instances of a given class **must have identical field names**. Violating
this is an error in both MATLAB and Octave:

```
Cannot change the number of fields of class 'X' without first typing 'clear classes'.   % MATLAB
error: mismatch in number of fields                                                      % Octave
```

The _values_ (and even runtime types) of the fields may differ between
instances; only the set of field names is fixed. (GeoPDEs relies on this: a
`msh_cartesian`'s `boundary` field is sometimes a `msh_cartesian` object array
and sometimes `[]`.) An implementation should record the field-name set the
first time a class is constructed and validate subsequent constructions.

## 4. Field access: forbidden outside methods unless `subsref` is overloaded

This is the central rule.

- **Inside a method** of the class, `obj.field` reads/writes the underlying
  struct data directly (default access). This does **not** invoke an overloaded
  `subsref`/`subsasgn` — so a delegating overload (see §6) cannot cause infinite
  recursion.
- **Outside the class's methods**, `obj.field` is an **error** by default:

  ```
  Access to an object's fields is only permitted within its methods.
  ```

  unless the class overloads `subsref` (read) / `subsasgn` (write), in which
  case `obj.field`, `obj(i)`, `obj{i}` dispatch to those methods.

GeoPDEs makes every class's data publicly readable/writable precisely by
providing the trivial delegating overloads in §6.

## 5. Method dispatch — function-call form only

- Methods are dispatched by the **function-call form** `method(obj, args...)`,
  **not** dot form. `obj.method(...)` is parsed as _field access_ `obj.method`
  (then an index), so for old-style classes it errors / hits `subsref` — it does
  **not** call the method. (This is the opposite of `classdef`, where
  `obj.method()` works.)
- Dispatch selects the method folder of the **leftmost argument that is a class
  instance**, regardless of its position:
  - `combine(a, b)` with `a::aa`, `b::bb` → `@aa/combine.m`
  - `combine(b, a)` → `@bb/combine.m`
  - `combine(3, a)` → `@aa/combine.m` (a double, then the object)
  - Arguments bind positionally as written; dispatch only chooses _which_ method
    file runs, it does not reorder args. (So in `combine(3,a)`, inside
    `@aa/combine`, the first parameter is `3`.)
- `@folder` methods are **not** globally callable — they exist only via dispatch
  on an instance of their class. Calling `helper(3)` where `helper` lives only
  in `@aa/` errors with "Incorrect number or types of inputs".
- Within a method, sibling methods/private functions of the same class are
  directly callable (still by function-call dispatch on an instance).
- `feval('method', obj, ...)` dispatches the same way.
- Operators are methods with reserved names: `plus` (`+`), `minus` (`-`),
  `mtimes` (`*`), `times` (`.*`), `mrdivide`, `eq`, `lt`, `uminus`, `ctranspose`,
  etc. `5 + p` dispatches to the object operand's `plus`. `disp(p)` →
  `@class/disp.m`; bare `p` (no semicolon) → `@class/display.m`, with
  `inputname(1)` giving the variable name. (GeoPDEs defines **no** `disp`/
  `display`/operator overloads — only `subsref`/`subsasgn` — so a correct default
  display is sufficient; operator overloading support is general goodness but not
  required for GeoPDEs.)

## 6. `subsref` / `subsasgn` and `builtin(...)`

GeoPDEs' every class ships exactly these two files, whose only purpose (per their
own comments) is to "override private read/write access of object data" — i.e.
expose the fields publicly:

```matlab
% @msh_cartesian/subsref.m
function value = subsref (obj, S)
  value = builtin ('subsref', obj, S);
end
% @msh_cartesian/subsasgn.m
function obj = subsasgn (obj, S, value)
  obj = builtin('subsasgn', obj, S, value);
end
```

So the implementation must provide:

1. **Dispatch of indexing to overloaded `subsref`/`subsasgn`** when the base
   value is an old-style object and the class defines them:
   - `obj.field`, `obj(idx)`, `obj{idx}` (read) → `subsref(obj, S)`
   - `obj.field = v`, `obj(idx) = v` (write) → `obj = subsasgn(obj, S, v)`
   - `S` is a **struct array** of substructs, one per chained operation, each
     `{type, subs}`:
     - `type`: `'.'` | `'()'` | `'{}'`
     - `subs`: for `'.'` a char field name; for `'()'`/`'{}'` a cell array of the
       index args (`{2}`, `{':'}`, `{1,3}`, …).
     - Chained access `obj.a(2).b` produces a 3-element `S`.

   numbl already constructs `substruct`-style `S` arrays elsewhere
   (`runtimeDispatch.ts` builds `RTV.structArray(["type","subs"], …)` for
   indexing) — reuse that shape.

2. **`builtin('subsref', obj, S)` / `builtin('subsasgn', obj, S, v)` defaults**
   for old-style objects — the "private" default behavior that the overloads
   delegate to:
   - `'.'` → read/write the named field of the object's underlying struct data.
     Reading a missing field errors; writing a new field is **disallowed** (field
     set is fixed, §3).
   - `'()'` → index into an **object array** (see §7): select elements / assign
     elements.
   - `'{}'` → error for these classes (not used).
   - Chained `S` is applied left to right.
   - `builtin(name, ...)` must bypass the class's own overloads (call the real
     builtin), which numbl's `callBuiltin` path already expresses — extend it to
     accept `'subsref'`/`'subsasgn'` operating on class instances.

3. Recall (§4) the recursion guard: default field access _inside_ methods must
   **not** route back through the overloaded `subsref`/`subsasgn`.

## 7. Object arrays

GeoPDEs constructs arrays of objects, e.g. in `@msh_cartesian/msh_cartesian.m`:

```matlab
for iside = 1:2*msh.ndim
  msh.boundary(iside) = msh_cartesian (...);   % grow object array
  msh.boundary(iside).rdim = msh.rdim;          % chained assign into element
end
```

numbl already has `RuntimeClassInstanceArray` (`elements` + `shape`) and default
`horzcat`/`vertcat` for instances. Required behaviors (verified in MATLAB _and_
Octave):

- `[a a a]` → a `1×3` object array, `class` = the element class, `numel`/
  `length`/`size` as usual.
- **Growth from an _undefined_ variable/field works**: `arr(2) = obj` when `arr`
  is not yet defined creates/extends an object array. Likewise
  `s.boundary(iside) = obj` when `s.boundary` is undefined.
- **Indexing** an object array: `arr(k)` → element (a scalar instance);
  `arr(k).field` → chained subsref; `arr(k).field = v` and `arr(k) = obj2` →
  chained subsasgn.

### Important divergence: growth from explicit `[]`

There is a real difference between _undefined_ and _explicitly `[]`_ targets, and
between the **assignment statement** and the **`subsasgn` function**:

- `arr(i) = obj` as a **statement**, with `arr` _undefined_, grows an object
  array — **this is the case GeoPDEs needs and it works in MATLAB and Octave.**
- Calling `subsasgn([], S, obj)` / `builtin('subsasgn', [], S, obj)` _as a
  function_ with a `double []` base **fails** in modern MATLAB
  (`value of type 'X' is not convertible to 'double'`). So does
  `x = []; x(i) = obj`.

Implication for numbl: the statement-level lowering of `lhs(i) = rhs` must treat
an **undefined / never-assigned** LHS base as an empty array _of the RHS's class_
when the RHS is an old-style object, then append — rather than first
materializing a numeric `[]`. This is what makes the GeoPDEs constructor's
boundary loop work. (numbl can choose to be _more_ permissive than R2025b here —
treating `[]` itself as growable into an object array — since that only accepts
more programs; matching the "undefined grows / `[]` stays double" distinction
exactly is not required for GeoPDEs but documents MATLAB's real rule.)

## 8. Value semantics

Old-style objects are **value types** (no handle classes here). `q = scale(p,
10)` returns a modified copy; `p` is unchanged. This maps onto numbl's existing
value-class path (`RuntimeClassInstance` with `isHandleClass = false`), including
copy-on-write/refcount behavior already in place for value class instances.

## 9. Introspection / builtins that must recognize old-style instances

- `class(obj)` → class name (1-arg form already implemented; only the 2-arg
  _constructor_ form is missing).
- `isa(obj, 'name')`, `isa(obj, 'numeric'|'float'|...)` → name match only; old
  instances are not numeric.
- `isobject(obj)` → `true`.
- `isstruct(obj)` → `false`.
- `fieldnames(obj)` / `struct(obj)` → underlying field names / a struct copy
  (allowed outside methods even though direct `.field` is not).
- `methods('name')` / `methods(obj)` → method file names (including the
  constructor).
- `exist('name')` → nonzero (class folder on path).
- `isfield`, `numel`, `size`, `length`, `ndims`, `isscalar`, `isempty` on
  instances and instance arrays.

## 10. numbl integration points

Concrete touch points for the implementer (paths relative to repo root):

- **`class` 2-arg constructor** — `src/numbl-core/interpreter/builtins/introspection.ts`
  (`class` builtin currently 1-arg only). Build a `RuntimeClassInstance` from the
  struct's fields, `isHandleClass=false`. Validate/record the field-name set.
- **Class-folder registration without `classdef`** —
  `src/numbl-core/lowering/loweringContext.ts`, the `classFolderGroups` loop. The
  `if (!group.classDefFile) continue;` branch must instead register an _old-style_
  class: synthesize a `ClassInfo` whose constructor is `@Name/Name.m`, whose
  `methodNames`/`externalMethodFiles` are the other `.m` files, with a flag (e.g.
  `isOldStyle: true`) and no property declarations. `ClassInfo` is defined in
  `src/numbl-core/lowering/classInfo.ts`.
- **Function-call method dispatch on the leftmost object arg** — the resolution
  path used by `dispatch()` in `src/numbl-core/runtime/runtimeDispatch.ts` and
  `resolveClassMethod` in `src/numbl-core/interpreter/interpreter.ts` /
  `functionResolve.ts`. Old-style classes dispatch the same way classdef external
  methods already do; the new part is choosing the method by _argument class_
  for the function-call form and **not** treating `obj.method(...)` as a call.
- **`subsref`/`subsasgn` overload dispatch + `builtin` defaults** — the
  member-access and indexed-assignment evaluation in the interpreter
  (`interpreterExec.ts` / `struct-access.ts` / `runtimeIndexing.ts`). Add: if the
  base is an old-style `RuntimeClassInstance(Array)` and the class defines
  `subsref`/`subsasgn`, build the `S` struct array and dispatch; otherwise apply
  the default (field access only inside methods; `()` on arrays). Provide
  `builtin('subsref'/'subsasgn', obj, S[, v])` via the `callBuiltin` path.
- **Object-array growth from undefined LHS** — indexed-assignment lowering must
  seed an empty object array of the RHS class (§7).
- **Recursion guard** — track "currently executing inside a method of class C"
  (numbl already threads `methodScope`/`className` through `withFileContext`) so
  default `.field` access inside methods bypasses the overload.

## 11. Suggested implementation phases

1. `class(s,'name')` constructor + field-set invariant + introspection
   (`class`/`isa`/`isobject`/`isstruct`/`fieldnames`/`struct`/`methods`).
2. Old-style `@folder` registration (constructor + methods, no `classdef`).
3. Function-call method dispatch (leftmost-object rule); confirm GeoPDEs methods
   like `msh_precompute(msh, …)` resolve.
4. `subsref`/`subsasgn` overload dispatch + `builtin` defaults + the in-method
   recursion guard. This unblocks GeoPDEs' field access.
5. Object arrays + undefined-LHS growth.
6. (Optional, not needed by GeoPDEs) operator/`disp`/`display` overloads,
   old-style multiple inheritance via `class(s,'name',parent,…)`.

After each phase, re-run `mip test geopdes` and the existing class test scripts
(`numbl_test_scripts/classes/`, `abstract_classes/`) to guard against
regressions in the `classdef` path, which shares much of this machinery.

## 12. Reproduction corpus

The probes used to derive this spec are simple to recreate: a `@polynom`
value class (constructor + `coeffs`/`double`/`plus`/`mtimes`/`display`/`disp`/
`subsref`/`subsasgn`) exercises §3–§6; a recursive `@msh`-like class whose
constructor grows a `boundary` object array from an undefined field (and whose
base case sets `boundary = []`) reproduces §7 and matches GeoPDEs'
`@msh_cartesian` constructor. When validating an implementation, mirror these
against both `matlab -batch` and `octave --eval` — GeoPDEs targets Octave
semantics, and the two agree on every behavior in this spec.

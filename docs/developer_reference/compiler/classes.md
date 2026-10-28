# Classes

This page covers `classdef` classes: the class model built at registration, how member access is enforced, Constant properties, and the meta.class objects behind `?Name`. Old-style `@folder` classes are described in [old-style-classes.md](../old-style-classes.md).

## The class model

When the `CodeIndex` registers a classdef (a workspace file or a class in the main file), `extractClassInfo` (`workspace/classInfo.ts`) reads the parsed `ClassDef` once and produces a `ClassInfo`. Besides the names used for resolution (`methodNames`, `staticMethodNames`, `constructorName`, `enumMembers`), it holds:

- `properties`: one `PropertyInfo` per declared property, in declaration order, with `getAccess`, `setAccess`, `constant`, `dependent`, `hidden`, `abstract`, `transient` (parsed only), the default expression, and `definingClass`.
- `methods`: one `MethodInfo` per declared method (bodies and signatures, including the constructor, static and abstract methods; not the `get.X`/`set.X` accessors), with `access`, `static`, `hidden`, `abstract`, `sealed`, input and output names, and `definingClass`.
- class attributes `abstract`, `sealed`, `hidden`.
- `propertyNames` / `propertyDefaults`: the properties stored on each instance, which excludes Dependent, Constant and Abstract properties.

An access value (`MemberAccess`) is `public`, `protected`, `private`, `immutable` (SetAccess), `none` (SetAccess, e.g. of a Constant property), or a class list `{ classes }` from `Access = {?A, ?B}` or `Access = ?A`. In a block, `Access` sets both get and set access and a later `GetAccess`/`SetAccess` overrides one of them. The parser keeps attribute values as text; it understands `~Flag` (false), `Name = ?pkg.Class` and quoted values.

`ClassTable` (`workspace/classTable.ts`) answers member queries over the inheritance chain. `findProperty` and `findMethod` return the declaration an instance of a class sees (its own, else the first ancestor's in method-resolution order), cached per class and name. `canAccess(access, definingClass, contextClass)` is the access rule: private admits the defining class; protected also its subclasses; a class list also the listed classes and their subclasses (the defining class always keeps access). `publicProperties`, `publicMethodNames`, `allProperties` and `unimplementedAbstractMembers` back introspection and instantiation checks.

## Access enforcement

The class of the running code is the running `CallFrame`'s class (see [interpreter.md](interpreter.md)). Methods, the local functions of a classdef file, and anonymous functions created in them run with their class, as in MATLAB. The runtime reads it through the `rt.codeSite` hook and the table through `rt.classTable`, both installed by the interpreter. Objects without a classdef (`MException`, `datetime`, `graph` and other built-in objects) have no `ClassInfo`, so these checks do not apply to them; MException's read-only properties are enforced by its own runtime code.

The checks are in `runtime/classAccess.ts`, which returns MATLAB's error (identifier and message) or null:

- Reading a property (`getMember`, and `obj.prop(args)` in `methodDispatch`): `MATLAB:class:GetProhibited`. Code of a subclass sees a superclass's private member as missing (`MATLAB:noSuchMethodOrField`).
- Setting a property (`setMemberReturn`): `MATLAB:class:SetProhibited`, with "is not supported" when the code cannot read the property either and "because it is read-only" otherwise. `immutable` admits only the constructor of the defining class; a Constant property admits no one. Assigning a Dependent property without a set method raises `MATLAB:class:noSetMethod`, and an undeclared property of a classdef object (not a `dynamicprops`, without an overloaded subsasgn) raises `MATLAB:noPublicFieldForClass`.
- Calling a method (`interpretClassMethod`, which every class-method call goes through, and the constructor in `instantiateClass`): `MATLAB:class:MethodRestricted`.

When a property access is denied and the class overloads `subsref`/`subsasgn`, the access goes to the overload instead, which runs in the class's context.

`instantiateClass` also refuses an abstract class (`MATLAB:class:abstract`, declared or with unimplemented inherited abstract members) and, once per class, a class that subclasses a Sealed class or overrides a Sealed method.

The JIT does not check access, so its class support (`jit/lowering/classDefs.ts`) declines classes with non-public property access; methods blocks with attributes other than `Static` were already declined. Such classes run on the interpreter.

## Property defaults and Constant properties

Property defaults are evaluated in the context of the class that declares them (`classContextFrame` in `oop.ts`: the class's file and class, an empty workspace), so a default may call the class's private static methods.

A Constant property is not stored on instances. Its default is evaluated on first use by `constantPropertyValue` and cached per class on the interpreter (`constantValues`; a circular definition is an error). It is read as:

- `ClassName.Prop`: `resolveQualified` returns a `classConstant` resolution (see [function-resolution.md](function-resolution.md)), so the constructor is not called; `ClassName.Prop(i)` indexes the value.
- `obj.Prop`, including inside methods and on subclass instances: `getMember` and `methodDispatch` check the property's declaration first.

MATLAB evaluates all Constant defaults when the class is first used; numbl evaluates each one when it is first read, which only differs when a default has side effects.

## Introspection and display

`properties`, `fieldnames` and object display list the public, non-Hidden properties (Constant and Dependent ones included, own before inherited); `isprop` is true for every property whatever its access; `ismethod` and `methods` list public, non-Hidden methods (with the constructor, static methods and, for handle classes, the methods of `handle`).

## meta.class

`?Name` and `metaclass(obj)` (`interpreter/metaClass.ts`) return a handle object of class `matlab.metadata.Class` (MATLAB's current name; `isa(mc, 'meta.class')` holds). One object is built per class and run, so `DefiningClass` fields refer back to it. It has `Name`, `Description`, `DetailedDescription`, `Hidden`, `Sealed`, `Abstract`, `Enumeration`, `ConstructOnLoad`, `HandleCompatible`, `InferiorClasses`, `RestrictsSubclassing`, `PropertyList`, `MethodList`, `EnumerationMemberList` and `SuperclassList`. The lists are column arrays of `matlab.metadata.Property` (with access, flags, `HasDefault`/`DefaultValue`, `DefiningClass`), `matlab.metadata.Method` (with `Access`, flags, `InputNames`, `OutputNames`, `DefiningClass`) and `matlab.metadata.EnumeratedValue` objects. As in MATLAB, a subclass's lists omit the private members of its superclasses. Property access on an object array gives a comma-separated list, so `{mc.PropertyList.Name}` works. Built-in classes (`double`, `handle`, ...) get an object with empty lists (`MException` lists its public properties), and an unknown name gives an empty 0x0 Class array.

Not modelled: events and listeners, `findobj` on meta objects, `meta.class.fromName`, `GetMethod`/`SetMethod` handles, `Validation`, `Namespace`/`ContainingPackage`, and the exact order of `MethodList`.

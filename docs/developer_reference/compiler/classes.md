# Classes

This page covers `classdef` classes: the class model built at registration, how member access is enforced, Constant properties, events and listeners, and the meta.class objects behind `?Name`. Old-style `@folder` classes are described in [old-style-classes.md](../old-style-classes.md).

## The class model

When the `CodeIndex` registers a classdef (a workspace file or a class in the main file), `extractClassInfo` (`workspace/classInfo.ts`) reads the parsed `ClassDef` once and produces a `ClassInfo`. Besides the names used for resolution (`methodNames`, `staticMethodNames`, `constructorName`, `enumMembers`), it holds:

- `properties`: one `PropertyInfo` per declared property, in declaration order, with `getAccess`, `setAccess`, `constant`, `dependent`, `hidden`, `abstract`, `transient` (parsed only), `setObservable`, `getObservable`, the default expression, and `definingClass`.
- `methods`: one `MethodInfo` per declared method (bodies and signatures, including the constructor, static and abstract methods; not the `get.X`/`set.X` accessors), with `access`, `static`, `hidden`, `abstract`, `sealed`, input and output names, and `definingClass`.
- `events`: one `EventInfo` per event of an `events` block, with `listenAccess`, `notifyAccess` (both default `public`), `hidden` and `definingClass`.
- class attributes `abstract`, `sealed`, `hidden`.
- `propertyNames` / `propertyDefaults`: the properties stored on each instance, which excludes Dependent, Constant and Abstract properties.

An access value (`MemberAccess`) is `public`, `protected`, `private`, `immutable` (SetAccess), `none` (SetAccess, e.g. of a Constant property), or a class list `{ classes }` from `Access = {?A, ?B}` or `Access = ?A`. In a block, `Access` sets both get and set access and a later `GetAccess`/`SetAccess` overrides one of them. The parser keeps attribute values as text; it understands `~Flag` (false), `Name = ?pkg.Class` and quoted values.

`ClassTable` (`workspace/classTable.ts`) answers member queries over the inheritance chain. `findProperty` and `findMethod` return the declaration an instance of a class sees (its own, else the first ancestor's in method-resolution order), cached per class and name. `canAccess(access, definingClass, contextClass)` is the access rule: private admits the defining class; protected also its subclasses; a class list also the listed classes and their subclasses (the defining class always keeps access). `publicProperties`, `publicMethodNames`, `allProperties` and `unimplementedAbstractMembers` back introspection and instantiation checks. `allEvents` lists a class's events (own, then inherited, then `ObjectBeingDestroyed`, which every handle class inherits from `handle`) and `findEvent` looks one up.

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

## Events and listeners

`interpreter/events.ts` implements events of handle classes, with the builtins in `builtins/events.ts`. The objects are instances of four stdlib classes (`src/stdlib/shims/+event/`): `event.EventData` (`Source`, `EventName`, private set access), its sealed subclass `event.PropertyEvent` (adds `AffectedObject`), `event.listener` (`Source`, `EventName`, `Callback`, `Enabled`, `Recursive`) and its subclass `event.proplistener` (adds `Object`). A user class can subclass `event.EventData` to pass data with an event.

- `addlistener(src, Name, cb)` and `addlistener(src, Prop, PropEvent, cb)` check the event exists and the running code may listen to it (`MATLAB:class:ListenRestricted`), or that the property is accessible and SetObservable/GetObservable, and return one listener per source object. `listener(...)` and `event.listener(...)` (whose constructor is numbl's, see `instantiateClass`) create the same objects.
- The `EventRegistry` on the interpreter maps each source object (weakly) to its registrations. A listener from `addlistener` is held by the registry, so it lives as long as its source. A listener from `listener` is not: its `_onDestroy` hook removes it when its refcount reaches zero, so it stops firing once no variable holds it, as in MATLAB.
- `notify(src, Name[, data])` checks `NotifyAccess` (`MATLAB:class:NotifyRestricted`), fills in the data's `Source` and `EventName` (a new `event.EventData` when none is given) and calls the enabled listeners with `(src, data)`. As in MATLAB, the most recently added listener runs first, listeners added during the notification are not called for it, deleted ones are skipped, and a listener whose callback is running is not called again for an event it triggers unless `Recursive` is true.
- An error in a callback is caught and printed as MATLAB's warning (`Warning: Error occurred while executing the listener callback for event E defined for class C:` and the message, identifier `MATLAB:callback:error`); the remaining listeners still run. The callback runs with `tryDepth` raised, as a `try` body does, so an error in a compiled loop or script inside it is re-raised by the interpreter (see [executors.md](../executors.md#runtime-errors-and-bails)).
- Property events: `setMemberReturn` triggers PreSet and PostSet around the assignment of a SetObservable property (including indexed assignments and assignments in methods, not inside its own set method), and `getMember` PreGet and PostGet around a read of a GetObservable property, through the runtime's `propertyEvent` hook. The callback gets the property's meta.property (`metaPropertyFor`, the same object as in `PropertyList`) and an `event.PropertyEvent`.
- Deleting a handle object (`delete(h)`, or the class's own `delete` method, which `interpretClassMethod` wraps) triggers `ObjectBeingDestroyed`, runs the delete method, marks the object `_deleted` and drops its registrations. Reading or setting a property of a deleted object raises `MATLAB:class:InvalidHandle`, `isvalid` is false, and display shows `handle to deleted C`. `==`/`~=` on handle objects compare identity.
- `events(obj)` / `events('Name')` lists the events any code may listen to that are not Hidden.

`obj.notify(...)`, `obj.addlistener(...)`, `obj.delete()` and the other methods of `handle` reach these builtins when the class does not define the method (`classDefinesMethod` in `interpreter.ts`).

Not modelled: MATLAB moves a listener whose `Enabled` is set back to true to the front of the order (numbl keeps its place); `lastwarn` does not record the callback warning, and the warning does not include the error's stack; reading a property through `obj.Prop(i)` does not trigger PreGet/PostGet; an object is not deleted (and its delete method not called) when its last reference goes away, only by an explicit `delete`; `AbortSet`; listener `Source` lists for `findobj`-style queries.

The JIT never compiles handle classes or calls to these builtins (they have no JIT definition), so code that uses events runs on the interpreter; `jit_parity/scripts/B04_listener-callbacks.m` checks that loops around `notify` and callbacks that call compiled functions agree across `--opt` levels.

## Introspection and display

`properties`, `fieldnames` and object display list the public, non-Hidden properties (Constant and Dependent ones included, own before inherited); `isprop` is true for every property whatever its access; `ismethod` and `methods` list public, non-Hidden methods (with the constructor, static methods and, for handle classes, the methods of `handle`). Displayed without an output, `methods` leaves out the methods a handle class inherits from `handle` and refers to `methods('handle')`, as MATLAB does.

## meta.class

`?Name` and `metaclass(obj)` (`interpreter/metaClass.ts`) return a handle object of class `matlab.metadata.Class` (MATLAB's current name; `isa(mc, 'meta.class')` holds). One object is built per class and run, so `DefiningClass` fields refer back to it. It has `Name`, `Description`, `DetailedDescription`, `Hidden`, `Sealed`, `Abstract`, `Enumeration`, `ConstructOnLoad`, `HandleCompatible`, `InferiorClasses`, `RestrictsSubclassing`, `PropertyList`, `MethodList`, `EventList`, `EnumerationMemberList` and `SuperclassList`. The lists are column arrays of `matlab.metadata.Property` (with access, flags including `SetObservable`/`GetObservable`, `HasDefault`/`DefaultValue`, `DefiningClass`), `matlab.metadata.Method` (with `Access`, flags, `InputNames`, `OutputNames`, `DefiningClass`), `matlab.metadata.Event` (with `Hidden`, `NotifyAccess`, `ListenAccess`, `DefiningClass`) and `matlab.metadata.EnumeratedValue` objects. As in MATLAB, a subclass's lists omit the private members of its superclasses. Property access on an object array gives a comma-separated list, so `{mc.PropertyList.Name}` works. Built-in classes (`double`, `handle`, ...) get an object with empty lists (`MException` lists its public properties), and an unknown name gives an empty 0x0 Class array.

Not modelled: `findobj` on meta objects, `meta.class.fromName`, `GetMethod`/`SetMethod` handles, `Validation`, `Namespace`/`ContainingPackage`, and the exact order of `MethodList`.

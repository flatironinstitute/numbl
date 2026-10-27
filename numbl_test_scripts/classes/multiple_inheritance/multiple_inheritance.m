% A class with two superclasses inherits methods and properties from both.
obj = MiChild_();
assert(obj.fromA() == 10);
assert(obj.fromB() == 200);
assert(fromB(obj) == 200);
assert(obj.total() == 6);
assert(isequal(sort(properties(obj))', {'a', 'b', 'c'}));

assert(isa(obj, 'MiBaseA_'));
assert(isa(obj, 'MiBaseB_'));
assert(isa(obj, 'matlab.mixin.Copyable'));
assert(isa(obj, 'handle'));
assert(ismethod(obj, 'fromB'));
assert(ismethod(obj, 'setB'));
s = superclasses(obj);
assert(any(strcmp(s, 'MiBaseA_')) && any(strcmp(s, 'MiBaseB_')) && any(strcmp(s, 'handle')));

% Handle semantics come through either parent.
obj2 = obj;
obj.setB(5);
assert(obj2.b == 5);

disp('SUCCESS')

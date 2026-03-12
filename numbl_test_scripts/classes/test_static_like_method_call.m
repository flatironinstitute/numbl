% Test ClassName.staticMethod(arg) where the static method is in an
% external file (@ClassName/method.m).
% Mirrors chebfun pattern: chebtech.techPref(pref) inside constructor,
% called via make() from a child class.

% Test 1: Direct call with no arguments
p0 = TechBase.techPref();
assert(p0.alpha == 10, 'default alpha should be 10');
assert(p0.beta == 20, 'default beta should be 20');

% Test 2: Direct call with struct argument
s = struct();
s.alpha = 99;
p1 = TechBase.techPref(s);
assert(p1.alpha == 99, 'merged alpha should be 99');
assert(p1.beta == 20, 'beta should remain 20');

% Test 3: Call from base class constructor directly
s2 = struct();
s2.alpha = 42;
obj1 = TechBase(1, 2, s2);
assert(obj1.data.alpha == 42, 'base ctor: alpha should be 42');
assert(obj1.data.beta == 20, 'base ctor: beta should be 20');

% Test 4: Call from child class constructor (mirrors chebfun)
s3 = struct();
s3.alpha = 77;
obj2 = TechChild(1, 2, s3);
assert(obj2.data.alpha == 77, 'child ctor: alpha should be 77');
assert(obj2.data.beta == 20, 'child ctor: beta should be 20');

% Test 5: Call via make (static method that calls constructor)
s4 = struct();
s4.alpha = 55;
obj3 = TechChild.make(1, 2, s4);
assert(obj3.data.alpha == 55, 'make: alpha should be 55');
assert(obj3.data.beta == 20, 'make: beta should be 20');

% Test 6: Function handle value should survive the merge
s5 = struct();
s5.alpha = @(x) x + 1;
p5 = TechBase.techPref(s5);
assert(isa(p5.alpha, 'function_handle'), 'function handle should be preserved');
assert(p5.alpha(5) == 6, 'function handle should work correctly');

% Test 7: Call static method via INSTANCE (not class name)
s6 = struct();
s6.alpha = 88;
obj4 = TechBase(1, 2, struct());
p6 = obj4.techPref(s6);
assert(p6.alpha == 88, 'instance static call: alpha should be 88');
assert(p6.beta == 20, 'instance static call: beta should be 20');

% Test 8: Call make via INSTANCE (not class name) — like chebfun's f.make(op,data,pref)
s7 = struct();
s7.alpha = 33;
obj5 = TechChild(1, 2, struct());
obj6 = obj5.make(1, 2, s7);
assert(obj6.data.alpha == 33, 'instance make: alpha should be 33');
assert(obj6.data.beta == 20, 'instance make: beta should be 20');

% Test 9: doCompose pattern — base method calls self.techPref(pref) then self.make(...)
% This mirrors chebtech/compose.m calling f.techPref(pref) then f.make(op,data,pref)
s8 = struct();
s8.alpha = 123;
obj7 = TechChild(1, 2, struct());
obj8 = obj7.doCompose(@sin, struct(), s8);
assert(obj8.data.alpha == 123, 'doCompose: alpha should be 123');
assert(obj8.data.beta == 20, 'doCompose: beta should be 20');

% Test 10: doCompose with function handle in pref (like refinementFunction)
s9 = struct();
s9.alpha = @(x) x * 2;
obj9 = TechChild(1, 2, struct());
obj10 = obj9.doCompose(@sin, struct(), s9);
assert(isa(obj10.data.alpha, 'function_handle'), 'doCompose fh: should be function_handle');
assert(obj10.data.alpha(5) == 10, 'doCompose fh: function should work');

disp('SUCCESS');

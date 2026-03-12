% Test calling an external static method declared as a prototype
% in a methods (Static = true) block.
% This tests that prototype declarations (signatures without function keyword)
% are properly recognized as static methods.

% Test 1: Call static method from outside the class
r1 = StaticMethodClass.addOne(5);
assert(r1 == 6, 'Static method call from outside failed');

% Test 2: Call static method from inside the class via an instance method
obj = StaticMethodClass(10);
r2 = obj.useStatic();
assert(r2 == 11, 'Static method call from inside class failed');

disp('SUCCESS');

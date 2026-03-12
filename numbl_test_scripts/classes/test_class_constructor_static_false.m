% Test that constructors in methods blocks with explicit Static=false
% are correctly called during class instantiation.
% This tests a bug where Static=false was incorrectly treated as Static=true,
% causing the constructor to be skipped.

obj = StaticFalseCtorTest_(42);
assert(obj.value == 42, 'Constructor was not called');
assert(obj.initialized == 1, 'Initialized flag not set');

% Also test operator overloading works (depends on constructor being called)
obj2 = StaticFalseCtorTest_(10);
result = obj + obj2;
assert(result.value == 52, 'Operator overloading failed');

disp('SUCCESS');

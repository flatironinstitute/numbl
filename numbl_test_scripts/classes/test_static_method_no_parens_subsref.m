% Test that ClassName.staticMethod without parentheses calls the static method,
% even when the class has a custom subsref and the constructor returns an empty instance.

% Test 1: Call with parentheses (should work)
r1 = StaticSubsref.getDefault();
assert(r1 == 42, 'Static method with parens failed');

% Test 2: Call without parentheses (should also call the static method)
r2 = StaticSubsref.getDefault;
assert(r2 == 42, 'Static method without parens failed');

disp('SUCCESS')

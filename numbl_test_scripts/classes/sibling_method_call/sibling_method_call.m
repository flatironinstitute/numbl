% Test: class method dispatch when instance is not the first argument
% In MATLAB, @folder methods are dispatched when ANY argument is a class
% instance, not just the first. This is the pattern used in chebfun where
% sampleTest(op, values, f, data, pref) dispatches to @chebtech/sampleTest.m
% because f is a chebtech instance (3rd argument).

obj = SiblingTest_(5);

% compute() calls helper_work(x, 10, obj) where obj is the 3rd argument
result = obj.compute(3);
assert(result == 35, 'Expected 35 (3*10 + 5)');

disp('SUCCESS')

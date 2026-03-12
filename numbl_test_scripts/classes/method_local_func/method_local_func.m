% Test: class method file with local helper functions
% A method file (@LocalHelper_/compute.m) calls a local function
% (applyOffset) defined in the same file. This should work just like
% local functions in regular .m files.

obj = LocalHelper_(10);
result = compute(obj, 5);
assert(result == 15, 'Expected 15');

obj2 = LocalHelper_(3);
result2 = obj2.compute(7);
assert(result2 == 10, 'Expected 10');

disp('SUCCESS')

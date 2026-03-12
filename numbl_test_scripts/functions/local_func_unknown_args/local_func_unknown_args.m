% Test that local functions in workspace files are found
% when called with unknown-typed arguments during JIT compilation.
% The workspace function is called from a class method to trigger JIT,
% with known-type args so the main function gets JIT-compiled,
% but the local function receives an unknown-typed argument.
obj = MyClass();
result = obj.run_it(3, [1 2 3]);
assert(isequal(result, [3 6 9]));
disp('SUCCESS');

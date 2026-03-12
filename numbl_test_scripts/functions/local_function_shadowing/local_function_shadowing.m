% Test that local functions in one file do not shadow primary functions
% from other files.
%
% compute(5) should return 50 (from compute.m, x*10), not 4995
% (from z_helper.m's local compute function, x*999).
%
% We extract the argument from a cell array so its type is Unknown at compile
% time. The presence of MyCompute (a class with a 'compute' method) then
% triggers the class-method-ambiguity path, producing empty lowering
% candidates and forcing runtime dispatch through $primaryFunctions.
% That is where the local-function shadowing bug would cause the wrong
% function to be called.

c = {5};
result = compute(c{1});
assert(result == 50);

disp('SUCCESS')

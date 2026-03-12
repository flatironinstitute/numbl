% Test that a handle to a local function defined in a classdef external
% method file can be passed to another method and called there.

obj = Processor(5);
result = obj.apply_op();
assert(result == 15, 'Expected my_local_func(5) == 15');

disp('SUCCESS');

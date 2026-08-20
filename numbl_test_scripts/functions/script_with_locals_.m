% A script whose first statement is not a function declaration, but which also
% defines a local function (MATLAB R2016b and later). Helper for
% script_with_local_functions.m; not a standalone test.
sl_y = 10;
sl_x = triple_(sl_y);

function out = triple_(in)
out = 3 * in;
end

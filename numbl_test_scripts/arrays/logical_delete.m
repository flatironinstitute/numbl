% Test element deletion with logical indexing: x(mask) = []
% In MATLAB, x(logical_mask) = [] removes elements where the mask is true.

% Basic logical deletion
x = [10 20 30 40 50];
mask = logical([0 0 1 0 1]);
x(mask) = [];
assert(isequal(x, [10 20 40]), 'logical delete should remove elements 3 and 5');

% Delete using isnan
y = [1 2 NaN 4 NaN];
y(isnan(y)) = [];
assert(isequal(y, [1 2 4]), 'isnan delete should remove NaN elements');

% Delete all true
z = [10 20 30];
z(logical([1 1 1])) = [];
assert(isempty(z), 'deleting all should give empty');

% Delete none
w = [10 20 30];
w(logical([0 0 0])) = [];
assert(isequal(w, [10 20 30]), 'deleting none should keep all');

% Delete single element with logical mask
v = [10 20 30 40];
v(logical([0 1 0 0])) = [];
assert(isequal(v, [10 30 40]), 'should remove second element');

disp('SUCCESS');

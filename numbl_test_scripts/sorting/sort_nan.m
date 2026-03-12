% Test that sort, unique, and union handle NaN correctly
% In MATLAB, NaN values sort to the end

% sort with NaN
x = [3, NaN, 1, 2];
y = sort(x);
assert(y(1) == 1, 'sort: first element should be 1');
assert(y(2) == 2, 'sort: second element should be 2');
assert(y(3) == 3, 'sort: third element should be 3');
assert(isnan(y(4)), 'sort: NaN should be at the end');

% unique with NaN
x = [3, NaN, 1, 2, 3];
y = unique(x);
assert(y(1) == 1, 'unique: first element should be 1');
assert(y(2) == 2, 'unique: second element should be 2');
assert(y(3) == 3, 'unique: third element should be 3');
assert(isnan(y(4)), 'unique: NaN should be at the end');

% union with NaN
a = [1, NaN, 3];
b = [2, 4];
y = union(a, b);
assert(y(1) == 1, 'union: first element should be 1');
assert(y(2) == 2, 'union: second element should be 2');
assert(y(3) == 3, 'union: third element should be 3');
assert(y(4) == 4, 'union: fourth element should be 4');
assert(isnan(y(5)), 'union: NaN should be at the end');

% sort should preserve NaN count
x = [NaN, 3, NaN, 1];
y = sort(x);
assert(y(1) == 1, 'sort multi-NaN: first should be 1');
assert(y(2) == 3, 'sort multi-NaN: second should be 3');
assert(isnan(y(3)), 'sort multi-NaN: third should be NaN');
assert(isnan(y(4)), 'sort multi-NaN: fourth should be NaN');

disp('SUCCESS');

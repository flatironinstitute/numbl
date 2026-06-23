% Test: repelem on value-class objects builds an object array (the pattern
% used by ultraSEM's @Quad constructor: obj = repelem(obj, nobj, 1)).
% All expected values verified against MATLAB R2025b.

% Block form on a scalar object -> nobj x 1 column object array.
obj = RBox();
arr = repelem(obj, 3, 1);
assert(isequal(size(arr), [3 1]), 'repelem(obj,3,1) size should be [3 1]');

% Each element must be an independent copy (no aliasing).
for k = 1:3
    arr(k).v = k * 10;
end
assert(arr(1).v == 10 && arr(2).v == 20 && arr(3).v == 30, ...
    'repelem copies must be independent');

% 2-arg scalar count on a row vector: each element repeated n times.
a = RBox(1); b = RBox(2);
row = [a b];
r2 = repelem(row, 2);
assert(isequal(size(r2), [1 4]), 'repelem(row,2) size should be [1 4]');
assert(r2(1).v == 1 && r2(2).v == 1 && r2(3).v == 2 && r2(4).v == 2, ...
    'repelem(row,2) values should be [1 1 2 2]');

% Block form on a row vector -> 1x4.
M = repelem(row, 1, 2);
assert(isequal(size(M), [1 4]), 'repelem(row,1,2) size should be [1 4]');
assert(M(1).v == 1 && M(2).v == 1 && M(3).v == 2 && M(4).v == 2, ...
    'repelem(row,1,2) values should be [1 1 2 2]');

% 2-arg scalar count on a column vector preserves the column orientation.
col = [a; b];
c2 = repelem(col, 2);
assert(isequal(size(c2), [4 1]), 'repelem(col,2) size should be [4 1]');
assert(c2(1).v == 1 && c2(4).v == 2, 'repelem(col,2) values');

disp('SUCCESS')

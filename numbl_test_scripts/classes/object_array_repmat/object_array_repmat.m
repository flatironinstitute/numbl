% Test: repmat on value-class objects tiles into an object array (the pattern
% used by spatialmath-matlab's RTBPose.mtimes: out = repmat(obj1, 1, N)).
% repmat TILES the array (vs repelem, which repeats each element).
% All expected values verified against MATLAB R2025b.

% repmat(scalar, 1, N) -> 1xN row object array.
obj = RBox(7);
arr = repmat(obj, 1, 3);
assert(isequal(size(arr), [1 3]), 'repmat(obj,1,3) size should be [1 3]');
assert(arr(1).v == 7 && arr(2).v == 7 && arr(3).v == 7, ...
    'repmat(obj,1,3) values should all be 7');

% Each element must be an independent copy (no aliasing).
for k = 1:3
    arr(k).v = k * 10;
end
assert(arr(1).v == 10 && arr(2).v == 20 && arr(3).v == 30, ...
    'repmat copies must be independent');

% repmat(scalar, n) -> n x n square object array.
sq = repmat(RBox(5), 2);
assert(isequal(size(sq), [2 2]), 'repmat(obj,2) size should be [2 2]');
assert(sq(1).v == 5 && sq(4).v == 5, 'repmat(obj,2) values should all be 5');

% Tiling a row vector: repmat([a b], 1, 2) -> [a b a b].
a = RBox(1); b = RBox(2);
row = [a b];
r2 = repmat(row, 1, 2);
assert(isequal(size(r2), [1 4]), 'repmat(row,1,2) size should be [1 4]');
assert(r2(1).v == 1 && r2(2).v == 2 && r2(3).v == 1 && r2(4).v == 2, ...
    'repmat(row,1,2) values should be [1 2 1 2]');

% Tiling a column vector: repmat([a;b], 2, 1) -> [a;b;a;b].
col = [a; b];
c2 = repmat(col, 2, 1);
assert(isequal(size(c2), [4 1]), 'repmat(col,2,1) size should be [4 1]');
assert(c2(1).v == 1 && c2(2).v == 2 && c2(3).v == 1 && c2(4).v == 2, ...
    'repmat(col,2,1) values should be [1 2 1 2]');

% Block tiling with size vector: repmat([a b], [2 2]) -> 2x4.
M = repmat(row, [2 2]);
assert(isequal(size(M), [2 4]), 'repmat(row,[2 2]) size should be [2 4]');
assert(M(1,1).v == 1 && M(1,2).v == 2 && M(2,3).v == 1 && M(2,4).v == 2, ...
    'repmat(row,[2 2]) tiling values');

disp('SUCCESS')

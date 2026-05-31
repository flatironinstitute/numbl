% First-order diff under the JIT (JS- and C-JIT). Covers row/column
% vectors, matrices, the default vs explicit axis, a single-element
% result, and an empty result.

%!numbl:assert_jit c

% Row vector (default axis = 2)
assert(isequal(diff([1 4 9 16 25]), [3 5 7 9]));
assert(isequal(diff([1 4 9 16 25], 1), [3 5 7 9]));
assert(isequal(diff([2 2 2 2]), [0 0 0]));

% Column vector (default axis = 1)
assert(isequal(diff([10; 20; 30]), [10; 10]));

% Matrix: default axis (dim 1, down columns) and explicit dim 2
M = [3 1; 2 5; 1 4];
assert(isequal(diff(M), [-1 4; -1 -1]));
assert(isequal(diff(M, 1, 1), [-1 4; -1 -1]));
assert(isequal(diff(M, 1, 2), [-2; 3; 3]));

% Explicit dim 2 on a row vector
assert(isequal(diff([1 2 3 4], 1, 2), [1 1 1]));

% Single-element result (2-element input → 1x1)
assert(isequal(diff([1 5]), 4));
r = diff([7 3]);
assert(r(1) == -4);
assert(isequal(size(r), [1 1]));

% Empty result: diff along a singleton axis
e = diff([1 2 3], 1, 1);
assert(isequal(size(e), [0 3]));
assert(isempty(e));

% Negative / mixed values
assert(isequal(diff([5 -2 -2 7]), [-7 0 9]));

% A larger numeric vector
v = [0 1 3 6 10 15];
assert(isequal(diff(v), [1 2 3 4 5]));

disp('SUCCESS');

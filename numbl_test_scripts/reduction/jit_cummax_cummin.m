% cummax / cummin prefix scans under the JIT (JS- and C-JIT).
% Covers vectors, dims, negatives, NaN propagation, and the dim>ndim
% no-op copy.

%!numbl:assert_jit c

% Vectors
assert(isequal(cummax([3 1 4 1 5 9 2 6]), [3 3 4 4 5 9 9 9]));
assert(isequal(cummin([3 1 4 1 5 9 2 6]), [3 1 1 1 1 1 1 1]));

% Negative values
assert(isequal(cummax([-5 -2 -8 -1]), [-5 -2 -2 -1]));
assert(isequal(cummin([-2 -5 -1 -8]), [-2 -5 -5 -8]));

% Single element / two element
assert(cummax(7) == 7);
assert(isequal(cummin([4 9]), [4 4]));

% NaN propagates once it enters the running accumulator (matches
% numbl's cumOp(Math.max) — NOT MATLAB's NaN-skip).
m = cummax([1 3 NaN 2]);
assert(m(1) == 1 && m(2) == 3 && isnan(m(3)) && isnan(m(4)));
m2 = cummax([NaN 1 2]);
assert(isnan(m2(1)) && isnan(m2(2)) && isnan(m2(3)));
n = cummin([5 NaN 1]);
assert(n(1) == 5 && isnan(n(2)) && isnan(n(3)));

% Matrix: default axis (dim 1), explicit dim 1 and dim 2
M = [3 1; 2 5; 1 4];
assert(isequal(cummax(M), [3 1; 3 5; 3 5]));
assert(isequal(cummax(M, 1), [3 1; 3 5; 3 5]));
assert(isequal(cummax(M, 2), [3 3; 2 5; 1 4]));
assert(isequal(cummin(M, 2), [3 1; 2 2; 1 1]));

% Row vector default axis reduces along its only non-singleton dim
assert(isequal(cummax([4 2 7 3]), [4 4 7 7]));

% Reducing along a dim > ndims is a no-op copy of the input
assert(isequal(cummax([10 20 30], 5), [10 20 30]));
assert(isequal(cummin([10 5 30], 3), [10 5 30]));

% +Inf / -Inf handled
assert(isequal(cummax([1 Inf 2]), [1 Inf Inf]));
assert(isequal(cummin([1 -Inf 2]), [1 -Inf -Inf]));

disp('SUCCESS');

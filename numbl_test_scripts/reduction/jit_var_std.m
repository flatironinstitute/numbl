% var / std reductions under the JIT (JS- and C-JIT).
% Covers default/explicit weight, default/explicit dim, 'all', logical
% input, and the std == sqrt(var) relationship.

%!numbl:assert_jit c

% Vector, default weight (÷ n-1)
assert(abs(var([1 2 3 4]) - 5 / 3) < 1e-12);
assert(abs(std([1 2 3 4]) - sqrt(5 / 3)) < 1e-12);
% Population weight (÷ n)
assert(abs(var([1 2 3 4], 1) - 1.25) < 1e-12);
assert(abs(std([1 2 3 4], 1) - sqrt(1.25)) < 1e-12);
% Explicit weight 0 matches default
assert(var([2 4 6 8], 0) == var([2 4 6 8]));

% Scalar input → 0 (regardless of weight)
assert(var(7) == 0);
assert(std(7) == 0);
assert(var(7, 1) == 0);

% std^2 == var (population weight avoids the n-1 mismatch)
x = [2.5 -1 4 0 7 3];
assert(abs(std(x, 1) ^ 2 - var(x, 1)) < 1e-12);

% Matrix: reduce along dim 1 (down columns) and dim 2 (across rows)
A = [1 2 3; 4 6 8];
c = var(A, 0, 1);
assert(isequal(size(c), [1, 3]));
assert(abs(c(1) - 4.5) < 1e-12);
assert(abs(c(2) - 8) < 1e-12);
assert(abs(c(3) - 12.5) < 1e-12);
r = var(A, 0, 2);
assert(isequal(size(r), [2, 1]));
assert(abs(r(1) - 1) < 1e-12);
assert(abs(r(2) - 4) < 1e-12);

% 'all' reduces every element
assert(abs(var(A, 0, 'all') - 6.8) < 1e-12);
assert(abs(std(A, 0, 'all') - sqrt(6.8)) < 1e-12);

% Default-axis var on a matrix reduces along dim 1 (first non-singleton)
d = var(A);
assert(isequal(size(d), [1, 3]));
assert(abs(d(1) - 4.5) < 1e-12);

% std of a column slice
s = std(A(:));
assert(abs(s - sqrt(6.8)) < 1e-12);

% Logical input
assert(abs(var([true false true true]) - 0.25) < 1e-12);

% NaN propagates (no omitnan): any NaN element makes the result NaN
assert(isnan(var([1 NaN 3])));
assert(isnan(std([1 2 NaN])));

% Reducing along a dim > ndims is a no-op copy (matches numbl's
% reduceDim: it returns the input verbatim, NOT zeros).
nop = var([10 20 30], 0, 5);
assert(isequal(nop, [10 20 30]));
assert(isequal(std([10 20 30], 0, 4), [10 20 30]));

% Reducing a row vector along dim 1 leaves length-1 fibers → all zeros
z = var([1 2 3], 0, 1);
assert(isequal(z, [0 0 0]));

disp('SUCCESS');

% Test that max/min on a matrix returns a row vector, not a scalar
% and that subsequent operations treat it as a vector

% max on matrix should return row vector
A = [1 2 3; 4 5 6];
m = max(A);
assert(isequal(size(m), [1 3]), 'max(A) should be 1x3');
assert(isequal(m, [4 5 6]), 'max(A) should be [4 5 6]');

% Comparison on result of max should be element-wise
r = m < 10;
assert(isequal(r, [true true true]), 'max(A) < 10 should be [1 1 1]');

r2 = m > 4;
assert(isequal(r2, [false true true]), 'max(A) > 4 should be [0 1 1]');

% min on matrix should return row vector
n = min(A);
assert(isequal(size(n), [1 3]), 'min(A) should be 1x3');
assert(isequal(n, [1 2 3]), 'min(A) should be [1 2 3]');

r3 = n < 2;
assert(isequal(r3, [true false false]), 'min(A) < 2 should be [1 0 0]');

% Arithmetic on result of max should be element-wise
m2 = max(A) + 1;
assert(isequal(m2, [5 6 7]), 'max(A) + 1 should be [5 6 7]');

% max with abs
m3 = max(abs(A));
r4 = m3 < 10;
assert(isequal(r4, [true true true]), 'max(abs(A)) < 10 should be [1 1 1]');

disp('SUCCESS');

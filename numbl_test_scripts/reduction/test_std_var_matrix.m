% Test std and var on matrices (should reduce along dim 1)

A = [1 2 3; 4 5 6];

% std on matrix - should reduce along dim 1, returning 1x3
r1 = std(A);
expected_std = [sqrt(4.5) sqrt(4.5) sqrt(4.5)];
assert(isequal(size(r1), [1 3]), 'std(matrix) should be 1x3');
assert(max(abs(r1 - expected_std)) < 1e-10, 'std(matrix) values');

% var on matrix - should reduce along dim 1, returning 1x3
r2 = var(A);
assert(isequal(size(r2), [1 3]), 'var(matrix) should be 1x3');
assert(max(abs(r2 - [4.5 4.5 4.5])) < 1e-10, 'var(matrix) values');

% std on vector should still be scalar
r3 = std([1 2 3 4 5]);
assert(isscalar(r3), 'std(vector) is scalar');

% var on vector should still be scalar
r4 = var([1 2 3 4 5]);
assert(isscalar(r4), 'var(vector) is scalar');

% std with population normalization (w=1)
r5 = std(A, 1);
expected_pop = [1.5 1.5 1.5];
assert(isequal(size(r5), [1 3]), 'std(A,1) should be 1x3');
assert(max(abs(r5 - expected_pop)) < 1e-10, 'std(A,1) values');

% Comparison operations on result should work element-wise
r6 = std(A) > 2;
assert(isequal(r6, [true true true]), 'std(A) > 2');

disp('SUCCESS');

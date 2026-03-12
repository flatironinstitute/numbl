% Test that sum/prod/mean on a matrix return row vectors, not scalars,
% and that subsequent operations treat them as vectors.

A = [1 2 3; 4 5 6];

% sum on matrix should return row vector
s = sum(A);
assert(isequal(size(s), [1 3]), 'sum(A) should be 1x3');
assert(isequal(s, [5 7 9]), 'sum(A) should be [5 7 9]');
r1 = s > 6;
assert(isequal(r1, [false true true]), 'sum(A) > 6 should be [0 1 1]');

% Arithmetic on sum result should be element-wise
r2 = s + 1;
assert(isequal(r2, [6 8 10]), 'sum(A) + 1 should be [6 8 10]');

% prod on matrix should return row vector
p = prod(A);
assert(isequal(size(p), [1 3]), 'prod(A) should be 1x3');
assert(isequal(p, [4 10 18]), 'prod(A) should be [4 10 18]');
r3 = p > 5;
assert(isequal(r3, [false true true]), 'prod(A) > 5 should be [0 1 1]');

% mean on matrix should return row vector
m = mean(A);
assert(isequal(size(m), [1 3]), 'mean(A) should be 1x3');
assert(isequal(m, [2.5 3.5 4.5]), 'mean(A) should be [2.5 3.5 4.5]');
r4 = m > 3;
assert(isequal(r4, [false true true]), 'mean(A) > 3 should be [0 1 1]');

% sum on vector should still return scalar
v = [1 2 3];
sv = sum(v);
assert(sv == 6, 'sum([1 2 3]) should be 6');

disp('SUCCESS');

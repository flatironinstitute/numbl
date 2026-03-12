% Test diff with dimension argument: diff(X, N, DIM)

% Test 1: diff along dim 2 (columns) for a matrix
A = [1 3 6 10; 2 5 9 14];
result = diff(A, 1, 2);
assert(isequal(result, [2 3 4; 3 4 5]), 'diff along dim 2');
assert(isequal(size(result), [2 3]), 'diff dim 2 size');

% Test 2: diff along dim 1 (rows) for a matrix — same as default
result2 = diff(A, 1, 1);
expected2 = [1 2 3 4];
assert(isequal(result2, expected2), 'diff along dim 1');
assert(isequal(size(result2), [1 4]), 'diff dim 1 size');

% Test 3: diff along dim 2 with n=2
B = [1 4 9 16; 0 1 4 9];
result3 = diff(B, 2, 2);
assert(isequal(result3, [2 2; 2 2]), 'diff n=2 along dim 2');
assert(isequal(size(result3), [2 2]), 'diff n=2 dim 2 size');

% Test 4: The chebfun pattern - diff(F(1:m-1,:), 1, 2) should reduce cols
F = reshape(1:24, 4, 6);
dfdx = diff(F(1:3, :), 1, 2);
assert(isequal(size(dfdx), [3 5]), 'chebfun pattern: diff along dim 2 reduces cols');

dfdy = diff(F(:, 1:5), 1, 1);
assert(isequal(size(dfdy), [3 5]), 'chebfun pattern: diff along dim 1 reduces rows');

% Test 5: diff of column vector along dim 1
v = [1; 3; 6; 10];
result5 = diff(v, 1, 1);
assert(isequal(result5, [2; 3; 4]), 'diff column vector along dim 1');

% Test 6: diff of row vector along dim 2
w = [1 3 6 10];
result6 = diff(w, 1, 2);
assert(isequal(result6, [2 3 4]), 'diff row vector along dim 2');

disp('SUCCESS');

% Test issymmetric() and ishermitian()

% --- issymmetric ---

% Test 1: symmetric matrix
assert(issymmetric([1 2; 2 3]) == true);

% Test 2: non-symmetric matrix
assert(issymmetric([1 2; 3 4]) == false);

% Test 3: identity is symmetric
assert(issymmetric(eye(3)) == true);

% Test 4: scalar is always symmetric
assert(issymmetric(5) == true);
assert(issymmetric(0) == true);

% Test 5: non-square is not symmetric
assert(issymmetric([1 2 3; 4 5 6]) == false);

% Test 6: skew-symmetric
A = [0 2 -1; -2 0 3; 1 -3 0];
assert(issymmetric(A) == false);
assert(issymmetric(A, 'skew') == true);

% Test 7: zero matrix is both symmetric and skew-symmetric
assert(issymmetric(zeros(3)) == true);
assert(issymmetric(zeros(3), 'skew') == true);

% Test 8: 1x1 matrix
assert(issymmetric([7]) == true);

% Test 9: larger symmetric matrix
A = [4 1 2; 1 5 3; 2 3 6];
assert(issymmetric(A) == true);

% --- ishermitian ---

% Test 10: real symmetric is Hermitian
assert(ishermitian([1 2; 2 3]) == true);

% Test 11: complex Hermitian matrix
A = [1 1+1i; 1-1i 2];
assert(ishermitian(A) == true);

% Test 12: complex non-Hermitian
A = [1 1+1i; 1+1i 2];
assert(ishermitian(A) == false);

% Test 13: real non-symmetric is not Hermitian
assert(ishermitian([1 2; 3 4]) == false);

% Test 14: skew-Hermitian
A = [1i 2+1i; -2+1i 3i];
assert(ishermitian(A, 'skew') == true);

% Test 15: scalar
assert(ishermitian(5) == true);

disp('SUCCESS');

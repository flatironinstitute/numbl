% Test null() — null space / kernel

% Test 1: rank-deficient 2x2
A = [1 2; 3 6];
Z = null(A);
assert(size(Z, 1) == 2);
assert(size(Z, 2) == 1);
% A*Z should be zero
assert(norm(A * Z) < 1e-10);
% Z should be unit length
assert(abs(norm(Z) - 1) < 1e-10);

% Test 2: full-rank matrix has empty null space
A = [1 0; 0 1];
Z = null(A);
assert(size(Z, 1) == 2);
assert(size(Z, 2) == 0);

% Test 3: zero matrix — null space is all of R^n
A = zeros(2, 3);
Z = null(A);
assert(size(Z, 1) == 3);
assert(size(Z, 2) == 3);
% Z should be orthonormal
assert(norm(Z' * Z - eye(3)) < 1e-10);

% Test 4: tall rank-deficient matrix
A = [1 2 3; 2 4 6; 3 6 9; 4 8 12];
Z = null(A);
assert(size(Z, 1) == 3);
assert(size(Z, 2) == 2);
assert(norm(A * Z) < 1e-10);
% Columns should be orthonormal
assert(norm(Z' * Z - eye(2)) < 1e-10);

% Test 5: wide matrix
A = [1 0 0; 0 1 0];
Z = null(A);
assert(size(Z, 1) == 3);
assert(size(Z, 2) == 1);
assert(norm(A * Z) < 1e-10);

% Test 6: scalar
assert(isequal(null(0), 1));
Z = null(5);
assert(size(Z, 1) == 1);
assert(size(Z, 2) == 0);

% Test 7: identity has empty null space
Z = null(eye(4));
assert(size(Z, 1) == 4);
assert(size(Z, 2) == 0);

% Test 8: 1x3 row vector
A = [1 2 3];
Z = null(A);
assert(size(Z, 1) == 3);
assert(size(Z, 2) == 2);
assert(norm(A * Z) < 1e-10);
assert(norm(Z' * Z - eye(2)) < 1e-10);

% Test 9: 3x1 column vector (full rank)
A = [1; 2; 3];
Z = null(A);
assert(size(Z, 1) == 1);
assert(size(Z, 2) == 0);

disp('SUCCESS');

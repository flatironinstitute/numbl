% Test orth() — orthonormal basis for column space

% Test 1: full-rank square matrix
A = [1 2; 3 4];
Q = orth(A);
assert(size(Q, 1) == 2);
assert(size(Q, 2) == 2);
% Q should be orthonormal
assert(norm(Q' * Q - eye(2)) < 1e-10);

% Test 2: rank-deficient matrix
A = [1 2; 2 4];
Q = orth(A);
assert(size(Q, 1) == 2);
assert(size(Q, 2) == 1);
assert(abs(norm(Q) - 1) < 1e-10);

% Test 3: tall matrix
A = [1 0; 0 1; 0 0];
Q = orth(A);
assert(size(Q, 1) == 3);
assert(size(Q, 2) == 2);
assert(norm(Q' * Q - eye(2)) < 1e-10);

% Test 4: wide matrix (rank limited by rows)
A = [1 2 3; 4 5 6];
Q = orth(A);
assert(size(Q, 1) == 2);
assert(size(Q, 2) == 2);
assert(norm(Q' * Q - eye(2)) < 1e-10);

% Test 5: zero matrix
A = zeros(3, 2);
Q = orth(A);
assert(size(Q, 1) == 3);
assert(size(Q, 2) == 0);

% Test 6: identity matrix
Q = orth(eye(3));
assert(size(Q, 1) == 3);
assert(size(Q, 2) == 3);
assert(norm(Q' * Q - eye(3)) < 1e-10);

% Test 7: scalar
Q = orth(5);
assert(isequal(Q, 1));
Q = orth(0);
assert(size(Q, 1) == 1);
assert(size(Q, 2) == 0);

% Test 8: column space should span A's columns
A = [1 1; 1 1; 0 0];
Q = orth(A);
assert(size(Q, 2) == 1);
% Q should be a unit vector in the direction of [1;1;0]
expected = [1; 1; 0] / sqrt(2);
assert(abs(abs(Q' * expected) - 1) < 1e-10);

% Test 9: larger rank-deficient matrix
A = [1 2 3; 4 5 6; 7 8 9];
Q = orth(A);
assert(size(Q, 1) == 3);
assert(size(Q, 2) == 2);
assert(norm(Q' * Q - eye(2)) < 1e-10);

disp('SUCCESS');

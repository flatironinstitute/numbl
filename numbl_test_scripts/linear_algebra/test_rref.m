% Test rref() — reduced row echelon form

% Test 1: identity matrix stays identity
R = rref(eye(3));
assert(norm(R - eye(3)) < 1e-10);

% Test 2: basic 2x3 system
A = [1 2 3; 4 5 6];
R = rref(A);
assert(abs(R(1,1) - 1) < 1e-10);
assert(abs(R(1,2)) < 1e-10);
assert(abs(R(2,1)) < 1e-10);
assert(abs(R(2,2) - 1) < 1e-10);

% Test 3: singular 3x3 matrix
A = [1 2 3; 4 5 6; 7 8 9];
R = rref(A);
% Should have rank 2
assert(abs(R(1,1) - 1) < 1e-10);
assert(abs(R(2,2) - 1) < 1e-10);
assert(abs(R(3,1)) < 1e-10);
assert(abs(R(3,2)) < 1e-10);
assert(abs(R(3,3)) < 1e-10);

% Test 4: with pivot columns output
A = [1 2 3; 4 5 6; 7 8 9];
[R, pivots] = rref(A);
assert(isequal(pivots, [1 2]));

% Test 5: augmented system [A|b]
A = [2 1 -1 8; -3 -1 2 -11; -2 1 2 -3];
R = rref(A);
% Should solve to x=[2; 3; -1]
assert(abs(R(1,4) - 2) < 1e-10);
assert(abs(R(2,4) - 3) < 1e-10);
assert(abs(R(3,4) - (-1)) < 1e-10);

% Test 6: zero matrix
R = rref(zeros(2, 3));
assert(norm(R) < 1e-10);

% Test 7: scalar
R = rref(5);
assert(abs(R - 1) < 1e-10);
R = rref(0);
assert(abs(R) < 1e-10);

% Test 8: wide matrix (more columns than rows)
A = [1 2 3 4; 5 6 7 8];
[R, pivots] = rref(A);
assert(length(pivots) == 2);
assert(abs(R(1,1) - 1) < 1e-10);
assert(abs(R(2,2) - 1) < 1e-10);
assert(abs(R(1,2)) < 1e-10);

% Test 9: tall matrix (more rows than columns)
A = [1 2; 3 4; 5 6];
R = rref(A);
assert(abs(R(1,1) - 1) < 1e-10);
assert(abs(R(2,2) - 1) < 1e-10);
assert(abs(R(3,1)) < 1e-10);
assert(abs(R(3,2)) < 1e-10);

% Test 10: full-rank 3x3
A = [2 -1 0; -1 2 -1; 0 -1 2];
[R, pivots] = rref(A);
assert(norm(R - eye(3)) < 1e-10);
assert(isequal(pivots, [1 2 3]));

disp('SUCCESS');

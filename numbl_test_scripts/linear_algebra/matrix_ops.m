% Matrix operations
tol = 1e-4;

% Matrix multiply
A = [1, 2; 3, 4];
B = [5, 6; 7, 8];
C = A * B;
assert(C(1,1) == 19)
assert(C(1,2) == 22)
assert(C(2,1) == 43)
assert(C(2,2) == 50)

% Element-wise operations
D = A .* B;
assert(D(1,1) == 5)
assert(D(1,2) == 12)
assert(D(2,1) == 21)
assert(D(2,2) == 32)

% Matrix power
I2 = eye(2);
A2 = A * A;
assert(A2(1,1) == 7)
assert(A2(2,2) == 22)

% Transpose
T = A';
assert(T(1,2) == 3)
assert(T(2,1) == 2)

% det
d = det(A);
assert(abs(d - (-2)) < tol)

% trace
tr = trace(A);
assert(abs(tr - 5) < tol)

% norm
v = [3; 4];
assert(abs(norm(v) - 5) < tol)

disp('SUCCESS')

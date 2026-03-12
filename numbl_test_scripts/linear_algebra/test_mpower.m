% Test matrix power (mpower, ^) for matrices

% A^0 should be identity
A = [0 1; -1 0];
B = A^0;
assert(isequal(B, eye(2)), 'A^0 should be identity');

% A^1 should be A
C = A^1;
assert(isequal(C, A), 'A^1 should be A');

% A^2 should be A*A
D = A^2;
assert(isequal(D, A*A), 'A^2 should be A*A');

% A^3 should be A*A*A
E = A^3;
assert(isequal(E, A*A*A), 'A^3 should be A*A*A');

% A^(-1) should be inv(A)
F = [2 0; 0 3];
G = F^(-1);
assert(abs(G(1,1) - 0.5) < 1e-10, 'F^(-1) (1,1)');
assert(abs(G(1,2)) < 1e-10, 'F^(-1) (1,2)');
assert(abs(G(2,1)) < 1e-10, 'F^(-1) (2,1)');
assert(abs(G(2,2) - 1/3) < 1e-10, 'F^(-1) (2,2)');

% A^(-2) should be inv(A)^2
H = F^(-2);
assert(abs(H(1,1) - 0.25) < 1e-10, 'F^(-2) (1,1)');
assert(abs(H(2,2) - 1/9) < 1e-10, 'F^(-2) (2,2)');

% 3x3 matrix
M = [1 2 3; 0 1 4; 0 0 1];
M0 = M^0;
assert(isequal(M0, eye(3)), '3x3 A^0');
M2 = M^2;
assert(isequal(M2, M*M), '3x3 A^2');

disp('SUCCESS');

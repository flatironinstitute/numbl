% Test matrix decompositions

% Matrix inverse
A = [2, 1; 5, 3];
B = inv(A);
% B should be [3, -1; -5, 2]
assert(abs(B(1,1) - 3) < 1e-4);
assert(abs(B(1,2) - (-1)) < 1e-4);
assert(abs(B(2,1) - (-5)) < 1e-4);
assert(abs(B(2,2) - 2) < 1e-4);

% A*inv(A) should be identity
C = A * B;
assert(abs(C(1,1) - 1) < 1e-4);
assert(abs(C(1,2) - 0) < 1e-4);
assert(abs(C(2,1) - 0) < 1e-4);
assert(abs(C(2,2) - 1) < 1e-4);

% QR decomposition
M = [1, 2; 3, 4; 5, 6];
[Q, R] = qr(M);
% Q*R should recover M
MR = Q * R;
assert(abs(MR(1,1) - 1) < 1e-4);
assert(abs(MR(2,1) - 3) < 1e-4);
assert(abs(MR(3,2) - 6) < 1e-4);

% SVD - singular values only (works with JS fallback)
D = [4, 3; 2, 1];
s = svd(D);
% Check that singular values are positive and in descending order
assert(s(1) >= s(2));
assert(s(2) > 0);
% Verify we got a column vector
assert(size(s, 1) == 2);
assert(size(s, 2) == 1);

% SVD - full decomposition
[U, S, V] = svd(D);
% U*S*V' should recover D
DR = U * S * V';
assert(abs(DR(1,1) - 4) < 1e-4);
assert(abs(DR(1,2) - 3) < 1e-4);
assert(abs(DR(2,1) - 2) < 1e-4);
assert(abs(DR(2,2) - 1) < 1e-4);

% SVD - economy mode for tall matrix
T = [1, 2; 3, 4; 5, 6];
[U_econ, S_econ, V_econ] = svd(T, 0);
% U*S*V' should recover T
TR = U_econ * S_econ * V_econ';
assert(abs(TR(1,1) - 1) < 1e-4);
assert(abs(TR(2,1) - 3) < 1e-4);
assert(abs(TR(3,2) - 6) < 1e-4);

disp('SUCCESS')

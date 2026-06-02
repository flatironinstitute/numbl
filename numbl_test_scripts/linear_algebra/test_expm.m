% Matrix exponential expm

tol = 1e-9;

% Scalar: expm(x) == exp(x)
assert(abs(expm(2) - exp(2)) < tol)

% expm of the zero matrix is the identity
Z = expm(zeros(3));
assert(abs(Z(1,1) - 1) < tol)
assert(abs(Z(2,2) - 1) < tol)
assert(abs(Z(1,2)) < tol)

% Diagonal matrix: expm is exp on the diagonal
D = diag([1 2 3]);
E = expm(D);
assert(abs(E(1,1) - exp(1)) < tol)
assert(abs(E(2,2) - exp(2)) < tol)
assert(abs(E(3,3) - exp(3)) < tol)
assert(abs(E(1,2)) < tol)

% MATLAB documentation example (upper triangular, distinct eigenvalues)
A = [1 1 0; 0 0 2; 0 0 -1];
M = expm(A);
assert(abs(M(1,1) - 2.7182818285) < 1e-6)
assert(abs(M(1,2) - 1.7182818285) < 1e-6)
assert(abs(M(1,3) - 1.0861612696) < 1e-6)
assert(abs(M(2,2) - 1.0000000000) < 1e-6)
assert(abs(M(2,3) - 1.2642411177) < 1e-6)
assert(abs(M(3,3) - 0.3678794412) < 1e-6)
% Lower triangle stays zero
assert(abs(M(2,1)) < tol)
assert(abs(M(3,1)) < tol)
assert(abs(M(3,2)) < tol)

% expm(A)*expm(-A) == I for a general matrix
B = [0.4 -0.2 0.1; 0.3 0.5 -0.6; -0.1 0.2 0.7];
P = expm(B) * expm(-B);
assert(abs(P(1,1) - 1) < 1e-8)
assert(abs(P(2,2) - 1) < 1e-8)
assert(abs(P(3,3) - 1) < 1e-8)
assert(abs(P(1,2)) < 1e-8)
assert(abs(P(2,3)) < 1e-8)

% Complex skew-Hermitian argument gives a unitary propagator: U'*U == I.
% This is the core operation behind quantum time evolution, U = expm(-i*H*t).
H = [0 0.1; 0.1 0];
U = expm(-i * H * 3);
UU = U' * U;
assert(abs(UU(1,1) - 1) < tol)
assert(abs(UU(2,2) - 1) < tol)
assert(abs(UU(1,2)) < tol)
assert(abs(UU(2,1)) < tol)

% Complex diagonal: expm(-i*diag(w)*t) == diag(exp(-i*w*t))
W = diag([0.5 1.5 2.5]);
Ud = expm(-i * W * 0.1);
e0 = exp(-i * 0.5 * 0.1);
assert(abs(Ud(1,1) - e0) < tol)
assert(abs(Ud(2,2) - exp(-i * 1.5 * 0.1)) < tol)
assert(abs(Ud(3,3) - exp(-i * 2.5 * 0.1)) < tol)
assert(abs(Ud(1,2)) < tol)

disp('SUCCESS')

% Acceptance test for the sparse direct solvers: kron stays sparse, sparse
% chol / lu return sparse factors, triangular solves and mldivide are exact.
rng(1);

% kron stays sparse and matches dense
A = sprandn(20, 20, 0.2); B = speye(7);
K = kron(A, B); assert(issparse(K)); assert(norm(full(K) - kron(full(A), full(B)), 1) < 1e-12);

% chol of a 2-D Laplacian-based SPD matrix
n = 40; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
L2 = kron(speye(n), D) + kron(D, speye(n));
A = 4*speye(n^2) - L2;                     % SPD, m = 1600
R = chol(A); assert(issparse(R)); assert(istriu(R));
assert(norm(R'*R - A, 1) / norm(A, 1) < 1e-12);
b = randn(n^2, 1); x = R \ (R' \ b);
assert(norm(A*x - b) / norm(b) < 1e-10);
[R2, p] = chol(-A); assert(p > 0);         % not SPD, no throw with two outputs

% lu with pivoting on a nonsymmetric sparse matrix
N = A + 0.3*spdiags(ones(n^2, 1), 1, n^2, n^2);
[Lf, Uf, P] = lu(N); assert(issparse(Lf) && issparse(Uf) && issparse(P));
assert(norm(P*N - Lf*Uf, 1) / norm(N, 1) < 1e-12);
y = Uf \ (Lf \ (P*b)); assert(norm(N*y - b) / norm(b) < 1e-10);

% mldivide on sparse square does not densify and is exact
z = A \ b; assert(norm(A*z - b) / norm(b) < 1e-10);
disp('SUCCESS')

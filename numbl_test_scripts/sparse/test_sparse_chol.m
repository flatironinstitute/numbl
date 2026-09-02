% Sparse Cholesky: upper/lower factors, failure reporting, permuted form,
% and triangular solves with one or many right-hand sides.
rng(2);
n = 12; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
A = 4*speye(n^2) - (kron(speye(n), D) + kron(D, speye(n)));   % SPD, 144x144
m = n^2;

% upper factor
R = chol(A);
assert(issparse(R) && istriu(R) && size(R, 1) == m && size(R, 2) == m);
assert(norm(full(R'*R - A), 1) < 1e-10);

% lower factor
L = chol(A, 'lower');
assert(issparse(L) && istril(L));
assert(norm(full(L*L' - A), 1) < 1e-10);
assert(norm(full(L - R'), 1) < 1e-12);

% [R, p]: success reports p = 0
[R2, p2] = chol(A);
assert(p2 == 0 && norm(full(R2 - R), 1) < 1e-12);

% [R, p]: failure reports the failing column; for sparse input R holds the
% first p-1 rows of the upper factor (size (p-1) x m, as in MATLAB)
B = A; B(5, 5) = -10;
[R3, p3] = chol(B);
assert(p3 == 5);
assert(isequal(size(R3), [4 m]));
M3 = R3'*R3;
assert(norm(full(M3(1:4, :) - B(1:4, :)), 1) < 1e-10);
[L3, p3b] = chol(B, 'lower');
assert(p3b == 5 && isequal(size(L3), [m 4]));

% one output throws for non-SPD input
threw = false;
try
    chol(B);
catch err
    threw = true;
    assert(~isempty(strfind(err.message, 'positive definite')));
end
assert(threw);

% [R, p, P]: permuted factorization R'*R = P'*A*P
[R4, p4, P4] = chol(A);
assert(p4 == 0 && issparse(P4));
assert(norm(full(R4'*R4 - P4'*A*P4), 1) < 1e-10);
[R5, p5, q5] = chol(A, 'vector');
assert(p5 == 0 && isequal(sort(q5(:))', 1:m));
assert(norm(full(R5'*R5 - A(q5, q5)), 1) < 1e-10);

% only the upper triangle is read (no symmetry check)
U = triu(A) + 0.5*tril(A, -1);          % lower part perturbed
Ru = chol(U);
assert(norm(full(Ru - R), 1) < 1e-12);

% triangular solves: single and multiple right-hand sides
b = randn(m, 1);
x = R \ (R' \ b);
assert(norm(A*x - b) / norm(b) < 1e-10);
Bm = randn(m, 3);
X = R \ (R' \ Bm);
assert(isequal(size(X), [m 3]));
assert(norm(A*X - Bm, 1) / norm(Bm, 1) < 1e-10);

% sparse \ sparse stays sparse
bs = sparse(b);
xs = R \ (R' \ bs);
assert(issparse(xs));
assert(norm(A*full(xs) - b) / norm(b) < 1e-10);

% small explicit case cross-checked against the dense factor
S = sparse([4 2 0; 2 5 1; 0 1 3]);
Rs = chol(S);
assert(norm(full(Rs) - chol(full(S)), 1) < 1e-12);

% 1x1 and empty
assert(full(chol(sparse(9))) == 3);
E = chol(sparse(0, 0));
assert(issparse(E) && isempty(E));
disp('SUCCESS')

% Sparse mldivide / mrdivide / inv: triangular detection, the symmetric
% positive definite path, the general LU path, and result types.
rng(4);
n = 9; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
A = 4*speye(n^2) - (kron(speye(n), D) + kron(D, speye(n)));
m = n^2;
b = randn(m, 1);
Bm = randn(m, 4);

% SPD: exact solve, dense result for dense rhs
x = A \ b;
assert(~issparse(x) && isequal(size(x), [m 1]));
assert(norm(A*x - b) / norm(b) < 1e-10);
X = A \ Bm;
assert(norm(A*X - Bm, 1) / norm(Bm, 1) < 1e-10);

% symmetric but indefinite: LU path
Ind = A - 6*speye(m);
xi = Ind \ b;
assert(norm(Ind*xi - b) / norm(b) < 1e-9);

% nonsymmetric
N = A + 0.5*spdiags(ones(m, 1), 2, m, m);
xn = N \ b;
assert(norm(N*xn - b) / norm(b) < 1e-10);
assert(norm(xn - full(N) \ b) / norm(xn) < 1e-9);

% triangular matrices are solved by substitution
U = triu(N);
xu = U \ b;
assert(norm(U*xu - b) / norm(b) < 1e-10);
Lw = tril(N);
xl = Lw \ b;
assert(norm(Lw*xl - b) / norm(b) < 1e-10);
% an explicitly stored zero off the triangle does not defeat detection
U2 = U; U2(5, 1) = 0;
xu2 = U2 \ b;
assert(norm(xu2 - xu) < 1e-12);

% diagonal
Dg = spdiags((1:m)', 0, m, m);
assert(norm(Dg \ b - b ./ (1:m)') < 1e-12);

% sparse \ sparse is sparse
xs = A \ sparse(b);
assert(issparse(xs));
assert(norm(A*xs - b) / norm(b) < 1e-10);

% mrdivide with a sparse divisor
c = randn(1, m);
y = c / N;
assert(isequal(size(y), [1 m]));
assert(norm(y*N - c) / norm(c) < 1e-10);
Y = Bm' / A;
assert(norm(Y*A - Bm', 1) / norm(Bm, 1) < 1e-10);

% O(nnz) sparse matrix norms agree with the dense ones
Nf = full(N);
assert(abs(norm(N, 1) - norm(Nf, 1)) < 1e-12);
assert(abs(norm(N, Inf) - norm(Nf, Inf)) < 1e-12);
assert(abs(norm(N, 'fro') - norm(Nf, 'fro')) < 1e-10);

% inv of a sparse matrix is sparse and correct
Ai = inv(sparse([4 1 0; 1 4 1; 0 1 4]));
assert(issparse(Ai));
assert(norm(full(Ai) - inv([4 1 0; 1 4 1; 0 1 4]), 1) < 1e-12);

% scalar and small cases agree with the dense solver
S2 = sparse([2 1; 1 3]);
assert(norm(full(S2 \ [1; 2]) - [2 1; 1 3] \ [1; 2]) < 1e-12);
assert(abs(full(sparse(4) \ 2) - 0.5) < 1e-15);

% least squares (non-square) still works through the dense path
Tall = sparse([1 0; 0 1; 1 1]);
xls = Tall \ [1; 2; 3];
assert(norm(xls - full(Tall) \ [1; 2; 3]) < 1e-12);
disp('SUCCESS')

% Sparse LU with partial pivoting: three- and four-output forms, 'vector'
% permutations, the two-output permuted L, and rectangular input.
rng(3);
n = 10; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
A = 4*speye(n^2) - (kron(speye(n), D) + kron(D, speye(n)));
m = n^2;
N = A + 0.7*spdiags(ones(m, 1), 1, m, m) - 0.2*spdiags(ones(m, 1), -3, m, m);

% [L, U, P]
[L, U, P] = lu(N);
assert(issparse(L) && issparse(U) && issparse(P));
assert(istril(L) && istriu(U));
assert(all(abs(diag(L) - 1) < 1e-14));
assert(norm(full(P*N - L*U), 1) / norm(N, 1) < 1e-12);
b = randn(m, 1);
x = U \ (L \ (P*b));
assert(norm(N*x - b) / norm(b) < 1e-10);

% 'vector' form
[L2, U2, p] = lu(N, 'vector');
assert(isequal(sort(p(:))', 1:m));
assert(norm(full(N(p, :) - L2*U2), 1) / norm(N, 1) < 1e-12);

% [L, U, P, Q] with column reordering
[L3, U3, P3, Q3] = lu(N);
assert(issparse(Q3));
assert(norm(full(P3*N*Q3 - L3*U3), 1) / norm(N, 1) < 1e-12);
x3 = Q3 * (U3 \ (L3 \ (P3*b)));
assert(norm(N*x3 - b) / norm(b) < 1e-10);
[L4, U4, p4, q4] = lu(N, 'vector');
assert(isequal(sort(q4(:))', 1:m));
assert(norm(full(N(p4, q4) - L4*U4), 1) / norm(N, 1) < 1e-12);

% two outputs: A = L*U with a row-permuted L
[L5, U5] = lu(N);
assert(issparse(L5) && issparse(U5));
assert(norm(full(N - L5*U5), 1) / norm(N, 1) < 1e-12);

% pivot threshold argument (1.0 = plain partial pivoting)
[L6, U6, P6] = lu(N, 1.0);
assert(norm(full(P6*N - L6*U6), 1) / norm(N, 1) < 1e-12);

% a matrix that needs row pivoting (zero diagonal)
Z = sparse([0 1 0; 1 0 2; 0 3 1]);
[Lz, Uz, Pz] = lu(Z);
assert(norm(full(Pz*Z - Lz*Uz), 1) < 1e-12);
assert(norm(full(Z \ [1; 2; 3]) - full(Z) \ [1; 2; 3]) < 1e-12);

% rectangular: tall and wide
T = sparse([1 2; 3 4; 5 6]);
[Lt, Ut, Pt] = lu(T);
assert(isequal(size(Lt), [3 2]) && isequal(size(Ut), [2 2]));
assert(norm(full(Pt*T - Lt*Ut), 1) < 1e-12);
W = sparse([1 2 3; 4 5 6]);
[Lw, Uw, Pw] = lu(W);
assert(isequal(size(Lw), [2 2]) && isequal(size(Uw), [2 3]));
assert(norm(full(Pw*W - Lw*Uw), 1) < 1e-12);
disp('SUCCESS')

% Fill-reducing orderings and banded predicates for sparse matrices.
n = 15; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
A = 4*speye(n^2) - (kron(speye(n), D) + kron(D, speye(n)));
m = n^2;

% symrcm returns a permutation and does not increase the bandwidth of a
% matrix given in a scrambled order
rng(5);
s = randperm(m);
As = A(s, s);
p = symrcm(As);
assert(isequal(size(p), [1 m]) && isequal(sort(p), 1:m));
assert(bandwidth(As(p, p)) <= bandwidth(As));
assert(bandwidth(As(p, p)) < m / 2);

% amd returns a permutation and its Cholesky factor is no denser than the
% natural ordering's
q = amd(A);
assert(isequal(size(q), [1 m]) && isequal(sort(q), 1:m));
assert(nnz(chol(A(q, q))) <= nnz(chol(A)));

% colamd returns a permutation of the columns
N = A + spdiags(ones(m, 1), 3, m, m);
c = colamd(N);
assert(isequal(sort(c), 1:m));
[L, U, P] = lu(N(:, c));
assert(norm(full(P*N(:, c) - L*U), 1) / norm(N, 1) < 1e-12);

% istriu / istril / isbanded on sparse and dense input
R = chol(A);
assert(istriu(R) && ~istril(R));
assert(istril(R') && ~istriu(R'));
assert(istriu(speye(m)) && istril(speye(m)));
assert(istriu(triu(magic(4))) && ~istriu(magic(4)));
assert(istril(tril(magic(4))) && ~istril(magic(4)));
assert(isbanded(A, n, n) && ~isbanded(A, n - 1, n - 1));
assert(isbanded(speye(5), 0, 0));
assert(isbanded(magic(4), 3, 3) && ~isbanded(magic(4), 2, 3));
assert(istriu(sparse(3, 3)) && istril(sparse(3, 3)));
assert(istriu(5) && istril(5));
disp('SUCCESS')

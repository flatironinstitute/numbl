% Test eigs() - subset of eigenvalues/eigenvectors.
% Results are checked against eig() computed in the same run, so the
% assertions hold regardless of the underlying algorithm.

tol = 1e-6;

% Symmetric tridiagonal (1-D Laplacian): SPD with distinct eigenvalues.
n = 12;
A = 2 * eye(n) - diag(ones(n - 1, 1), 1) - diag(ones(n - 1, 1), -1);
desc = sort(eig(A), 'descend');
asc = sort(eig(A));

% Test 1: default returns the 6 largest-magnitude eigenvalues, descending.
d = eigs(A);
assert(numel(d) == 6);
assert(max(abs(d - desc(1:6))) < tol);

% Test 2: eigs(A,k) returns the k largest.
d3 = eigs(A, 3);
assert(numel(d3) == 3);
assert(max(abs(d3 - desc(1:3))) < tol);

% Test 3: 'smallestabs' returns the k smallest, ascending.
ds = eigs(A, 4, 'smallestabs');
assert(max(abs(ds - asc(1:4))) < tol);

% Test 4: 'largestabs' is the default and is a column vector.
dl = eigs(A, 5, 'largestabs');
assert(size(dl, 2) == 1 && size(dl, 1) == 5);
assert(max(abs(dl - desc(1:5))) < tol);

% Test 5: numeric sigma returns the eigenvalues closest to it.
sigma = 2.0;
all_e = eig(A);
[~, ix] = sort(abs(all_e - sigma));
closest = sort(all_e(ix(1:3)));
dn = eigs(A, 3, sigma);
assert(max(abs(sort(dn) - closest)) < tol);

% Test 6: [V,D,flag] - residual A*V = V*D, flag converged, correct shapes.
[V, D, flag] = eigs(A, 3);
assert(flag == 0);
assert(isequal(size(D), [3 3]));
assert(size(V, 1) == n && size(V, 2) == 3);
assert(max(max(abs(A * V - V * D))) < tol);

% Test 7: generalized problem A*V = B*V*D with SPD B.
B = diag(1:n);
[Vg, Dg] = eigs(A, B, 3);
assert(max(max(abs(A * Vg - B * Vg * Dg))) < tol);

% Test 8: eigs(A,[],k) solves the standard problem (empty B).
de = eigs(A, [], 3);
assert(max(abs(de - desc(1:3))) < tol);

% Test 9: 'largestreal'/'smallestreal' on a nonsymmetric matrix with
% real eigenvalues (upper triangular -> eigenvalues are the diagonal).
C = [5 2 1 0 0 0; 0 4 1 1 0 0; 0 0 3 2 1 0; ...
     0 0 0 6 1 1; 0 0 0 0 2 1; 0 0 0 0 0 1];
lr = eigs(C, 2, 'largestreal');
assert(max(abs(sort(real(lr)) - [5; 6])) < tol);
sr = eigs(C, 2, 'smallestreal');
assert(max(abs(sort(real(sr)) - [1; 2])) < tol);

% Test 10: function handle Afun(x) = A*x (default largest magnitude).
Afun = @(x) A * x;
df = eigs(Afun, n, 3);
assert(max(abs(df - desc(1:3))) < tol);

% Test 11: function handle Afun(x) = A\x with 'smallestabs'.
Asolve = @(x) A \ x;
dsm = eigs(Asolve, n, 3, 'smallestabs');
assert(max(abs(sort(dsm) - asc(1:3))) < tol);

% Test 12: k larger than the matrix size is clamped to n.
small = [2 0; 0 3];
dk = eigs(small, 10);
assert(numel(dk) == 2);
assert(max(abs(sort(dk) - [2; 3])) < tol);

% Test 13: symmetric matrix with repeated eigenvalues. The dense symmetric
% path underneath must return an orthonormal basis across the degenerate
% block, which the general path does not guarantee.
m = 8;
u = (1:m)' - (m + 1) / 2;  u = u / sqrt(u' * u);
Qh = eye(m) - 2 * (u * u');
S = Qh * diag([1 2 2 2 5 6 7 8]) * Qh';
S = (S + S') / 2;
assert(max(abs(eigs(S, 4) - [8; 7; 6; 5])) < tol);
[Vs, Ds] = eigs(S, 3, 'smallestabs');
assert(max(abs(sort(diag(Ds)) - [1; 2; 2])) < tol);
assert(max(max(abs(S * Vs - Vs * Ds))) < tol);
assert(max(max(abs(Vs' * Vs - eye(3)))) < tol);

disp('SUCCESS')

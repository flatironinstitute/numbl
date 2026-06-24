% Test gallery('tridiag', ...) — sparse tridiagonal test matrices.

% Form 1: gallery('tridiag', n) == negative second-difference matrix
% (-1 sub/super, 2 on diagonal), sparse.
A = gallery('tridiag', 5);
assert(issparse(A));
assert(isequal(size(A), [5 5]));
assert(nnz(A) == 13);                 % 3n-2 nonzeros
Af = full(A);
expected = [ 2 -1  0  0  0
            -1  2 -1  0  0
             0 -1  2 -1  0
             0  0 -1  2 -1
             0  0  0 -1  2];
assert(isequal(Af, expected));

% Eigenvalues of gallery('tridiag', n) are 2 - 2*cos(k*pi/(n+1)), all > 0.
n = 8;
ev = sort(eig(full(gallery('tridiag', n))));
k = (1:n).';
evExpected = sort(2 - 2*cos(k*pi/(n+1)));
assert(max(abs(ev - evExpected)) < 1e-10);

% Form 2: gallery('tridiag', n, c, d, e) Toeplitz tridiagonal (scalars).
B = full(gallery('tridiag', 4, -1, 2, -1));
assert(isequal(B, full(gallery('tridiag', 4))));
C = full(gallery('tridiag', 4, 3, 5, 7));
expectedC = [5 7 0 0
             3 5 7 0
             0 3 5 7
             0 0 3 5];
assert(isequal(C, expectedC));

% Form 3: gallery('tridiag', x, y, z) from vectors.
c = [1 2 3]; d = [4 5 6 7]; e = [8 9 10];
D = full(gallery('tridiag', c, d, e));
expectedD = [4  8  0  0
             1  5  9  0
             0  2  6 10
             0  0  3  7];
assert(isequal(D, expectedD));

% Explicit zero off-diagonals are not stored.
Z = gallery('tridiag', 5, 0, 2, 0);
assert(nnz(Z) == 5);

% Works as the negative of a stable matrix (M-M.E.S.S. usage pattern).
S = -gallery('tridiag', 6);
assert(all(eig(full(S)) < 0));

disp('SUCCESS');

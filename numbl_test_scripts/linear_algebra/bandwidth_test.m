% bandwidth: lower/upper matrix bandwidth. Verified against MATLAB R2025b.

A = [1 2 0 0; 3 4 5 0; 0 6 7 8; 0 0 9 10];   % tridiagonal
[l, u] = bandwidth(A);
assert(l == 1 && u == 1, 'tridiagonal should be (1,1)');
assert(bandwidth(A, 'lower') == 1, 'lower should be 1');
assert(bandwidth(A, 'upper') == 1, 'upper should be 1');
assert(bandwidth(A) == 1, 'single-output bandwidth(A) returns lower');

% Zero / diagonal / scalar matrices have zero bandwidth.
[l, u] = bandwidth(zeros(3)); assert(l == 0 && u == 0, 'zeros');
[l, u] = bandwidth(diag([1 2 3])); assert(l == 0 && u == 0, 'diag');
[l, u] = bandwidth(5); assert(l == 0 && u == 0, 'scalar');
[l, u] = bandwidth([]); assert(l == 0 && u == 0, 'empty');

% Upper-triangular-only nonzero.
F = [0 0 1; 0 0 0; 0 0 0];
[l, u] = bandwidth(F);
assert(l == 0 && u == 2, 'single upper corner should be (0,2)');

% Lower-triangular matrix: zero upper bandwidth.
L = [1 0 0; 2 3 0; 4 5 6];
[l, u] = bandwidth(L);
assert(l == 2 && u == 0, 'lower-triangular should be (2,0)');

% Sparse matrix.
S = sparse(A);
[l, u] = bandwidth(S);
assert(l == 1 && u == 1, 'sparse bandwidth should match dense');

% Complex matrix: nonzero imaginary parts count as nonzeros.
C = [0 0 0; 0 0 0; 2i 0 0];
[l, u] = bandwidth(C);
assert(l == 2 && u == 0, 'complex entry should count');

disp('SUCCESS')

% null: orthonormal basis (columns) for the null space of a matrix, via SVD.
% The basis itself is not unique (only the subspace is), so we assert
% basis-invariant properties that hold in both numbl and MATLAB.

% 1. Full-rank square matrix -> empty null space (n x 0).
A = magic(3);
Z = null(A);
assert(isequal(size(Z), [3 0]), 'null of full-rank 3x3 should be 3x0');

% 2. Rank-deficient square matrix (row 2 = 2*row 1) -> 1-D null space.
A = [1 2 3; 2 4 6; 1 1 1];
Z = null(A);
assert(isequal(size(Z), [3 1]), 'null should be 3x1');
assert(norm(A*Z) < 1e-10, 'A*Z should be ~0');
assert(abs(norm(Z) - 1) < 1e-10, 'null basis vector should be a unit vector');

% 3. Wide matrix (more columns than rows).
A = [1 0 -1 0; 0 1 0 -1];
Z = null(A);
assert(isequal(size(Z), [4 2]), 'null should be 4x2');
assert(norm(A*Z) < 1e-10, 'A*Z ~0');
assert(norm(Z'*Z - eye(2)) < 1e-10, 'columns should be orthonormal');

% 4. Zero matrix -> full space; an n x n orthonormal basis.
Z = null(zeros(3));
assert(isequal(size(Z), [3 3]), 'null(zeros(3)) should be 3x3');
assert(norm(Z'*Z - eye(3)) < 1e-10, 'columns should be orthonormal');

% 5. The null-space projector P = Z*Z' is basis-independent.
A = [1 1 1 1];
Z = null(A);
assert(isequal(size(Z), [4 3]), 'null(1x4) should be 4x3');
P = Z * Z';
assert(norm(P * A') < 1e-10, 'projector should annihilate the row space');

disp('SUCCESS')

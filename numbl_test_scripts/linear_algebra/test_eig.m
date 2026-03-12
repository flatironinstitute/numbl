% Test eig() function

% Test 1: eigenvalues of diagonal matrix
A = diag([1 2 3]);
e = eig(A);
e_sorted = sort(e);
assert(abs(e_sorted(1) - 1) < 1e-10);
assert(abs(e_sorted(2) - 2) < 1e-10);
assert(abs(e_sorted(3) - 3) < 1e-10);

% Test 2: eigenvalues of symmetric 2x2 matrix
A = [2 1; 1 3];
e = eig(A);
e_sorted = sort(e);
expected1 = (5 - sqrt(5)) / 2;
expected2 = (5 + sqrt(5)) / 2;
assert(abs(e_sorted(1) - expected1) < 1e-10);
assert(abs(e_sorted(2) - expected2) < 1e-10);

% Test 3: [V,D] = eig(A) - verify A*V = V*D
A = [4 1; 2 3];
[V, D] = eig(A);
residual = A * V - V * D;
assert(max(max(abs(residual))) < 1e-10);

% Test 4: [V,D,W] = eig(A) - verify W'*A = D*W'
A = [4 1; 2 3];
[V, D, W] = eig(A);
residual2 = W' * A - D * W';
assert(max(max(abs(residual2))) < 1e-10);

% Test 5: eigenvalues of scalar
e = eig(5);
assert(abs(e - 5) < 1e-10);

% Test 6: complex eigenvalues (rotation matrix)
A = [0 -1; 1 0];
e = eig(A);
% eigenvalues should be +/- i
e_imag_sorted = sort(imag(e));
assert(abs(e_imag_sorted(1) - (-1)) < 1e-10);
assert(abs(e_imag_sorted(2) - 1) < 1e-10);
assert(abs(real(e(1))) < 1e-10);
assert(abs(real(e(2))) < 1e-10);

% Test 7: complex eigenvalues with [V,D] = eig(A)
A = [0 -1; 1 0];
[V, D] = eig(A);
residual3 = A * V - V * D;
assert(max(max(abs(residual3))) < 1e-10);

% Test 8: 'nobalance' option
A = [4 1; 2 3];
e_nb = eig(A, 'nobalance');
e_nb_sorted = sort(real(e_nb));
e_sorted2 = sort(eig(A));
assert(max(abs(e_nb_sorted - e_sorted2)) < 1e-10);

% Test 9: 'vector' output form
A = [4 1; 2 3];
[V, d] = eig(A, 'vector');
assert(size(d, 2) == 1);
assert(size(d, 1) == 2);

% Test 10: 'matrix' output form (default)
A = [4 1; 2 3];
[V, D] = eig(A, 'matrix');
% D should be diagonal
assert(abs(D(1,2)) < 1e-10);
assert(abs(D(2,1)) < 1e-10);

% Test 11: larger matrix (4x4)
A = [4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4];
[V, D] = eig(A);
residual4 = A * V - V * D;
assert(max(max(abs(residual4))) < 1e-10);

disp('SUCCESS')

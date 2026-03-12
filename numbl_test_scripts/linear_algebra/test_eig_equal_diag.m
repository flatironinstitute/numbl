% Test eig with 2x2 matrices having equal diagonal entries
% Bug: ts-lapack fallback returns wrong eigenvalues (e.g. -Infinity)
% when the diagonal entries are equal

% Symmetric 2x2 with equal diagonals
E = [2 1; 1 2];
eigenvals = sort(eig(E));
assert(abs(eigenvals(1) - 1) < 1e-8);
assert(abs(eigenvals(2) - 3) < 1e-8);

% Another symmetric case
E2 = [3 1; 1 3];
ev2 = sort(eig(E2));
assert(abs(ev2(1) - 2) < 1e-8);
assert(abs(ev2(2) - 4) < 1e-8);

% Swap matrix [0 1; 1 0]
E3 = [0 1; 1 0];
ev3 = sort(eig(E3));
assert(abs(ev3(1) - (-1)) < 1e-8);
assert(abs(ev3(2) - 1) < 1e-8);

% 2-output form [V, D] = eig(A)
[V, D] = eig(E);
eigenvals2 = sort(diag(D));
assert(abs(eigenvals2(1) - 1) < 1e-8);
assert(abs(eigenvals2(2) - 3) < 1e-8);
% Verify A*V = V*D
residual = E * V - V * D;
assert(max(max(abs(residual))) < 1e-8);

disp('SUCCESS');

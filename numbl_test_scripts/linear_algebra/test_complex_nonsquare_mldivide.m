% Complex least-squares / minimum-norm solve (A\b for non-square complex A)
% must be correct in the pure-TypeScript LAPACK fallback (no native addon).
% Regression: the hand-rolled complex Householder QR applied H instead of
% H^H (used tau, not conj(tau)), giving a non-unitary transform and wrong
% answers for non-square complex systems.

% overdetermined (m > n): least-squares solution => normal equations hold
A = [1+2i, 3; 4, 5-1i; 2, 1i];
b = [1; 2; 3];
x = A \ b;
assert(norm(A' * (A*x - b)) < 1e-10);

% overdetermined, multiple right-hand sides
B = [1, 0; 2, 1i; 3, 2];
X = A \ B;
assert(norm(A' * (A*X - B)) < 1e-10);

% underdetermined (m < n): the solution must satisfy A*x = b
A2 = [1+2i, 3, 5; 4, 5-1i, 2];
b2 = [1; 2];
x2 = A2 \ b2;
assert(norm(A2*x2 - b2) < 1e-10);

disp('SUCCESS')

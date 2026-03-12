% Test cond builtin

% Identity matrix: condition number is 1
A = eye(3);
assert(abs(cond(A) - 1) < 1e-10);

% Scalar
assert(cond(5) == 1);
assert(cond(0) == Inf);

% 2x2 matrix
B = [1 0; 0 2];
% Singular values are 2 and 1, so cond = 2
assert(abs(cond(B) - 2) < 1e-10);

% Singular matrix: cond should be very large
C = [1 2; 2 4];
assert(cond(C) > 1e6);

% Well-conditioned matrix
D = [2 1; 1 3];
c = cond(D);
assert(c > 1);
assert(isfinite(c));

% Complex scalar
assert(cond(3 + 4i) == 1);
assert(cond(0 + 0i) == Inf);

disp('SUCCESS');

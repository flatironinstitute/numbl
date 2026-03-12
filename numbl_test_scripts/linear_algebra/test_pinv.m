% Test pinv (pseudoinverse) builtin

% Square invertible matrix
A = [1 2; 3 4];
B = pinv(A);
assert(norm(A * B * A - A) < 1e-10);
assert(norm(B * A * B - B) < 1e-10);

% Rectangular matrix (more rows than cols)
A2 = [1 2; 3 4; 5 6];
B2 = pinv(A2);
assert(isequal(size(B2), [2, 3]));
assert(norm(A2 * B2 * A2 - A2) < 1e-10);

% Rectangular matrix (more cols than rows)
A3 = [1 2 3; 4 5 6];
B3 = pinv(A3);
assert(isequal(size(B3), [3, 2]));
assert(norm(A3 * B3 * A3 - A3) < 1e-10);

% Identity
I = eye(3);
assert(norm(pinv(I) - I) < 1e-10);

% Zero matrix
Z = zeros(2, 3);
assert(norm(pinv(Z)) < 1e-10);

% Scalar
assert(abs(pinv(4) - 0.25) < 1e-10);
assert(pinv(0) == 0);

disp('SUCCESS');

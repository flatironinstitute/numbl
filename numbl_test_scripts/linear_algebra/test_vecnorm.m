% Test vecnorm builtin

% --- 1-norm and 2-norm of a vector ---
x = [2 2 2];
assert(abs(vecnorm(x) - sqrt(12)) < 1e-10);
assert(abs(vecnorm(x, 1) - 6) < 1e-10);

% --- Inf-norm of a vector ---
assert(abs(vecnorm([3 -5 2], Inf) - 5) < 1e-10);

% --- 2-norm of matrix columns (default) ---
A = [2 0 1; -1 1 0; -3 3 0];
n = vecnorm(A);
assert(abs(n(1) - sqrt(14)) < 1e-10);
assert(abs(n(2) - sqrt(10)) < 1e-10);
assert(abs(n(3) - 1) < 1e-10);

% --- 1-norm of matrix columns ---
n1 = vecnorm(A, 1);
assert(abs(n1(1) - 6) < 1e-10);
assert(abs(n1(2) - 4) < 1e-10);
assert(abs(n1(3) - 1) < 1e-10);

% --- Inf-norm of matrix columns ---
ninf = vecnorm(A, Inf);
assert(abs(ninf(1) - 3) < 1e-10);
assert(abs(ninf(2) - 3) < 1e-10);
assert(abs(ninf(3) - 1) < 1e-10);

% --- vecnorm along dim=2 (row norms) ---
n2 = vecnorm(A, 2, 2);
assert(abs(n2(1) - sqrt(5)) < 1e-10);
assert(abs(n2(2) - sqrt(2)) < 1e-10);
assert(abs(n2(3) - sqrt(18)) < 1e-10);

% --- 1-norm along dim=2 ---
n12 = vecnorm(A, 1, 2);
assert(abs(n12(1) - 3) < 1e-10);
assert(abs(n12(2) - 2) < 1e-10);
assert(abs(n12(3) - 6) < 1e-10);

% --- Column vector (default dim should be 1) ---
v = [3; 4];
assert(abs(vecnorm(v) - 5) < 1e-10);

% --- Row vector (default dim should be 2) ---
v2 = [3 4];
assert(abs(vecnorm(v2) - 5) < 1e-10);

% --- Scalar input ---
assert(abs(vecnorm(5) - 5) < 1e-10);
assert(abs(vecnorm(-3) - 3) < 1e-10);

% --- p=3 norm ---
x3 = [1 2 3];
expected = (1^3 + 2^3 + 3^3)^(1/3);
assert(abs(vecnorm(x3, 3) - expected) < 1e-10);

% --- Complex vector ---
z = [3+4i, 0];
assert(abs(vecnorm(z) - 5) < 1e-10);

% --- Complex matrix columns ---
Z = [1+1i 2; 3 4+1i];
nz = vecnorm(Z);
expected1 = sqrt(abs(1+1i)^2 + abs(3)^2);
expected2 = sqrt(abs(2)^2 + abs(4+1i)^2);
assert(abs(nz(1) - expected1) < 1e-10);
assert(abs(nz(2) - expected2) < 1e-10);

% --- dim greater than ndims returns abs(A) ---
A2 = [1 -2; 3 -4];
n3 = vecnorm(A2, 2, 3);
assert(all(all(abs(n3 - abs(A2)) < 1e-10)));

% --- Size preservation ---
B = [1 2 3; 4 5 6];
nb = vecnorm(B);
assert(isequal(size(nb), [1 3]));
nb2 = vecnorm(B, 2, 2);
assert(isequal(size(nb2), [2 1]));

% --- NaN propagation ---
xnan = [1 NaN 3];
assert(isnan(vecnorm(xnan)));

disp('SUCCESS');

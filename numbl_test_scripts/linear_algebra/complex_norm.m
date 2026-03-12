% Test norm() with complex input

% Scalar complex: norm(1+2i) should be sqrt(5)
z = 1 + 2i;
n = norm(z);
assert(abs(n - sqrt(5)) < 1e-10);

% Vector norm with complex tensor: norm([1+2i, 3+4i]) = sqrt(|1+2i|^2 + |3+4i|^2)
% = sqrt(5 + 25) = sqrt(30)
v = [1+2i, 3+4i];
n2 = norm(v);
assert(abs(n2 - sqrt(30)) < 1e-10);

% Infinity norm: max absolute value
n3 = norm(v, Inf);
assert(abs(n3 - 5) < 1e-10);

% 1-norm: sum of absolute values
n4 = norm(v, 1);
assert(abs(n4 - (sqrt(5) + 5)) < 1e-10);

disp('SUCCESS')

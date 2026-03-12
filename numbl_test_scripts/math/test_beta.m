% Test beta function: beta(x,y) = gamma(x)*gamma(y)/gamma(x+y)

% Scalar inputs
assert(abs(beta(1, 5) - 0.2) < 1e-10);
assert(abs(beta(3, sqrt(2)) - 0.1716) < 1e-4);
assert(abs(beta(pi, exp(1)) - 0.0379) < 1e-4);
assert(abs(beta(2, 3) - 1/12) < 1e-10);
assert(abs(beta(4, 4) - 1/140) < 1e-10);

% Zero argument gives Inf
assert(isinf(beta(0, 1)));

% Matrix inputs
A = [1 2; 3 4];
B = beta(A, 1);
assert(abs(B(1,1) - 1) < 1e-10);
assert(abs(B(1,2) - 0.5) < 1e-10);
assert(abs(B(2,1) - 1/3) < 1e-10);
assert(abs(B(2,2) - 0.25) < 1e-10);

disp('SUCCESS');

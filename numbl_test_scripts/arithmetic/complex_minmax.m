% Test that min/max of complex tensors returns complex values
% and that arithmetic on the result works correctly

x = [1+2i, 3+4i, 5+6i];

% max should return a complex number
m = max(x);
assert(~isreal(m));
y = 1 + m;
assert(~isreal(y));
assert(abs(real(y) - 6) < 1e-10);
assert(abs(imag(y) - 6) < 1e-10);

% min should return a complex number
n = min(x);
assert(~isreal(n));
z = n + 2;
assert(~isreal(z));
assert(abs(real(z) - 3) < 1e-10);
assert(abs(imag(z) - 2) < 1e-10);

fprintf('SUCCESS\n');

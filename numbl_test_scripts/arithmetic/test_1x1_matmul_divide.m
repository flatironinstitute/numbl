% Test that 1x1 matmul results behave as scalars in division and multiplication
a = [1; 2]' * [3; 4];  % = 11, 1x1 via matmul
b = [1; 2]' * [5; 6];  % = 17, 1x1 via matmul
c = a / b;              % = 11/17, should behave as scalar
x = [1; 2];
y = c * x;              % Should work: scalar * vector
assert(abs(y(1) - 11/17) < 1e-10);
assert(abs(y(2) - 22/17) < 1e-10);

% Also test mldivide: b \ a should be same as a / b for scalars
c2 = b \ a;
y2 = c2 * x;
assert(abs(y2(1) - 11/17) < 1e-10);
assert(abs(y2(2) - 22/17) < 1e-10);

% Test that the result is truly scalar-like
assert(isscalar(c));
assert(isscalar(c2));

disp('SUCCESS')

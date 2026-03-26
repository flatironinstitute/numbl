% Test: power() function with negative bases must return complex results
% (power(a,b) is the functional form of a.^b)

% Scalar: power(-1, 0.5) should equal i
r1 = power(-1, 0.5);
assert(abs(real(r1)) < 1e-10);
assert(abs(imag(r1) - 1) < 1e-10);

% Scalar: power(-4, 0.5) should equal 2i
r2 = power(-4, 0.5);
assert(abs(real(r2)) < 1e-10);
assert(abs(imag(r2) - 2) < 1e-10);

% Array: power on vector with negative values
x = [-4; -1; 0; 1];
y = power(x, 0.5);
assert(abs(imag(y(1)) - 2) < 1e-10);
assert(abs(imag(y(2)) - 1) < 1e-10);
assert(y(3) == 0);
assert(y(4) == 1);

% Via function handle (used by chebfun compose)
f = @(z) power(z, 0.5);
y2 = f(x);
assert(abs(imag(y2(1)) - 2) < 1e-10);
assert(abs(imag(y2(2)) - 1) < 1e-10);

% Integer exponents stay real
assert(power(-2, 2) == 4);
assert(power(-2, 3) == -8);

% power() and .^ must agree
x2 = [-3; -2; -1; 1; 2];
assert(norm(power(x2, 0.5) - x2.^0.5) < 1e-10);

disp('SUCCESS');

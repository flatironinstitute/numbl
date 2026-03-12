% Test: (-1).^(array of integers) should produce real results, not complex
% In MATLAB, (-1).^n for integer n is always real.
% Bug: when the exponent is a tensor of integers, numbl used the complex
% power path exp(n*log(-1)), producing tiny imaginary parts.

% Scalar exponent (already worked)
assert((-1)^2 == 1);
assert((-1)^3 == -1);

% Tensor exponent with all-integer values
x = (-1).^(1:6);
assert(isreal(x));
assert(isequal(x, [-1 1 -1 1 -1 1]));

% Negative integer exponents
y = (-1).^(-3:3);
assert(isreal(y));
assert(isequal(y, [-1 1 -1 1 -1 1 -1]));

% Non-trivial negative base with integer exponents
z = (-2).^(0:4);
assert(isreal(z));
assert(isequal(z, [1 -2 4 -8 16]));

% Mixed negative and positive base, integer exponents
w = [-1 2 -3].^[2 3 4];
assert(isreal(w));
assert(isequal(w, [1 8 81]));

% Non-integer exponent should still produce complex
c = (-1).^0.5;
assert(~isreal(c));
assert(abs(c - 1i) < 1e-14);

disp('SUCCESS');

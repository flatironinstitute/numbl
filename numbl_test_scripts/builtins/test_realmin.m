% Test realmin constant value
% realmin should return the smallest normalized double-precision float
% which is 2.2250738585072014e-308, NOT the smallest subnormal (5e-324)

% Check the value is correct
assert(abs(realmin - 2.2250738585072014e-308) < 1e-320);

% Check realmin / 2 is still positive (subnormal range)
x = realmin / 2;
assert(x > 0);
assert(x < realmin);

% Check realmin is positive
assert(realmin > 0);

% Check realmin * 2 is larger
assert(realmin * 2 > realmin);

disp('SUCCESS');

% single: rounds values to single precision. numbl has no distinct single
% type (everything is double), but the precision rounding is observable.
% Verified against MATLAB R2025b.

% Exactly representable values are unchanged.
assert(single(2) == 2, 'single(2) == 2');
assert(single(0.5) == 0.5, 'single(0.5) == 0.5');
assert(isequal(single([1 2 3]), [1 2 3]), 'integer vector unchanged');

% A value needing more than 24 bits of mantissa is rounded.
x = 1 + 2^-30;                  % not representable in single
assert(single(x) == 1, 'single(1+2^-30) rounds to 1');
assert(x ~= 1, 'double keeps the bit');

% pi loses its tail in single precision.
assert(single(pi) ~= pi, 'single(pi) differs from double pi');
assert(abs(single(pi) - pi) < 1e-6, 'but stays close');

% Complex values round both parts.
z = single(pi + pi*1i);
assert(real(z) == single(pi) && imag(z) == single(pi), 'complex rounded');

% Tensors round elementwise and preserve shape.
M = single([pi 1; 2 0.5]);
assert(isequal(size(M), [2 2]), 'shape preserved');
assert(M(2,2) == 0.5 && M(1,1) == single(pi), 'elementwise rounding');

% Logicals and chars convert to numeric.
assert(single(true) == 1, 'single(true) == 1');
assert(single('A') == 65, 'single(''A'') == 65');

disp('SUCCESS')

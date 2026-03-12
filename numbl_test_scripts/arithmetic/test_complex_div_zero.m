% Test complex division by zero returns Inf, not NaN

% Non-zero complex / 0
r = (5 + 3i) / 0;
assert(real(r) == Inf, 'real(5+3i)/0 should be Inf');
assert(imag(r) == Inf, 'imag(5+3i)/0 should be Inf');

r = (-5 + 3i) / 0;
assert(real(r) == -Inf, 'real(-5+3i)/0 should be -Inf');
assert(imag(r) == Inf, 'imag(-5+3i)/0 should be Inf');

r = (5 - 3i) / 0;
assert(real(r) == Inf, 'real(5-3i)/0 should be Inf');
assert(imag(r) == -Inf, 'imag(5-3i)/0 should be -Inf');

r = (-5 - 3i) / 0;
assert(real(r) == -Inf, 'real(-5-3i)/0 should be -Inf');
assert(imag(r) == -Inf, 'imag(-5-3i)/0 should be -Inf');

% Pure real complex / 0
r = (5 + 0i) / 0;
assert(real(r) == Inf, 'real(5+0i)/0 should be Inf');
assert(imag(r) == 0, 'imag(5+0i)/0 should be 0');

% Pure imaginary / 0
r = (0 + 3i) / 0;
assert(real(r) == 0, 'real(0+3i)/0 should be 0');
assert(imag(r) == Inf, 'imag(0+3i)/0 should be Inf');

% Element-wise division
r = (5 + 3i) ./ 0;
assert(real(r) == Inf, 'real elem (5+3i)./0 should be Inf');
assert(imag(r) == Inf, 'imag elem (5+3i)./0 should be Inf');

r = (-5 - 3i) ./ 0;
assert(real(r) == -Inf, 'real elem (-5-3i)./0 should be -Inf');
assert(imag(r) == -Inf, 'imag elem (-5-3i)./0 should be -Inf');

% Complex / complex zero
r = (5 + 3i) / (0 + 0i);
assert(real(r) == Inf, 'real (5+3i)/(0+0i) should be Inf');
assert(imag(r) == Inf, 'imag (5+3i)/(0+0i) should be Inf');

disp('SUCCESS');

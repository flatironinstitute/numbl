% Test that diff of a scalar returns 0x1 (empty column), not 1x0
% In MATLAB, diff(5) returns a 0x1 double (empty column vector)

% Test 1: diff of scalar should be 0x0
d = diff(5);
assert(isequal(size(d), [0 0]), 'diff(scalar) should be 0x0');

% Test 2: vertical concatenation of false with diff(scalar) should work
% since false is 1x1 and diff(scalar) is 0x0 (empty, ignored in concat)
y = [false; diff(5)];
assert(isequal(size(y), [1 1]), 'concat false with diff(scalar)');
assert(y == 0, 'result should be 0 (false)');

% Test 3: the original failing pattern
x = [1;2;3;4];
rBreaks = x;
rootTol = 0.5;
y = [false; diff(rBreaks) < rootTol];
assert(isequal(size(y), [4 1]), 'chebfun pattern size');
assert(y(1) == 0, 'first element is false');

disp('SUCCESS');

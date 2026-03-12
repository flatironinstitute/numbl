% Test two-argument max/min NaN handling
% MATLAB ignores NaN when the other value is non-NaN

% Scalar cases
assert(max(3, NaN) == 3);
assert(min(3, NaN) == 3);
assert(max(NaN, 3) == 3);
assert(min(NaN, 3) == 3);

% Both NaN
assert(isnan(max(NaN, NaN)));
assert(isnan(min(NaN, NaN)));

% Element-wise with vectors
a = [1 NaN 3];
b = [NaN 2 NaN];
r = max(a, b);
assert(r(1) == 1);
assert(r(2) == 2);
assert(r(3) == 3);

r = min(a, b);
assert(r(1) == 1);
assert(r(2) == 2);
assert(r(3) == 3);

disp('SUCCESS');

% interp1 extras: numeric extrapolation (fill) value, and the
% piecewise-polynomial output form interp1(x, y, 'linear', 'pp').

x = [1 2 3];
y = [10 20 30];

% Numeric fill value for out-of-range queries
r = interp1(x, y, [0 2.5 5], 'linear', 0);
assert(isequal(r, [0 25 0]));
r2 = interp1(x, y, [0 2 5], 'linear', -1);
assert(isequal(r2, [-1 20 -1]));

% Default out-of-range is still NaN
r3 = interp1(x, y, [0 2], 'linear');
assert(isnan(r3(1)) && r3(2) == 20);

% 'extrap' still extrapolates linearly
assert(interp1(x, y, 4, 'linear', 'extrap') == 40);

% pp output form
pp = interp1(x, y, 'linear', 'pp');
assert(strcmp(pp.form, 'pp'));
assert(pp.pieces == 2);
assert(pp.order == 2);
assert(ppval(pp, 1) == 10);
assert(ppval(pp, 2.5) == 25);
assert(ppval(pp, 3) == 30);
assert(abs(ppval(pp, 1.25) - 12.5) < 1e-12);

disp('SUCCESS');

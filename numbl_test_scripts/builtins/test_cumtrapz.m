% Test cumtrapz - cumulative trapezoidal integration

% Basic cumtrapz with unit spacing
y = [1 2 3 4 5];
ct = cumtrapz(y);
expected = [0 1.5 4 7.5 12];
assert(norm(ct - expected) < 1e-10);

% cumtrapz with x spacing
x = [0 1 2 3 4];
y2 = [0 1 4 9 16];
ct2 = cumtrapz(x, y2);
expected2 = [0 0.5 3 9.5 22];
assert(norm(ct2 - expected2) < 1e-10);

% cumtrapz with non-uniform spacing
x3 = [0 0.5 1 2];
y3 = [0 0.25 1 4];
ct3 = cumtrapz(x3, y3);
expected3 = [0 0.0625 0.375 2.875];
assert(norm(ct3 - expected3) < 1e-10);

% Single element
ct4 = cumtrapz([5]);
assert(ct4 == 0);

% Two elements
ct5 = cumtrapz([3 7]);
assert(abs(ct5(1) - 0) < 1e-10);
assert(abs(ct5(2) - 5) < 1e-10);

% Column vector preservation
y_col = [1; 2; 3; 4];
ct6 = cumtrapz(y_col);
assert(size(ct6, 1) == 4);
assert(size(ct6, 2) == 1);

disp('SUCCESS');

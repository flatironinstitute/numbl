% Test that nan() works as an alias for NaN()
% In MATLAB, nan and NaN are interchangeable

% Scalar
assert(isnan(nan), 'nan should return NaN');

% Array constructor
x = nan(2, 3);
assert(all(size(x) == [2, 3]), 'nan(2,3) should create 2x3 matrix');
assert(all(all(isnan(x))), 'nan(2,3) should be all NaN');

% Square shorthand
y = nan(3);
assert(all(size(y) == [3, 3]), 'nan(3) should create 3x3 matrix');

disp('SUCCESS');

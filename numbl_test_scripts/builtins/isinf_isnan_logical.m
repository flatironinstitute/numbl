% Test that isinf, isnan, isfinite return logical values (not double)
% This is important for logical indexing: A(row, isinf(x)) should work

% Scalar inputs should return logical
assert(islogical(isinf(1)), 'isinf should return logical for scalar');
assert(islogical(isnan(1)), 'isnan should return logical for scalar');
assert(islogical(isfinite(1)), 'isfinite should return logical for scalar');

% Negation of logical results should work
assert(~isnan(1), '~isnan(1) should be true');
assert(~isinf(1), '~isinf(1) should be true');
assert(isfinite(1), 'isfinite(1) should be true');

% Logical indexing with result of isinf
a = [10; 20; 30];
ind = isinf(1.5);
b = a(2, ind);
assert(isempty(b), 'a(2, isinf(1.5)) should be empty');

disp('SUCCESS');

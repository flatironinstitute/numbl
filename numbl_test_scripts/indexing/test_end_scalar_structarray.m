% Test that the 'end' keyword resolves correctly for scalar, struct array,
% and sparse matrix bases (not just tensors).

% Scalar number: end should be 1
x = 5;
assert(x(end) == 5, 'scalar end failed');
assert(x(1:end) == 5, 'scalar 1:end failed');

% Struct array: end should be length of array
s(1).a = 10;
s(2).a = 20;
s(3).a = 30;
assert(s(end).a == 30, 'struct array end failed');
result = [s(1).a, s(end).a];
assert(isequal(result, [10, 30]), 'struct array [s(1).a, s(end).a] failed');

% Struct array with range
vals = arrayfun(@(x) x.a, s(2:end));
assert(isequal(vals, [20, 30]), 'struct array s(2:end) failed');

% Scalar in binary expression with end
y = 7;
assert(y(end-0) == 7, 'scalar end-0 failed');

disp('SUCCESS');

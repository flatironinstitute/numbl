% Test that indexing a scalar with an empty row range produces correct shape
% In MATLAB, scalar(empty_range, :) should return [0, 1], not [1, 0]

r = 1:0;  % empty range

% Scalar indexed with empty row range and colon column
A = 42;
s = A(r, :);
assert(size(s, 1) == 0, sprintf('Expected 0 rows, got %d', size(s, 1)));
assert(size(s, 2) == 1, sprintf('Expected 1 column, got %d', size(s, 2)));

% Verify concatenation works correctly with the empty result
result = [s ; A(1, :)];
assert(isequal(size(result), [1, 1]));
assert(result == 42);

disp('SUCCESS');

% Test logical matrix indexing on a vector
% In MATLAB, a logical matrix can index a vector when numel matches

rhs = (1:25)';
ii = false(5);
ii(2:4, 2:4) = true;

% Logical matrix indexing with colon second index
result = rhs(ii, :);
assert(isequal(size(result), [9, 1]), 'Should be 9x1');

% The true elements in ii are at linear indices corresponding to
% rows 2-4, cols 2-4 of a 5x5 matrix (column-major)
expected = rhs(find(ii));
assert(isequal(result, expected), 'Values should match find-based indexing');

% Also test without the colon
result2 = rhs(ii);
assert(isequal(result2, expected), 'rhs(ii) should also work');

fprintf('SUCCESS\n');

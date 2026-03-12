% Test that [v, idx] = max/min of scalar returns idx=1

% max of scalar
[v, idx] = max(42);
assert(v == 42);
assert(idx == 1, sprintf('max scalar: expected idx=1, got idx=%g', idx));

% min of scalar
[v, idx] = min(42);
assert(v == 42);
assert(idx == 1, sprintf('min scalar: expected idx=1, got idx=%g', idx));

% max of 1x1 matrix
[v, idx] = max([42]);
assert(v == 42);
assert(idx == 1);

% min of 1x1 matrix
[v, idx] = min([42]);
assert(v == 42);
assert(idx == 1);

% Verify indexing with the returned idx works correctly
A = [10; 20; 30];
tmp = max(A, [], 1);  % returns scalar 30
[val, col] = max(tmp);
assert(val == 30);
assert(col == 1);
B = A(:, col);  % should index column 1
assert(isequal(B, [10; 20; 30]));

disp('SUCCESS');

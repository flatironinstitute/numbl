% Test that unique preserves vector orientation (row vs column)

% Row vector input → row vector output
r = unique([3 1 2 1 3]);
assert(isequal(size(r), [1, 3]), 'unique of row vector should be row');
assert(isequal(r, [1 2 3]));

% Column vector input → column vector output
c = unique([3; 1; 2; 1; 3]);
assert(isequal(size(c), [3, 1]), 'unique of column vector should be column');
assert(isequal(c, [1; 2; 3]));

% Matrix input → column vector output
m = unique([3 1; 2 1; 3 2]);
assert(isequal(size(m), [3, 1]), 'unique of matrix should be column');
assert(isequal(m, [1; 2; 3]));

% Scalar input stays scalar
s = unique(5);
assert(isequal(s, 5));

disp('SUCCESS');

% Test logical indexing in 2D general case (neither index is colon)
% This tests the case where both row and column indices are non-colon
% and at least one is a logical mask.

A = [1 2 3; 4 5 6; 7 8 9];

% Test 1: numeric row range with logical column index (read)
mask_col = logical([1, 0, 1]);
B = A(1:2, mask_col);
assert(isequal(B, [1 3; 4 6]));

% Test 2: logical row index with numeric column range (read)
mask_row = logical([1, 1, 0]);
C = A(mask_row, 2:3);
assert(isequal(C, [2 3; 5 6]));

% Test 3: both logical indices (read)
D = A(mask_row, mask_col);
assert(isequal(D, [1 3; 4 6]));

% Test 4: numeric row range with logical column index (write)
E = zeros(3, 3);
E(1:2, mask_col) = [10 30; 40 60];
assert(E(1,1) == 10);
assert(E(1,2) == 0);
assert(E(1,3) == 30);
assert(E(2,1) == 40);
assert(E(2,2) == 0);
assert(E(2,3) == 60);

% Test 5: scalar result from logical indices
mask_one_row = logical([0, 1, 0]);
mask_one_col = logical([0, 1, 0]);
val = A(mask_one_row, mask_one_col);
assert(val == 5);

% Test 6: end-1 range with logical index (matches the original bug)
vals = [1 2 3; 4 5 6; 7 8 9; 10 11 12];
isSkew = logical([1, 0, 1]);
result = vals(1:end-1, isSkew);
assert(isequal(result, [1 3; 4 6; 7 9]));

disp('SUCCESS')

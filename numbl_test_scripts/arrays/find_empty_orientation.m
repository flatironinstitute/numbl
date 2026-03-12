% Test that find() returns correct empty shape based on input orientation

% Column vector input → empty column [0,1]
M = zeros(5, 1);
result = find(M);
assert(size(result, 1) == 0, 'find(col_zeros) should have 0 rows');
assert(size(result, 2) == 1, 'find(col_zeros) should have 1 col');

% Row vector input → empty row [1,0]
M = zeros(1, 5);
result = find(M);
assert(size(result, 1) == 1, 'find(row_zeros) should have 1 row');
assert(size(result, 2) == 0, 'find(row_zeros) should have 0 cols');

% Matrix input → empty column [0,1]
M = zeros(3, 4);
result = find(M);
assert(size(result, 1) == 0, 'find(matrix_zeros) should have 0 rows');
assert(size(result, 2) == 1, 'find(matrix_zeros) should have 1 col');

% Non-empty column vector
M = [0; 3; 0; 5; 0];
result = find(M);
assert(isequal(result, [2; 4]), 'find(col) should return column');

% Non-empty row vector
M = [0 3 0 5 0];
result = find(M);
assert(isequal(result, [2 4]), 'find(row) should return row');

% Non-empty matrix → column
M = [0 1; 0 0];
result = find(M);
assert(size(result, 2) == 1, 'find(matrix) should return column');

fprintf('SUCCESS\n');

% Test growing an empty matrix via colon-indexed column assignment
% In MATLAB, A(:, k) = v where A is [] grows A to accommodate v.

% Test 1: Assign first column to empty matrix
A = [];
A(:, 1) = [1; 2; 3];
assert(isequal(A, [1; 2; 3]), 'first column assignment to empty');
assert(isequal(size(A), [3 1]), 'size after first column');

% Test 2: Assign second column (grows columns)
A(:, 2) = [4; 5; 6];
assert(isequal(A, [1 4; 2 5; 3 6]), 'second column assignment');
assert(isequal(size(A), [3 2]), 'size after second column');

% Test 3: The chebfun pattern - column and row assignment in a loop
cols = [];
rows = [];
M = [10 20 30; 40 50 60; 70 80 90];
k = 1;
cols(:, k) = M(:, 2);
rows(k, :) = M(1, :);
assert(isequal(cols, [20; 50; 80]), 'cols pattern');
assert(isequal(rows, [10 20 30]), 'rows pattern');

k = 2;
cols(:, k) = M(:, 3);
rows(k, :) = M(3, :);
assert(isequal(cols, [20 30; 50 60; 80 90]), 'cols two columns');
assert(isequal(rows, [10 20 30; 70 80 90]), 'rows two rows');

disp('SUCCESS');

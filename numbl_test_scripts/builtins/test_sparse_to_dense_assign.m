% Test assigning sparse vectors to dense matrix columns/rows
x = zeros(5, 3);
v = sparse([1; 0; 3; 0; 5]);
x(:, 1) = v;
assert(isequal(x(:, 1), [1; 0; 3; 0; 5]));
assert(~issparse(x));

% Assign sparse row to a row of a dense matrix
y = zeros(3, 4);
y(2, :) = sparse([10 20 30 40]);
assert(isequal(y(2, :), [10 20 30 40]));

% Assign sparse column to a range of rows
z = ones(6, 2);
z(2:5, 1) = sparse([7; 8; 9; 10]);
assert(isequal(z(:, 1), [1; 7; 8; 9; 10; 1]));

disp('SUCCESS')

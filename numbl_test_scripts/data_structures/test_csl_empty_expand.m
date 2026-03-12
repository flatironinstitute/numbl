%% Empty cell CSL expansion vanishes in indexing context
ec = cell(1, 0);
x = [1 2; 3 4];
y = x(:, 1, ec{:});
assert(isequal(y, [1; 3]));

%% Non-empty cell CSL adds indices
c = {1};
z = x(:, c{:});
assert(isequal(z, [1; 3]));

disp('SUCCESS')

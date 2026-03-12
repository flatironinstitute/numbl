% Test comma-separated list (CSL) expansion of cell indexing inside []

% Single row cell with column vectors - expand and horzcat
rhs = {[1;2;3], [4;5;6]};
result = [rhs{1,:}];
assert(isequal(size(result), [3, 2]), 'size should be [3,2]');
assert(isequal(result, [1 4; 2 5; 3 6]), 'values should match');

% Single index CSL in concat
c = {10, 20, 30};
result = [c{:}];
assert(isequal(result, [10 20 30]), 'c{:} should expand in []');

% CSL with reshape
rhs2 = {[1;2;3], [4;5;6]};
result2 = reshape([rhs2{1,:}], 3, 2);
assert(isequal(size(result2), [3, 2]));
assert(isequal(result2, [1 4; 2 5; 3 6]));

fprintf('SUCCESS\n');

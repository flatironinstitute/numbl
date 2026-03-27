% Test setdiff preserves column vector orientation
% MATLAB: setdiff of column vectors returns a column vector

a = [1;2;3;4;5];
b = [2;4];

c = setdiff(a, b);
assert(isequal(c, [1;3;5]), 'setdiff: wrong values');
assert(isequal(size(c), [3, 1]), 'setdiff: should be column');

% Row vector input should stay row
c2 = setdiff([1 2 3 4 5], [2 4]);
assert(isequal(size(c2), [1, 3]), 'setdiff row: should be row');

disp('SUCCESS');

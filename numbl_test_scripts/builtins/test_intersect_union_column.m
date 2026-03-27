% Test intersect/union preserve column vector orientation
% MATLAB: set operations on column vectors return column vectors

% intersect with column vectors
c = intersect([3;1;2], [2;4;1]);
assert(isequal(c, [1;2]), 'intersect col: wrong values');
assert(isequal(size(c), [2, 1]), 'intersect col: should be column');

% intersect with row vectors stays row
c2 = intersect([3 1 2], [2 4 1]);
assert(isequal(size(c2), [1, 2]), 'intersect row: should be row');

% union with column vectors
u = union([3;1], [2;4]);
assert(isequal(u, [1;2;3;4]), 'union col: wrong values');
assert(isequal(size(u), [4, 1]), 'union col: should be column');

% union with row vectors stays row
u2 = union([3 1], [2 4]);
assert(isequal(size(u2), [1, 4]), 'union row: should be row');

disp('SUCCESS');

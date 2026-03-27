% Test conv preserves column vector orientation
% MATLAB: conv of column vectors returns a column vector

a = [1;2;3];
b = [1;1];

% Full
c = conv(a, b);
assert(isequal(size(c), [4, 1]), 'conv full: should be column');
assert(isequal(c, [1;3;5;3]), 'conv full: wrong values');

% Same
c = conv(a, b, 'same');
assert(isequal(size(c), [3, 1]), 'conv same: should be column');
assert(isequal(c, [3;5;3]), 'conv same: wrong values');

% Valid
c = conv(a, b, 'valid');
assert(isequal(size(c), [2, 1]), 'conv valid: should be column');
assert(isequal(c, [3;5]), 'conv valid: wrong values');

% Row vectors should still be row vectors
a2 = [1 2 3];
b2 = [1 1];
c2 = conv(a2, b2);
assert(isequal(size(c2), [1, 4]), 'conv row: should be row');

disp('SUCCESS');

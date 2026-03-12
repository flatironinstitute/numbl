% Test that trailing singleton dimensions are dropped (minimum 2D)
% MATLAB always drops trailing singleton dimensions from tensor shapes.

% reshape with trailing singleton
x = [1 2 3 4 5 6];
y = reshape(x, [2 3 1]);
assert(ndims(y) == 2, 'reshape [2 3 1]: expected ndims=2');
assert(isequal(size(y), [2 3]), 'reshape [2 3 1]: expected size [2 3]');

y2 = reshape(x, [1 6 1]);
assert(ndims(y2) == 2, 'reshape [1 6 1]: expected ndims=2');
assert(isequal(size(y2), [1 6]), 'reshape [1 6 1]: expected size [1 6]');

y3 = reshape(x, [1 2 3 1]);
assert(ndims(y3) == 3, 'reshape [1 2 3 1]: expected ndims=3');
assert(isequal(size(y3), [1 2 3]), 'reshape [1 2 3 1]: expected size [1 2 3]');

% Non-trailing singleton should be preserved
y4 = reshape(x, [1 2 3]);
assert(ndims(y4) == 3, 'reshape [1 2 3]: expected ndims=3');
assert(isequal(size(y4), [1 2 3]), 'reshape [1 2 3]: expected size [1 2 3]');

% zeros/ones/rand with trailing singletons
z = zeros(2, 3, 1);
assert(ndims(z) == 2, 'zeros(2,3,1): expected ndims=2');
assert(isequal(size(z), [2 3]), 'zeros(2,3,1): expected size [2 3]');

o = ones(3, 1, 1);
assert(ndims(o) == 2, 'ones(3,1,1): expected ndims=2');
assert(isequal(size(o), [3 1]), 'ones(3,1,1): expected size [3 1]');

% repmat preserving trailing singletons
a = [1 2 3];
b = repmat(a, 1, 1, 1);
assert(ndims(b) == 2, 'repmat trailing: expected ndims=2');
assert(isequal(size(b), [1 3]), 'repmat trailing: expected size [1 3]');

% Operations on results should not gain trailing singletons
c = reshape(x, [2 3 1]);
d = c + 1;
assert(ndims(d) == 2, 'add after reshape: expected ndims=2');

% Multiple trailing singletons
y5 = reshape(x, [2 3 1 1 1]);
assert(ndims(y5) == 2, 'reshape [2 3 1 1 1]: expected ndims=2');
assert(isequal(size(y5), [2 3]), 'reshape [2 3 1 1 1]: expected size [2 3]');

disp('SUCCESS');

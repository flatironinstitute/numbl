% Test that obj.DepProp(args) on a Dependent property with a get.XXX
% accessor means: call the getter (no args), then index the result with
% `args`.  Previously numbl treated this as a method call named XXX, so
% obj.r(:,:) on a class with `get.r` failed.

B = reshape(1:24, 2, 3, 4);
o = DepPropTensor_(B);

% Simple getter chain
v2d = o.view;
assert(isequal(size(v2d), [2 3]), 'view size');
assert(v2d(2, 3) == 6, 'view value');

% view(i, j) — call getter, then scalar index
assert(o.view(1, 1) == 1, 'view(1,1)');
assert(o.view(2, 3) == 6, 'view(2,3)');

% view(:) — call getter, then colon linearization
lin = o.view(:);
assert(isequal(size(lin), [6 1]), 'view(:) shape');
assert(lin(4) == 4, 'view(:) element');

% 3D getter — chaining with collapsed trailing-dim indexing
assert(isequal(size(o.view3d), [2 3 4]), 'view3d size');
flat = o.view3d(:, :);
assert(isequal(size(flat), [2 12]), 'view3d(:,:) shape');
assert(flat(1, 5) == 9, 'view3d(:,:) element');

% Scalar index into dependent 3D property with collapsed col
assert(o.view3d(2, 6) == 12, 'view3d(2,6) scalar');

% Multiple chained gets followed by a scalar
% view = B(:,:,1) which is [1 3 5; 2 4 6] column-major, so view(1,2)=3, view(2,1)=2
assert(o.view(1, 2) + o.view(2, 1) == 5, 'arithmetic chain');

disp('SUCCESS');

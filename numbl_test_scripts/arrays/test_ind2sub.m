% Test ind2sub builtin

% 2D matrix: 3x4
[r, c] = ind2sub([3, 4], 1);
assert(r == 1 && c == 1);

[r, c] = ind2sub([3, 4], 3);
assert(r == 3 && c == 1);

[r, c] = ind2sub([3, 4], 4);
assert(r == 1 && c == 2);

[r, c] = ind2sub([3, 4], 12);
assert(r == 3 && c == 4);

% Vector of indices
[r, c] = ind2sub([3, 4], [1 4 7 10]);
assert(isequal(r, [1 1 1 1]));
assert(isequal(c, [1 2 3 4]));

% 3D array
[i1, i2, i3] = ind2sub([2, 3, 4], 1);
assert(i1 == 1 && i2 == 1 && i3 == 1);

[i1, i2, i3] = ind2sub([2, 3, 4], 7);
assert(i1 == 1 && i2 == 1 && i3 == 2);

% Round-trip with sub2ind
sz = [3, 4, 2];
idx = sub2ind(sz, 2, 3, 1);
[a, b, c] = ind2sub(sz, idx);
assert(a == 2 && b == 3 && c == 1);

fprintf('SUCCESS\n');

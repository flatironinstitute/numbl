% Test max/min with vector of dimensions
A = reshape(1:24, [2,3,4]);

% max over dims [2,3] should give max over all but dim 1
r1 = max(A, [], [2,3]);
assert(isequal(size(r1), [2, 1]));
assert(r1(1) == 23);
assert(r1(2) == 24);

% min over dims [2,3]
r2 = min(A, [], [2,3]);
assert(isequal(size(r2), [2, 1]));
assert(r2(1) == 1);
assert(r2(2) == 2);

% max over dims [1,2]
r3 = max(A, [], [1,2]);
assert(isequal(size(r3), [1, 1, 4]));
assert(r3(1) == 6);
assert(r3(4) == 24);

% squeeze + max multi-dim
r4 = squeeze(max(A, [], [1,3]));
assert(isequal(size(r4), [1, 3]));
assert(r4(1) == 20);
assert(r4(2) == 22);
assert(r4(3) == 24);

fprintf('SUCCESS\n');

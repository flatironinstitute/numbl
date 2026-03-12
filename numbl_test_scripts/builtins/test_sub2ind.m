% Test sub2ind builtin
% Basic 2D
assert(sub2ind([3 4], 2, 3) == 8);
assert(sub2ind([3 4], 1, 1) == 1);
assert(sub2ind([3 4], 3, 4) == 12);

% Vector inputs
r = sub2ind([3 4], [1 2], [3 4]);
assert(r(1) == 7);
assert(r(2) == 11);

% Column vector inputs
r2 = sub2ind([3 4], [1; 2], [3; 4]);
assert(r2(1) == 7);
assert(r2(2) == 11);
assert(size(r2, 1) == 2);
assert(size(r2, 2) == 1);

fprintf('SUCCESS\n');

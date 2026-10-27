% true/false accept N-D sizes, size vectors and negative sizes (clamped to 0)
assert(isequal(size(true(2,3,4)), [2 3 4]));
assert(isequal(size(false(2,3,4)), [2 3 4]));
assert(all(true(2,3,4), 'all'));
assert(~any(false(2,3,4), 'all'));
assert(isequal(size(true([2 3])), [2 3]));
assert(isequal(size(false(3)), [3 3]));
assert(isequal(size(false(-1, 2)), [0 2]));
assert(islogical(true(2)) && islogical(false(2,2,2)));
disp('SUCCESS')

% Row vectors and basic operations

v = [1, 2, 3, 4, 5];
assert(length(v) == 5)
assert(v(1) == 1)
assert(v(3) == 3)
assert(v(5) == 5)

% Sum and basic stats
assert(sum(v) == 15)
assert(min(v) == 1)
assert(max(v) == 5)

% Element-wise arithmetic
v2 = v * 2;
assert(v2(1) == 2)
assert(v2(3) == 6)
assert(v2(5) == 10)

% Column vector
col = [1; 2; 3];
assert(size(col, 1) == 3)
assert(size(col, 2) == 1)

disp('SUCCESS')

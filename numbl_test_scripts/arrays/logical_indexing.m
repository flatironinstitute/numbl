% Logical indexing and comparisons

v = [1, 5, 3, 8, 2, 7];

% Logical comparisons produce logical arrays
mask = v > 4;
% Elements greater than 4: 5, 8, 7

% find
idx = find(v > 4);
assert(length(idx) == 3)
assert(idx(1) == 2)
assert(idx(2) == 4)
assert(idx(3) == 6)

% any / all
assert(any(v > 4))
assert(~all(v > 4))
assert(all(v > 0))

% Logical operators
a = 1; b = 0;
assert(a && b == 0)
assert(a || b == 1)
assert(~b == 1)

% Comparison operators
assert((3 ~= 4) == 1)
assert((3 == 3) == 1)
assert((3 < 4) == 1)
assert((4 > 3) == 1)
assert((3 <= 3) == 1)
assert((4 >= 4) == 1)

disp('SUCCESS')

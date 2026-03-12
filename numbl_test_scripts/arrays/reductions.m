% Array reduction functions

v = [3, 1, 4, 1, 5, 9, 2, 6];

assert(sum(v) == 31)
assert(min(v) == 1)
assert(max(v) == 9)
assert(length(v) == 8)
assert(numel(v) == 8)

% mean
assert(abs(mean(v) - 31/8) < 1e-5)

% prod
p = prod([1, 2, 3, 4]);
assert(p == 24)

% cumsum
cs = cumsum([1, 2, 3, 4]);
assert(cs(1) == 1)
assert(cs(2) == 3)
assert(cs(3) == 6)
assert(cs(4) == 10)

% sort
s = sort([3, 1, 4, 1, 5]);
assert(s(1) == 1)
assert(s(2) == 1)
assert(s(5) == 5)

% sort descending
sd = sort([3, 1, 4, 1, 5], 'descend');
assert(sd(1) == 5)
assert(sd(5) == 1)

% unique
u = unique([3, 1, 4, 1, 5, 3]);
assert(length(u) == 4)
assert(u(1) == 1)
assert(u(4) == 5)

% Column-wise operations on matrix
A = [1, 2; 3, 4; 5, 6];
s2 = sum(A);
assert(s2(1) == 9)
assert(s2(2) == 12)

disp('SUCCESS')

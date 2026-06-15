% Test dot(A, B, dim): dot product along a given dimension

A = [1 2; 3 4];
B = [5 6; 7 8];

% dim 1 (down columns) matches the default 2-arg behavior
r1 = dot(A, B, 1);
assert(isequal(r1, [26 44]), 'dot along dim 1');

% dim 2 (across rows)
% Row 1: 1*5 + 2*6 = 17
% Row 2: 3*7 + 4*8 = 53
r2 = dot(A, B, 2);
assert(isequal(r2, [17; 53]), 'dot along dim 2');

% Row vectors collapsed along dim 2 give a scalar
r3 = dot([1 2 3], [4 5 6], 2);
assert(r3 == 32, 'dot row vectors along dim 2');

% dim beyond the array rank is a no-op reduction: element-wise product
r4 = dot(A, B, 3);
assert(isequal(r4, A .* B), 'dot along singleton dim is element-wise');

% Complex dot conjugates the first argument
a = [1+2i, 3-1i];
b = [2-1i, 1+1i];
% conj(a).*b summed: (1-2i)(2-1i) + (3+1i)(1+1i) = (0-5i) + (2+4i) = 2 - 1i
r5 = dot(a, b, 2);
assert(abs(r5 - (2 - 1i)) < 1e-12, 'complex dot along dim 2 conjugates first arg');

disp('SUCCESS');

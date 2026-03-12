% Test speye builtin (sparse identity, treated as dense identity in numbl)

% speye() -> scalar 1
s = speye();
assert(s == 1);

% speye(n) -> n-by-n identity
A = speye(3);
assert(isequal(size(A), [3, 3]));
assert(isequal(A, eye(3)));

% speye(n,m) with n < m
B = speye(2, 4);
assert(isequal(size(B), [2, 4]));
assert(B(1,1) == 1);
assert(B(2,2) == 1);
assert(B(1,2) == 0);
assert(B(2,3) == 0);

% speye(n,m) with n > m
C = speye(4, 2);
assert(isequal(size(C), [4, 2]));
assert(C(1,1) == 1);
assert(C(2,2) == 1);
assert(C(3,1) == 0);
assert(C(3,2) == 0);

% speye(sz) with size vector
D = speye([3, 4]);
assert(isequal(size(D), [3, 4]));
assert(isequal(D, speye(3, 4)));

disp('SUCCESS');

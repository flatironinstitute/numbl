% Test speye builtin (sparse identity)

% speye(n) -> n-by-n sparse identity
A = speye(3);
assert(issparse(A));
assert(isequal(size(A), [3, 3]));
assert(isequal(full(A), eye(3)));

% speye(n,m) with n < m
B = speye(2, 4);
assert(issparse(B));
assert(isequal(size(B), [2, 4]));
assert(isequal(full(B), eye(2, 4)));

% speye(n,m) with n > m
C = speye(4, 2);
assert(issparse(C));
assert(isequal(size(C), [4, 2]));
assert(isequal(full(C), eye(4, 2)));

% speye(sz) with size vector
D = speye([3, 4]);
assert(issparse(D));
assert(isequal(size(D), [3, 4]));
assert(isequal(full(D), full(speye(3, 4))));

disp('SUCCESS');

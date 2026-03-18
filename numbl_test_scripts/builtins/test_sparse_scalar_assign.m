% Test that assigning a 1x1 sparse matrix to a dense array element works
tmp = zeros(5, 1);
x = sparse(1, 1, 3.14, 1, 1);
tmp(1) = x;
assert(tmp(1) == 3.14);
assert(~issparse(tmp));

% Also test with row vector target
tmp2 = zeros(1, 5);
tmp2(3) = sparse(1, 1, 2.0, 1, 1);
assert(tmp2(3) == 2.0);

% Assignment to matrix element
M = ones(3, 3);
M(2, 2) = sparse(1, 1, 7.0, 1, 1);
assert(M(2, 2) == 7.0);

disp('SUCCESS')

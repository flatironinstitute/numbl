% Test diag with offset argument (k parameter)
% D = diag(v,k) places elements of vector v on the kth diagonal
% k=0 main diagonal, k>0 above, k<0 below
% x = diag(A,k) returns column vector of elements on kth diagonal of A

% Superdiagonal (k=1)
v = [1; 2; 3];
A = diag(v, 1);
assert(isequal(size(A), [4, 4]));
expected = [0 1 0 0; 0 0 2 0; 0 0 0 3; 0 0 0 0];
assert(isequal(A, expected));

% Subdiagonal (k=-1)
B = diag(v, -1);
assert(isequal(size(B), [4, 4]));
expected2 = [0 0 0 0; 1 0 0 0; 0 2 0 0; 0 0 3 0];
assert(isequal(B, expected2));

% Main diagonal (k=0) should be same as diag(v)
C = diag(v, 0);
assert(isequal(C, diag(v)));

% Larger offset (k=2)
v2 = [5; 6];
D = diag(v2, 2);
assert(isequal(size(D), [4, 4]));
assert(D(1,3) == 5);
assert(D(2,4) == 6);

% Negative larger offset (k=-2)
E = diag(v2, -2);
assert(isequal(size(E), [4, 4]));
assert(E(3,1) == 5);
assert(E(4,2) == 6);

% Extract off-diagonal from matrix
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
d1 = diag(M, 1);
assert(isequal(d1, [2; 7; 12]));

d_neg1 = diag(M, -1);
assert(isequal(d_neg1, [5; 10; 15]));

d2 = diag(M, 2);
assert(isequal(d2, [3; 8]));

% The colleague matrix pattern from chebfun roots.m
oh = 0.5 * ones(3, 1);
A2 = diag(oh, 1) + diag(oh, -1);
assert(isequal(size(A2), [4, 4]));
assert(A2(1,2) == 0.5);
assert(A2(2,1) == 0.5);
assert(A2(1,1) == 0);

disp('SUCCESS');

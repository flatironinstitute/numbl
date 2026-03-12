% Test blkdiag - block diagonal matrix

A1 = ones(2,2);
A2 = 2*ones(3,2);
A3 = 3*ones(2,3);
B = blkdiag(A1, A2, A3);
assert(size(B,1) == 7);
assert(size(B,2) == 7);
assert(B(1,1) == 1);
assert(B(2,2) == 1);
assert(B(3,3) == 2);
assert(B(5,4) == 2);
assert(B(6,5) == 3);
assert(B(7,7) == 3);
assert(B(1,3) == 0);
assert(B(3,1) == 0);
assert(B(6,1) == 0);

% Single matrix
C = blkdiag([1 2; 3 4]);
assert(isequal(C, [1 2; 3 4]));

% Scalars
D = blkdiag(1, 2, 3);
assert(isequal(D, diag([1 2 3])));

% Mix of scalar and matrix
E = blkdiag(5, [1 2; 3 4]);
assert(size(E,1) == 3);
assert(size(E,2) == 3);
assert(E(1,1) == 5);
assert(E(2,2) == 1);
assert(E(2,3) == 2);
assert(E(3,2) == 3);
assert(E(3,3) == 4);
assert(E(1,2) == 0);
assert(E(2,1) == 0);

disp('SUCCESS');

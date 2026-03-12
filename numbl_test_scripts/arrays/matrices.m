% 2D matrices

A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
assert(size(A, 1) == 3)
assert(size(A, 2) == 3)
assert(A(1,1) == 1)
assert(A(2,2) == 5)
assert(A(3,3) == 9)
assert(A(1,3) == 3)
assert(A(3,1) == 7)

% Transpose
B = A';
assert(B(1,2) == 4)
assert(B(2,1) == 2)

% zeros and ones
Z = zeros(2, 3);
assert(size(Z, 1) == 2)
assert(size(Z, 2) == 3)
assert(Z(1,1) == 0)

O = ones(3, 2);
assert(O(2,2) == 1)

% eye
I = eye(3);
assert(I(1,1) == 1)
assert(I(1,2) == 0)
assert(I(2,2) == 1)

disp('SUCCESS')

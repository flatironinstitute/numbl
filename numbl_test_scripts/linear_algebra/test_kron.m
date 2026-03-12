% Kronecker tensor product

% 2x2 kron 2x2
A = [1 2; 3 4];
B = [5 6; 7 8];
K = kron(A, B);
assert(size(K, 1) == 4)
assert(size(K, 2) == 4)
% K should be:
%  1*[5 6; 7 8]  2*[5 6; 7 8]
%  3*[5 6; 7 8]  4*[5 6; 7 8]
% = [ 5  6 10 12;
%     7  8 14 16;
%    15 18 20 24;
%    21 24 28 32]
assert(K(1,1) == 5)
assert(K(1,2) == 6)
assert(K(1,3) == 10)
assert(K(1,4) == 12)
assert(K(2,1) == 7)
assert(K(2,2) == 8)
assert(K(2,3) == 14)
assert(K(2,4) == 16)
assert(K(3,1) == 15)
assert(K(3,2) == 18)
assert(K(3,3) == 20)
assert(K(3,4) == 24)
assert(K(4,1) == 21)
assert(K(4,2) == 24)
assert(K(4,3) == 28)
assert(K(4,4) == 32)

% kron with identity
I2 = eye(2);
K2 = kron(I2, A);
assert(size(K2, 1) == 4)
assert(size(K2, 2) == 4)
assert(K2(1,1) == 1)
assert(K2(1,2) == 2)
assert(K2(1,3) == 0)
assert(K2(2,1) == 3)
assert(K2(2,2) == 4)
assert(K2(3,3) == 1)
assert(K2(3,4) == 2)
assert(K2(4,3) == 3)
assert(K2(4,4) == 4)

% Non-square: 2x3 kron 3x2
C = [1 2 3; 4 5 6];
D = [7 8; 9 10; 11 12];
K3 = kron(C, D);
assert(size(K3, 1) == 6)
assert(size(K3, 2) == 6)
assert(K3(1,1) == 7)
assert(K3(1,2) == 8)
assert(K3(1,3) == 14)
assert(K3(3,1) == 11)
assert(K3(3,2) == 12)
assert(K3(4,5) == 6 * 7)
assert(K3(6,6) == 6 * 12)

% Scalar kron
K4 = kron(3, [1 2; 3 4]);
assert(K4(1,1) == 3)
assert(K4(1,2) == 6)
assert(K4(2,1) == 9)
assert(K4(2,2) == 12)

K5 = kron([1 2; 3 4], 2);
assert(K5(1,1) == 2)
assert(K5(1,2) == 4)
assert(K5(2,1) == 6)
assert(K5(2,2) == 8)

disp('SUCCESS')

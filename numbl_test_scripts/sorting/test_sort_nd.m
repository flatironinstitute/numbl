% Test sort with dim argument on n-dimensional arrays.
% A = reshape(24:-1:1, [2,3,4]) gives pages:
%   A(:,:,1) = [24 22 20; 23 21 19]
%   A(:,:,2) = [18 16 14; 17 15 13]
%   A(:,:,3) = [12 10  8; 11  9  7]
%   A(:,:,4) = [ 6  4  2;  5  3  1]

A = reshape(24:-1:1, [2, 3, 4]);

%% sort(A, 1): sort each column of each page along dim 1
B = sort(A, 1);
assert(isequal(size(B), [2, 3, 4]));
% Page 1, col 1: sort [24;23] → [23;24]
assert(B(1,1,1) == 23);
assert(B(2,1,1) == 24);
% Page 1, col 2: sort [22;21] → [21;22]
assert(B(1,2,1) == 21);
assert(B(2,2,1) == 22);
% Page 4, col 3: sort [2;1] → [1;2]
assert(B(1,3,4) == 1);
assert(B(2,3,4) == 2);

%% sort(A, 2): sort each row of each page along dim 2
C = sort(A, 2);
assert(isequal(size(C), [2, 3, 4]));
% Page 1, row 1: sort [24,22,20] → [20,22,24]
assert(C(1,1,1) == 20);
assert(C(1,2,1) == 22);
assert(C(1,3,1) == 24);
% Page 1, row 2: sort [23,21,19] → [19,21,23]
assert(C(2,1,1) == 19);
assert(C(2,2,1) == 21);
assert(C(2,3,1) == 23);
% Page 3, row 1: sort [12,10,8] → [8,10,12]
assert(C(1,1,3) == 8);
assert(C(1,2,3) == 10);
assert(C(1,3,3) == 12);

%% sort(A, 3): sort each (i,j) fiber along dim 3
D = sort(A, 3);
assert(isequal(size(D), [2, 3, 4]));
% (1,1,:): sort [24,18,12,6] → [6,12,18,24]
assert(D(1,1,1) == 6);
assert(D(1,1,2) == 12);
assert(D(1,1,3) == 18);
assert(D(1,1,4) == 24);
% (2,3,:): sort [19,13,7,1] → [1,7,13,19]
assert(D(2,3,1) == 1);
assert(D(2,3,2) == 7);
assert(D(2,3,3) == 13);
assert(D(2,3,4) == 19);

%% sort with 'descend' and dim
E = sort(A, 2, 'descend');
assert(isequal(size(E), [2, 3, 4]));
% Page 1, row 1: sort [24,22,20] descend → [24,22,20] (already descending)
assert(E(1,1,1) == 24);
assert(E(1,2,1) == 22);
assert(E(1,3,1) == 20);
% Page 4, row 2: sort [5,3,1] descend → [5,3,1]
assert(E(2,1,4) == 5);
assert(E(2,2,4) == 3);
assert(E(2,3,4) == 1);

%% [B,I] = sort(A, dim): verify indices
[F, I] = sort(A, 2);
assert(isequal(size(F), [2, 3, 4]));
assert(isequal(size(I), [2, 3, 4]));
% Page 1, row 1: [24,22,20] sorted → [20,22,24] with indices [3,2,1]
assert(I(1,1,1) == 3);
assert(I(1,2,1) == 2);
assert(I(1,3,1) == 1);
% Verify F(r,j,k) == A(r, I(r,j,k), k)
for k = 1:4
    for r = 1:2
        for j = 1:3
            assert(F(r,j,k) == A(r, I(r,j,k), k));
        end
    end
end

%% sort(A, dim) on 2D matrix (regression)
M = [3 1 4; 1 5 9; 2 6 5];
% sort along dim 2: sort each row
N = sort(M, 2);
assert(N(1,1) == 1); assert(N(1,2) == 3); assert(N(1,3) == 4);
assert(N(2,1) == 1); assert(N(2,2) == 5); assert(N(2,3) == 9);
assert(N(3,1) == 2); assert(N(3,2) == 5); assert(N(3,3) == 6);

disp('SUCCESS');

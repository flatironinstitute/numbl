% Test cumsum, cumprod, cummin, cummax on n-dimensional arrays
% Verifies dim argument works correctly for 3D arrays

A = reshape(1:24, [2, 3, 4]);
% A(:,:,1) = [1 3 5; 2 4 6]
% A(:,:,2) = [7 9 11; 8 10 12]
% A(:,:,3) = [13 15 17; 14 16 18]
% A(:,:,4) = [19 21 23; 20 22 24]

%% cumsum along dim 1 (default for non-row-vector)
B = cumsum(A);
assert(isequal(size(B), [2, 3, 4]));
assert(B(1,1,1) == 1);
assert(B(2,1,1) == 3);   % 1+2
assert(B(1,2,1) == 3);
assert(B(2,2,1) == 7);   % 3+4

%% cumsum along dim 2 (explicit)
C = cumsum(A, 2);
assert(isequal(size(C), [2, 3, 4]));
% Row 1, page 1: cumsum([1 3 5]) = [1 4 9]
assert(C(1,1,1) == 1);
assert(C(1,2,1) == 4);
assert(C(1,3,1) == 9);
% Row 2, page 1: cumsum([2 4 6]) = [2 6 12]
assert(C(2,1,1) == 2);
assert(C(2,2,1) == 6);
assert(C(2,3,1) == 12);
% Row 1, page 2: cumsum([7 9 11]) = [7 16 27]
assert(C(1,1,2) == 7);
assert(C(1,2,2) == 16);
assert(C(1,3,2) == 27);

%% cumsum along dim 3 (explicit)
D = cumsum(A, 3);
assert(isequal(size(D), [2, 3, 4]));
% (1,1,:): cumsum([1 7 13 19]) = [1 8 21 40]
assert(D(1,1,1) == 1);
assert(D(1,1,2) == 8);
assert(D(1,1,3) == 21);
assert(D(1,1,4) == 40);
% (2,2,:): cumsum([4 10 16 22]) = [4 14 30 52]
assert(D(2,2,1) == 4);
assert(D(2,2,2) == 14);
assert(D(2,2,3) == 30);
assert(D(2,2,4) == 52);

%% cumprod along dim 2
E = cumprod(A, 2);
assert(isequal(size(E), [2, 3, 4]));
% Row 1, page 1: cumprod([1 3 5]) = [1 3 15]
assert(E(1,1,1) == 1);
assert(E(1,2,1) == 3);
assert(E(1,3,1) == 15);
% Row 2, page 1: cumprod([2 4 6]) = [2 8 48]
assert(E(2,1,1) == 2);
assert(E(2,2,1) == 8);
assert(E(2,3,1) == 48);

%% cummax along dim 2
% Use a non-monotonic array to make cummax interesting
G = reshape([5 1 3 6 2 4 11 7 9 12 8 10 17 13 15 18 14 16 23 19 21 24 20 22], [2, 3, 4]);
% G(:,:,1) = [5 3 2; 1 6 4]
H = cummax(G, 2);
assert(isequal(size(H), [2, 3, 4]));
% Row 1, page 1: cummax([5 3 2]) = [5 5 5]
assert(H(1,1,1) == 5);
assert(H(1,2,1) == 5);
assert(H(1,3,1) == 5);
% Row 2, page 1: cummax([1 6 4]) = [1 6 6]
assert(H(2,1,1) == 1);
assert(H(2,2,1) == 6);
assert(H(2,3,1) == 6);

%% cummin along dim 2
I = cummin(G, 2);
assert(isequal(size(I), [2, 3, 4]));
% Row 1, page 1: cummin([5 3 2]) = [5 3 2]
assert(I(1,1,1) == 5);
assert(I(1,2,1) == 3);
assert(I(1,3,1) == 2);
% Row 2, page 1: cummin([1 6 4]) = [1 1 1]
assert(I(2,1,1) == 1);
assert(I(2,2,1) == 1);
assert(I(2,3,1) == 1);

%% cumsum along dim 2 for 2D matrix (regression)
M = [1 2 3; 4 5 6];
F = cumsum(M, 2);
assert(isequal(size(F), [2, 3]));
assert(F(1,1) == 1);
assert(F(1,2) == 3);
assert(F(1,3) == 6);
assert(F(2,1) == 4);
assert(F(2,2) == 9);
assert(F(2,3) == 15);

disp('SUCCESS');

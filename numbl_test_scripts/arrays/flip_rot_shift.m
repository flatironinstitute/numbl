% Test array manipulation: flip, rot90, circshift, repelem

%% flip - reverse along a dimension
% Row vector (default: first non-singleton dim = dim 2)
f = flip([1, 2, 3, 4]);
assert(f(1) == 4);
assert(f(2) == 3);
assert(f(3) == 2);
assert(f(4) == 1);

% Column vector (default dim = 1)
f2 = flip([1; 2; 3]);
assert(f2(1) == 3);
assert(f2(2) == 2);
assert(f2(3) == 1);

% Matrix flip along dim 1 (flip rows)
M = [1, 2; 3, 4; 5, 6];
f3 = flip(M, 1);
assert(f3(1,1) == 5);
assert(f3(2,1) == 3);
assert(f3(3,1) == 1);

% Matrix flip along dim 2 (flip columns)
f4 = flip(M, 2);
assert(f4(1,1) == 2);
assert(f4(1,2) == 1);

% Scalar
assert(flip(7) == 7);

%% rot90 - rotate matrix 90 degrees counter-clockwise
A = [1, 2; 3, 4];
R = rot90(A);
% rot90 CCW: [2,4;1,3]
assert(R(1,1) == 2);
assert(R(1,2) == 4);
assert(R(2,1) == 1);
assert(R(2,2) == 3);

% rot90 twice = 180 degree rotation
R2 = rot90(A, 2);
assert(R2(1,1) == 4);
assert(R2(1,2) == 3);
assert(R2(2,1) == 2);
assert(R2(2,2) == 1);

% rot90 three times = 270 degrees (= -90)
R3 = rot90(A, 3);
assert(R3(1,1) == 3);
assert(R3(1,2) == 1);
assert(R3(2,1) == 4);
assert(R3(2,2) == 2);

% rot90 four times = identity
R4 = rot90(A, 4);
assert(isequal(R4, A));

% Non-square matrix
B = [1, 2, 3; 4, 5, 6];
RB = rot90(B);
assert(size(RB, 1) == 3);
assert(size(RB, 2) == 2);
assert(RB(1,1) == 3);
assert(RB(1,2) == 6);

%% circshift - circular shift
% Row vector
c = circshift([1, 2, 3, 4, 5], 2);
assert(c(1) == 4);
assert(c(2) == 5);
assert(c(3) == 1);
assert(c(4) == 2);
assert(c(5) == 3);

% Negative shift
c2 = circshift([1, 2, 3, 4, 5], -1);
assert(c2(1) == 2);
assert(c2(2) == 3);
assert(c2(5) == 1);

% Column vector
c3 = circshift([1; 2; 3], 1);
assert(c3(1) == 3);
assert(c3(2) == 1);
assert(c3(3) == 2);

% Matrix with scalar shift (shifts along dim 1)
M2 = [1, 2; 3, 4; 5, 6];
c4 = circshift(M2, 1);
assert(c4(1,1) == 5);
assert(c4(2,1) == 1);
assert(c4(3,1) == 3);
assert(c4(1,2) == 6);

%% repelem - repeat elements
% Row vector: repelem(v, n) repeats each element n times
r = repelem([1, 2, 3], 2);
assert(length(r) == 6);
assert(r(1) == 1);
assert(r(2) == 1);
assert(r(3) == 2);
assert(r(4) == 2);
assert(r(5) == 3);
assert(r(6) == 3);

% Scalar repetition
r2 = repelem(5, 3);
assert(length(r2) == 3);
assert(r2(1) == 5);
assert(r2(3) == 5);

% Matrix: repelem(M, r, c) repeats each element r times vertically, c times horizontally
M3 = [1, 2; 3, 4];
r3 = repelem(M3, 2, 3);
assert(size(r3, 1) == 4);
assert(size(r3, 2) == 6);
assert(r3(1,1) == 1);
assert(r3(2,1) == 1);
assert(r3(1,3) == 1);
assert(r3(3,4) == 4);

disp('SUCCESS');

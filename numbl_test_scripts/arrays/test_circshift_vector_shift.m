% Test circshift with vector shift argument for 2D matrices
% Bug: circshift(M, [row_shift, col_shift]) errors with
% "Cannot convert non-scalar tensor to number"

% Shift columns only
M = [1 2 3; 4 5 6];
M2 = circshift(M, [0, 1]);
assert(M2(1,1) == 3);
assert(M2(1,2) == 1);
assert(M2(1,3) == 2);
assert(M2(2,1) == 6);
assert(M2(2,2) == 4);

% Shift rows only
M3 = circshift(M, [1, 0]);
assert(M3(1,1) == 4);
assert(M3(2,1) == 1);

% Shift both rows and columns
M4 = circshift(M, [1, 2]);
assert(M4(1,1) == 5);
assert(M4(1,2) == 6);
assert(M4(1,3) == 4);
assert(M4(2,1) == 2);

% Negative shifts
M5 = circshift(M, [0, -1]);
assert(M5(1,1) == 2);
assert(M5(1,2) == 3);
assert(M5(1,3) == 1);

disp('SUCCESS');

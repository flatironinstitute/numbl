% Test dot product on matrices (column-wise operation)

% dot on vectors should return scalar
r1 = dot([1 2 3], [4 5 6]);
assert(r1 == 32, 'dot on row vectors');

r2 = dot([1; 2; 3], [4; 5; 6]);
assert(r2 == 32, 'dot on col vectors');

% dot on matrices should operate column-wise
A = [1 2; 3 4];
B = [5 6; 7 8];
r3 = dot(A, B);
% Column 1: dot([1;3],[5;7]) = 1*5 + 3*7 = 26
% Column 2: dot([2;4],[6;8]) = 2*6 + 4*8 = 44
assert(isequal(r3, [26 44]), 'dot on matrices should be column-wise');

% 3-column matrix
C = [1 2 3; 4 5 6];
D = [7 8 9; 10 11 12];
r4 = dot(C, D);
% Col 1: 1*7 + 4*10 = 47
% Col 2: 2*8 + 5*11 = 71
% Col 3: 3*9 + 6*12 = 99
assert(isequal(r4, [47 71 99]), 'dot on 2x3 matrices');

% Result of dot on matrices should support element-wise operations
r5 = dot(A, B) > 30;
assert(isequal(r5, [false true]), 'dot(A,B) > 30');

disp('SUCCESS');

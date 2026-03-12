%% Test cell2mat - basic 2D numeric
C1 = {[1], [2 3 4]; [5; 9], [6 7 8; 10 11 12]};
A1 = cell2mat(C1);
assert(isequal(A1, [1 2 3 4; 5 6 7 8; 9 10 11 12]));

%% Test cell2mat - single cell
C2 = {[1 2; 3 4]};
A2 = cell2mat(C2);
assert(isequal(A2, [1 2; 3 4]));

%% Test cell2mat - scalar cells
C3 = {1, 2; 3, 4};
A3 = cell2mat(C3);
assert(isequal(A3, [1 2; 3 4]));

%% Test cell2mat - row of cells
C4 = {[1 2], [3 4], [5 6]};
A4 = cell2mat(C4);
assert(isequal(A4, [1 2 3 4 5 6]));

%% Test cell2mat - column of cells
C5 = {[1; 2]; [3; 4]; [5; 6]};
A5 = cell2mat(C5);
assert(isequal(A5, [1; 2; 3; 4; 5; 6]));

%% Test cell2mat - 1x1 cell with scalar
C6 = {42};
A6 = cell2mat(C6);
assert(A6 == 42);

%% Test cell2mat - mixed row/column sizes
C7 = {[1 2; 3 4], [5; 6]; [7 8], [9]};
A7 = cell2mat(C7);
assert(isequal(A7, [1 2 5; 3 4 6; 7 8 9]));

%% Test cell2mat - complex numbers
C8 = {1+2i, 3+4i; 5+6i, 7+8i};
A8 = cell2mat(C8);
assert(isequal(A8, [1+2i 3+4i; 5+6i 7+8i]));

%% Test cell2mat - 1xN cell of column vectors
C9 = {[1; 2; 3]};
A9 = cell2mat(C9);
assert(isequal(A9, [1; 2; 3]));

disp('SUCCESS')

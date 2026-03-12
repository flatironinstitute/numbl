%% Test mat2cell - basic 2D split
A = reshape(1:20, 5, 4)';  % 4x5 matrix
C = mat2cell(A, [2 2], [3 2]);
assert(iscell(C));
assert(isequal(size(C), [2, 2]));
assert(isequal(C{1,1}, [1 2 3; 6 7 8]));
assert(isequal(C{2,1}, [11 12 13; 16 17 18]));
assert(isequal(C{1,2}, [4 5; 9 10]));
assert(isequal(C{2,2}, [14 15; 19 20]));

%% Test mat2cell - single row distribution (column vector of cells)
A2 = reshape(1:20, 5, 4)';  % 4x5 matrix
C2 = mat2cell(A2, [1 3]);
assert(iscell(C2));
assert(isequal(size(C2), [2, 1]));
assert(isequal(C2{1}, [1 2 3 4 5]));
assert(isequal(C2{2}, [6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20]));

%% Test mat2cell - single element distributions (each row/col separate)
A3 = [1 2; 3 4; 5 6];
C3 = mat2cell(A3, [1 1 1], [1 1]);
assert(isequal(size(C3), [3, 2]));
assert(isequal(C3{1,1}, 1));
assert(isequal(C3{2,1}, 3));
assert(isequal(C3{3,1}, 5));
assert(isequal(C3{1,2}, 2));
assert(isequal(C3{2,2}, 4));
assert(isequal(C3{3,2}, 6));

%% Test mat2cell - entire array as single cell
A4 = [1 2 3; 4 5 6];
C4 = mat2cell(A4, 2, 3);
assert(isequal(size(C4), [1, 1]));
assert(isequal(C4{1,1}, [1 2 3; 4 5 6]));

%% Test mat2cell - row vector
v = [10 20 30 40 50];
C5 = mat2cell(v, 1, [2 3]);
assert(isequal(size(C5), [1, 2]));
assert(isequal(C5{1}, [10 20]));
assert(isequal(C5{2}, [30 40 50]));

%% Test mat2cell - column vector
v2 = [10; 20; 30; 40];
C6 = mat2cell(v2, [2 2], 1);
assert(isequal(size(C6), [2, 1]));
assert(isequal(C6{1}, [10; 20]));
assert(isequal(C6{2}, [30; 40]));

%% Test mat2cell - scalar
C7 = mat2cell(42, 1, 1);
assert(isequal(size(C7), [1, 1]));
assert(C7{1} == 42);

%% Test mat2cell - complex 2D split
Ac = [1+2i 3+4i; 5+6i 7+8i];
Cc = mat2cell(Ac, [1 1], [1 1]);
assert(isequal(size(Cc), [2, 2]));
assert(Cc{1,1} == 1+2i);
assert(Cc{2,1} == 5+6i);
assert(Cc{1,2} == 3+4i);
assert(Cc{2,2} == 7+8i);

%% Test mat2cell - complex matrix split into blocks
Ac2 = [1+1i 2+2i 3+3i 4+4i; 5+5i 6+6i 7+7i 8+8i; 9+9i 10+10i 11+11i 12+12i];
Cc2 = mat2cell(Ac2, [1 2], [2 2]);
assert(isequal(size(Cc2), [2, 2]));
assert(isequal(Cc2{1,1}, [1+1i 2+2i]));
assert(isequal(Cc2{2,1}, [5+5i 6+6i; 9+9i 10+10i]));
assert(isequal(Cc2{1,2}, [3+3i 4+4i]));
assert(isequal(Cc2{2,2}, [7+7i 8+8i; 11+11i 12+12i]));

%% Test mat2cell - complex single row dist
Ac3 = [1+2i 3+4i; 5+6i 7+8i];
Cc3 = mat2cell(Ac3, [1 1]);
assert(isequal(size(Cc3), [2, 1]));
assert(isequal(Cc3{1}, [1+2i 3+4i]));
assert(isequal(Cc3{2}, [5+6i 7+8i]));

%% Test cell paren-indexed assignment with colon: tmp(j, :) = cellRow
tmp = cell(2, 3);
C_row = mat2cell([1 2 3; 4 5 6], 2, ones(3,1));
tmp(1, :) = C_row;
assert(isequal(tmp{1,1}, [1; 4]));
assert(isequal(tmp{1,2}, [2; 5]));
assert(isequal(tmp{1,3}, [3; 6]));
% Row 2 should still be empty
assert(isequal(tmp{2,1}, []));

%% Cell paren-indexed assignment: second row
C_row2 = mat2cell([10 20 30; 40 50 60], 2, ones(3,1));
tmp(2, :) = C_row2;
assert(isequal(tmp{2,1}, [10; 40]));
assert(isequal(tmp{2,2}, [20; 50]));
assert(isequal(tmp{2,3}, [30; 60]));

%% Cell paren-indexed assignment with colon on rows: tmp(:, k) = cellCol
tmp2 = cell(3, 2);
colCell = {10; 20; 30};
tmp2(:, 1) = colCell;
assert(isequal(tmp2{1,1}, 10));
assert(isequal(tmp2{2,1}, 20));
assert(isequal(tmp2{3,1}, 30));

%% Test cell paren-read indexing with colon: tmp(:, k) and tmp(j, :)
tmp3 = cell(2, 3);
tmp3(1, :) = {10, 20, 30};
tmp3(2, :) = {40, 50, 60};

% Column selection: tmp3(:, 2)
col = tmp3(:, 2);
assert(iscell(col));
assert(isequal(size(col), [2 1]));
assert(isequal(col{1}, 20));
assert(isequal(col{2}, 50));

% Row selection: tmp3(1, :)
row = tmp3(1, :);
assert(iscell(row));
assert(isequal(size(row), [1 3]));
assert(isequal(row{1}, 10));
assert(isequal(row{2}, 20));
assert(isequal(row{3}, 30));

% Full colon: tmp3(:, :)
all = tmp3(:, :);
assert(iscell(all));
assert(isequal(size(all), [2 3]));
assert(isequal(all{1,1}, 10));
assert(isequal(all{2,3}, 60));

%% Cell paren-indexing with colon on 1x1 cell returns 1x1 cell (not the element)
tiny = {42};
sub = tiny(:, 1);
assert(iscell(sub));
assert(isequal(size(sub), [1 1]));
assert(isequal(sub{1}, 42));

%% Cell paren-indexing tmp(:, k) on nFuns x 1 returns nFuns x 1 cell
tmp4 = cell(1, 1);
tmp4{1} = 99;
col4 = tmp4(:, 1);
assert(iscell(col4));
assert(isequal(size(col4), [1 1]));
assert(isequal(col4{1}, 99));

%% Test mat2cell with char array
C_char = mat2cell(':::', 1, [1 1 1]);
assert(iscell(C_char));
assert(isequal(size(C_char), [1 3]));
assert(isequal(C_char{1}, ':'));
assert(isequal(C_char{2}, ':'));
assert(isequal(C_char{3}, ':'));

%% Test mat2cell with char - multi-char splits
C_char2 = mat2cell('abcdef', 1, [2 3 1]);
assert(isequal(C_char2{1}, 'ab'));
assert(isequal(C_char2{2}, 'cde'));
assert(isequal(C_char2{3}, 'f'));

disp('SUCCESS')

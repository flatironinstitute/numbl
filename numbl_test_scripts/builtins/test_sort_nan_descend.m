% Test that sort with 'descend' puts NaN at the BEGINNING.
% MATLAB places NaN at the end for ascending, beginning for descending.

% 1. Basic descend with NaN in a row vector
x = [3 NaN 1 2];
[y, idx] = sort(x, 'descend');
assert(isnan(y(1)), 'NaN should be at the beginning in descend');
assert(isequal(y(2:4), [3 2 1]), 'descend values wrong');
assert(idx(1) == 2, 'NaN index should be first');
assert(isequal(idx(2:4), [1 4 3]), 'descend indices wrong');

% 2. Multiple NaN values
x2 = [NaN 5 NaN 2 4];
[y2, idx2] = sort(x2, 'descend');
assert(isnan(y2(1)) && isnan(y2(2)), 'NaN should be at beginning');
assert(isequal(y2(3:5), [5 4 2]), 'multiple NaN descend values wrong');

% 3. Ascending puts NaN at the end (baseline)
[ya, idxa] = sort(x, 'ascend');
assert(isequal(ya(1:3), [1 2 3]), 'ascend values wrong');
assert(isnan(ya(4)), 'NaN should be at the end in ascend');

% 4. Column vector
x3 = [NaN; 3; 1; NaN; 2];
[y3, idx3] = sort(x3, 'descend');
assert(isnan(y3(1)) && isnan(y3(2)), 'column NaN should be at beginning');
assert(isequal(y3(3:5), [3; 2; 1]), 'column descend values wrong');

% 5. Matrix sort along dim 1 (sort each column descending)
M = [1 NaN; NaN 4; 3 2];
[yM, idxM] = sort(M, 1, 'descend');
% Column 1: [NaN; 3; 1], Column 2: [NaN; 4; 2]
assert(isnan(yM(1, 1)), 'matrix col1 NaN should be first');
assert(isequal(yM(2:3, 1), [3; 1]), 'matrix col1 descend wrong');
assert(isnan(yM(1, 2)), 'matrix col2 NaN should be first');
assert(isequal(yM(2:3, 2), [4; 2]), 'matrix col2 descend wrong');

disp('SUCCESS');

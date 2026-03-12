% Test unique with 'rows', 'stable', and multiple outputs
% Basic unique with rows and stable
A = [3 1; 2 4; 3 1; 1 5];
[C, ia, ic] = unique(A, 'rows', 'stable');
assert(size(C, 1) == 3);   % 3 unique rows
assert(C(1,1) == 3 && C(1,2) == 1);  % first unique row
assert(C(2,1) == 2 && C(2,2) == 4);  % second unique row
assert(ia(1) == 1);
assert(ia(2) == 2);
assert(ic(1) == 1);
assert(ic(3) == 1);  % row 3 same as row 1
assert(ic(4) == 3);  % row 4 is the third unique

% unique vector with stable
[C2, ~, ic2] = unique([5 3 5 1 3], 'stable');
assert(isequal(C2, [5 3 1]));
assert(ic2(1) == 1);
assert(ic2(3) == 1);  % same as first

% unique vector with sorted (default)
[C3, ~, ic3] = unique([5 3 5 1 3]);
assert(isequal(C3, [1 3 5]));

fprintf('SUCCESS\n');

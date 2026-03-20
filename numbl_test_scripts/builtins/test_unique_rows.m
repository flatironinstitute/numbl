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

% Sorted unique rows with 3 outputs
B = [5 6; 1 2; 3 4; 1 2; 5 6];
[C4, ia4, ic4] = unique(B, 'rows');
assert(isequal(C4, [1 2; 3 4; 5 6]), 'sorted unique rows C');
assert(isequal(ia4, [2; 3; 1]), 'sorted unique rows ia');
assert(isequal(ic4, [3; 1; 2; 1; 3]), 'sorted unique rows ic');

% Stable unique rows with 3 outputs
[C5, ia5, ic5] = unique(B, 'rows', 'stable');
assert(isequal(C5, [5 6; 1 2; 3 4]), 'stable unique rows C');
assert(isequal(ia5, [1; 2; 3]), 'stable unique rows ia');
assert(isequal(ic5, [1; 2; 3; 2; 1]), 'stable unique rows ic');

% All rows same
D = [1 2; 1 2; 1 2];
[C6, ia6, ic6] = unique(D, 'rows');
assert(isequal(C6, [1 2]), 'all-same C');
assert(isequal(ia6, [1]), 'all-same ia');
assert(isequal(ic6, [1; 1; 1]), 'all-same ic');

% All rows unique
F = [1 2; 3 4; 5 6];
[C7, ~, ic7] = unique(F, 'rows');
assert(isequal(C7, F), 'all unique C');
assert(isequal(ic7, [1; 2; 3]), 'all unique ic');

% Wide matrix (4 cols)
G = [1 2 3 4; 5 6 7 8; 1 2 3 4; 9 10 11 12];
[C8, ~, ic8] = unique(G, 'rows');
assert(size(C8, 1) == 3, 'wide unique count');
assert(isequal(ic8, [1; 2; 1; 3]), 'wide ic');

% Larger random: verify C(ic,:) reconstructs original
M = randi(10, 100, 3);
[C9, ia9, ic9] = unique(M, 'rows');
assert(isequal(C9(ic9, :), M), 'C(ic,:) should reconstruct original');
assert(isequal(C9, M(ia9,:)), 'C should equal M(ia,:)');

% Negative and float values
H = [-1.5 2.5; 3.5 -4.5; -1.5 2.5];
[C10, ~, ic10] = unique(H, 'rows');
assert(size(C10, 1) == 2, 'float unique count');
assert(isequal(ic10, [1; 2; 1]), 'float ic');

fprintf('SUCCESS\n');

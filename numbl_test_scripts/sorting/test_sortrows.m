% Test sortrows builtin

%% Basic: sort by first column
A = [3 1; 1 2; 2 3];
B = sortrows(A);
assert(isequal(B, [1 2; 2 3; 3 1]));

%% Sort by specific column
A = [1 3; 2 1; 3 2];
B = sortrows(A, 2);
assert(isequal(B, [2 1; 3 2; 1 3]));

%% Sort with index output
A = [3 1; 1 2; 2 3];
[B, idx] = sortrows(A);
assert(isequal(B, [1 2; 2 3; 3 1]));
assert(isequal(idx, [2; 3; 1]));

%% Sort by multiple columns (tiebreaker)
A = [1 3; 1 1; 2 2; 1 2];
B = sortrows(A);
assert(isequal(B, [1 1; 1 2; 1 3; 2 2]));

%% Descending sort by column
A = [1 3; 2 1; 3 2];
B = sortrows(A, -1);
assert(isequal(B, [3 2; 2 1; 1 3]));

%% Single column matrix
A = [5; 3; 1; 4; 2];
B = sortrows(A);
assert(isequal(B, [1; 2; 3; 4; 5]));

%% Already sorted
A = [1 2; 3 4; 5 6];
B = sortrows(A);
assert(isequal(B, A));

disp('SUCCESS');

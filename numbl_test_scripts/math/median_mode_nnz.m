% Test statistical functions: median, mode, nnz

%% median - middle value (sorted)
% Odd-length vector
assert(median([3, 1, 2]) == 2);
assert(median([5, 1, 3, 2, 4]) == 3);

% Even-length vector: average of two middle values
assert(median([1, 2, 3, 4]) == 2.5);
assert(median([4, 2, 1, 3]) == 2.5);

% Scalar
assert(median(7) == 7);

% Column vector
assert(median([3; 1; 2]) == 2);

% Matrix: median along dim 1 (each column)
M = [3, 6; 1, 4; 2, 5];
m = median(M);
assert(m(1) == 2);
assert(m(2) == 5);

% Matrix median along dim 2 (each row)
m2 = median(M, 2);
assert(m2(1) == 4.5);  % median([3, 6]) = 4.5
assert(m2(2) == 2.5);  % median([1, 4]) = 2.5
assert(m2(3) == 3.5);  % median([2, 5]) = 3.5

%% mode - most frequent value
assert(mode([1, 2, 2, 3]) == 2);
assert(mode([1, 1, 2, 2, 3]) == 1);  % tie: smallest value wins

% Scalar
assert(mode(5) == 5);

% Column vector
assert(mode([4; 4; 5; 5; 5]) == 5);

% Matrix: mode along dim 1 (each column)
M2 = [1, 3; 2, 3; 2, 4];
m3 = mode(M2);
assert(m3(1) == 2);
assert(m3(2) == 3);

%% nnz - number of non-zero elements
assert(nnz([0, 1, 0, 3, 0]) == 2);
assert(nnz([1, 2, 3]) == 3);
assert(nnz([0, 0, 0]) == 0);
assert(nnz(5) == 1);
assert(nnz(0) == 0);

% Matrix
assert(nnz([1, 0; 0, 2; 3, 0]) == 3);

% Logical
assert(nnz([true, false, true]) == 2);

disp('SUCCESS');

%% NaN with no args returns scalar NaN
x = NaN;
assert(isnan(x));
assert(numel(x) == 1);

%% NaN(n) returns n-by-n matrix
A = NaN(3);
assert(isequal(size(A), [3 3]));
assert(sum(sum(isnan(A))) == 9);

%% NaN(sz1, sz2) returns sz1-by-sz2 matrix
B = NaN(2, 4);
assert(isequal(size(B), [2 4]));
assert(sum(sum(isnan(B))) == 8);

%% NaN(1, n) returns row vector
C = NaN(1, 5);
assert(isequal(size(C), [1 5]));
assert(sum(isnan(C)) == 5);

%% NaN([sz]) returns array of given size
D = NaN([2 3]);
assert(isequal(size(D), [2 3]));
assert(sum(sum(isnan(D))) == 6);

disp('SUCCESS')

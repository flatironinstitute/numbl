%% all(A, dim) along columns (dim=1)
A = [1 0 1; 1 1 1; 1 0 1];
r = all(A, 1);
assert(isequal(r, [true false true]));

%% all(A, dim) along rows (dim=2)
r2 = all(A, 2);
assert(isequal(r2, [false; true; false]));

%% all(A, 'all')
B = [1 1; 1 1];
assert(all(B, 'all'));
C = [1 1; 0 1];
assert(~all(C, 'all'));

%% all(A, dim) with logical input from isnan
M = [NaN 1; NaN 2; NaN 3];
r3 = all(isnan(M), 2);
assert(isequal(r3, [false; false; false]));

M2 = [NaN NaN; 1 2; NaN NaN];
r4 = all(isnan(M2), 2);
assert(isequal(r4, [true; false; true]));

disp('SUCCESS')

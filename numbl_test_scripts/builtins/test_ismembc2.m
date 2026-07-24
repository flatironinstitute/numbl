% ismembc2(a, s): index of the last occurrence of each element of a in the
% sorted array s, or 0 if absent (undocumented MATLAB helper).

s = [1 2 3 5 9];
assert(isequal(ismembc2([2 5 7], s), [2 4 0]));
assert(isequal(ismembc2([1 9], s), [1 5]));
assert(ismembc2(4, s) == 0);
assert(ismembc2(0, s) == 0);
assert(ismembc2(10, s) == 0);

% Last occurrence for repeated values
s2 = [1 2 2 2 3];
assert(ismembc2(2, s2) == 4);
assert(ismembc2(1, s2) == 1);
assert(ismembc2(3, s2) == 5);

% Shape of the query is preserved
r = ismembc2([1; 3], s);
assert(isequal(size(r), [2 1]));
assert(isequal(r, [1; 3]));

disp('SUCCESS');

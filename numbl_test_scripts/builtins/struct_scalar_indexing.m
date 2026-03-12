% Test that scalar struct supports s(1) indexing
% In MATLAB, every value is implicitly a 1-element array, so s(1) == s

s = struct('a', 1, 'b', 2);

% s(1) should return the struct itself
s1 = s(1);
assert(isequal(s1, s), 's(1) should equal s');

% Access fields through s(1)
assert(s(1).a == 1, 's(1).a should be 1');
assert(s(1).b == 2, 's(1).b should be 2');

% Build a struct via indexed assignment then access it
data.x = 10;
data.y = 20;
val = data(1).x;
assert(val == 10, 'data(1).x should be 10');

disp('SUCCESS');

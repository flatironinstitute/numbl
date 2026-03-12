% Test that cell assignment preserves shape

% Column vector stays column
c = cell(5, 1);
c{3} = 10;
assert(isequal(size(c), [5, 1]));

% Row vector stays row
r = cell(1, 5);
r{3} = 20;
assert(isequal(size(r), [1, 5]));

% 2D cell stays 2D
m = cell(2, 3);
m{2} = 99;
assert(isequal(size(m), [2, 3]));

% Growth preserves column orientation
c2 = cell(3, 1);
c2{5} = 42;
assert(isequal(size(c2), [5, 1]));

% Growth preserves row orientation
r2 = cell(1, 3);
r2{5} = 42;
assert(isequal(size(r2), [1, 5]));

fprintf('SUCCESS\n');

% Test that cell() accepts a vector argument like cell(size(x))
x = {1, 2, 3; 4, 5, 6};
c = cell(size(x));
assert(iscell(c));
assert(size(c, 1) == 2);
assert(size(c, 2) == 3);
assert(isempty(c{1,1}));

% Also test cell([1 4])
c2 = cell([1, 4]);
assert(size(c2, 1) == 1);
assert(size(c2, 2) == 4);

% cell(3) should still work (scalar)
c3 = cell(3);
assert(size(c3, 1) == 3);
assert(size(c3, 2) == 3);

fprintf('SUCCESS\n');

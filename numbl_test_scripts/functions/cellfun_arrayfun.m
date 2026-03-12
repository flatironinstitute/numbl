% Test cellfun and arrayfun

% arrayfun: apply function to each element
v = [1, 2, 3, 4, 5];
result = arrayfun(@(x) x^2, v);
assert(result(1) == 1);
assert(result(2) == 4);
assert(result(3) == 9);
assert(result(5) == 25);

% arrayfun with another anonymous function
% Note: @localfunc handles for local functions in same file not yet supported (see TODO.md)
result2 = arrayfun(@(x) x * 2, v);
assert(result2(1) == 2);
assert(result2(3) == 6);
assert(result2(5) == 10);

% cellfun: apply to each cell element
words = {'hello', 'world', 'foo'};
lens = cellfun(@length, words);
assert(lens(1) == 5);
assert(lens(2) == 5);
assert(lens(3) == 3);

% cellfun returning logical scalars (e.g. from isa/strcmp) — result must be numeric
c = {1, 'hello', 3};
isnum = cellfun(@isnumeric, c);
assert(isnum(1) == 1);
assert(isnum(2) == 0);
assert(isnum(3) == 1);

% find on cellfun logical output
idx = find(cellfun(@isnumeric, c));
assert(idx(1) == 1);
assert(idx(2) == 3);

disp('SUCCESS')

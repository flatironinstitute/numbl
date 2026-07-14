% Test struct array comma-separated list expansion
% [s.field] should expand to [s(1).field, s(2).field, ...]

% Create struct array
s(1).name = 'Alice';
s(1).age = 30;
s(2).name = 'Bob';
s(2).age = 25;
s(3).name = 'Charlie';
s(3).age = 35;

% Field expansion into array
ages = [s.age];
assert(isequal(ages, [30 25 35]));

% Field expansion with two elements
s2(1).x = 10;
s2(2).x = 20;
vals = [s2.x];
assert(isequal(vals, [10 20]));

% Field expansion into a cell array: {s.field} gives one cell per element
names = {s.name};
assert(iscell(names));
assert(isequal(size(names), [1 3]));
assert(strcmp(names{1}, 'Alice'));
assert(strcmp(names{2}, 'Bob'));
assert(strcmp(names{3}, 'Charlie'));

% Field expansion into function arguments: one argument per element
joined = sprintf('%d,%d,%d', s.age);
assert(strcmp(joined, '30,25,35'));

% Expansion in a multi-row cell literal
grid = {s2.x; 'a', 'b'};
assert(isequal(size(grid), [2 2]));
assert(grid{1, 1} == 10);
assert(grid{1, 2} == 20);
assert(strcmp(grid{2, 1}, 'a'));

disp('SUCCESS');

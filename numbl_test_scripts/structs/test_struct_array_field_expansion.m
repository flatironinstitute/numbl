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

disp('SUCCESS');

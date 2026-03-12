%% horzcat two cell arrays: [cellA, cellB]
a = {1, 2};
b = {3};
c = [a, b];
assert(iscell(c));
assert(isequal(size(c), [1 3]));
assert(c{1} == 1);
assert(c{2} == 2);
assert(c{3} == 3);

%% horzcat cell with {value}
d = {10, 20};
e = [d, {30}];
assert(iscell(e));
assert(isequal(size(e), [1 3]));
assert(e{3} == 30);

%% horzcat empty cell with {value}
f = {};
g = [f, {42}];
assert(iscell(g));
assert(isequal(size(g), [1 1]));
assert(g{1} == 42);

%% vertcat two cell arrays: [cellA; cellB]
h = {1; 2};
j = {3};
k = [h; j];
assert(iscell(k));
assert(isequal(size(k), [3 1]));
assert(k{1} == 1);
assert(k{2} == 2);
assert(k{3} == 3);

disp('SUCCESS')

%% Delete cell elements with logical indexing: c(logicals) = []
c = {1, 2, 3, 4, 5};
c([false, true, false, true, false]) = [];
assert(iscell(c));
assert(isequal(size(c), [1 3]));
assert(c{1} == 1);
assert(c{2} == 3);
assert(c{3} == 5);

%% Delete cell elements with numeric indexing: c([2 4]) = []
d = {'a', 'b', 'c', 'd', 'e'};
d([2 4]) = [];
assert(iscell(d));
assert(isequal(size(d), [1 3]));
assert(strcmp(d{1}, 'a'));
assert(strcmp(d{2}, 'c'));
assert(strcmp(d{3}, 'e'));

%% Delete nothing (all false)
e = {10, 20, 30};
e([false, false, false]) = [];
assert(isequal(size(e), [1 3]));

%% Delete all elements
f = {1, 2};
f([true, true]) = [];
assert(iscell(f));
assert(isempty(f));

%% Delete from column cell
g = {1; 2; 3; 4};
g([1 3]) = [];
assert(iscell(g));
assert(isequal(size(g), [2 1]));
assert(g{1} == 2);
assert(g{2} == 4);

disp('SUCCESS')

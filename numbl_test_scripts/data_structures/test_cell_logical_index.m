%% Cell logical indexing: c(logicals)
c = {10, 20, 30, 40, 50};
d = c([true, false, true, false, true]);
assert(iscell(d));
assert(isequal(size(d), [1 3]));
assert(d{1} == 10);
assert(d{2} == 30);
assert(d{3} == 50);

%% Cell logical indexing: all true
e = {'a', 'b'};
f = e([true, true]);
assert(iscell(f));
assert(isequal(size(f), [1 2]));

%% Cell logical indexing: all false
g = {1, 2, 3};
h = g([false, false, false]);
assert(iscell(h));
assert(isempty(h));

%% Cell logical indexing on column cell
j = {1; 2; 3; 4};
k = j([false; true; false; true]);
assert(iscell(k));
assert(isequal(size(k), [2 1]));
assert(k{1} == 2);
assert(k{2} == 4);

disp('SUCCESS')

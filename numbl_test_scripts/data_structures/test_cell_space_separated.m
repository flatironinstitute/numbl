% Test space-separated cell array literals

%% Basic space-separated cell
c1 = {1 2 3};
assert(iscell(c1));
assert(numel(c1) == 3);
assert(c1{1} == 1);
assert(c1{2} == 2);
assert(c1{3} == 3);

%% Space-separated strings
c2 = {'a' 'b' 'c'};
assert(numel(c2) == 3);
assert(strcmp(c2{1}, 'a'));
assert(strcmp(c2{2}, 'b'));
assert(strcmp(c2{3}, 'c'));

%% Mixed types space-separated
c3 = {1 'hello' [1 2 3]};
assert(numel(c3) == 3);
assert(c3{1} == 1);
assert(strcmp(c3{2}, 'hello'));
assert(isequal(c3{3}, [1 2 3]));

%% Single element (should still work)
c4 = {42};
assert(numel(c4) == 1);
assert(c4{1} == 42);

disp('SUCCESS')

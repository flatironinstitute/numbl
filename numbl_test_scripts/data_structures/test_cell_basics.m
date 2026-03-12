% Test basic cell array functionality

%% Cell creation - row vector
c1 = {1, 2, 3};
assert(iscell(c1));
assert(numel(c1) == 3);
assert(length(c1) == 3);

%% Cell creation - mixed types
c2 = {42, 'hello', [1 2 3], true};
assert(numel(c2) == 4);

%% Curly-brace indexing (content access)
assert(c2{1} == 42);
assert(strcmp(c2{2}, 'hello'));
assert(isequal(c2{3}, [1 2 3]));
assert(c2{4} == true);

%% Curly-brace assignment
c3 = {0, 0, 0};
c3{1} = 10;
c3{2} = 'world';
c3{3} = [4 5 6];
assert(c3{1} == 10);
assert(strcmp(c3{2}, 'world'));
assert(isequal(c3{3}, [4 5 6]));

%% Cell auto-growth on assignment
c4 = {};
c4{1} = 'a';
c4{2} = 'b';
c4{3} = 'c';
assert(numel(c4) == 3);
assert(strcmp(c4{1}, 'a'));
assert(strcmp(c4{3}, 'c'));

%% Empty cell array
c5 = {};
assert(iscell(c5));
assert(numel(c5) == 0);
assert(isempty(c5));

%% cell() constructor
c6 = cell(1, 3);
assert(iscell(c6));
assert(numel(c6) == 3);
assert(isempty(c6{1}));

c7 = cell(2, 2);
assert(numel(c7) == 4);

%% Nested indexing - cell containing array
c8 = {[10 20 30], [40 50]};
assert(c8{1}(2) == 20);
assert(c8{2}(1) == 40);

%% Nested cells
c9 = {{1, 2}, {3, 4}};
assert(c9{1}{1} == 1);
assert(c9{1}{2} == 2);
assert(c9{2}{1} == 3);
assert(c9{2}{2} == 4);

%% iscell on non-cells
assert(~iscell(42));
assert(~iscell('hello'));
assert(~iscell([1 2 3]));
assert(~iscell(struct()));

%% Auto-create cell from undefined variable
y{3} = 2;
assert(iscell(y));
assert(numel(y) == 3);
assert(isempty(y{1}));
assert(isempty(y{2}));
assert(y{3} == 2);

%% class() of cell
assert(strcmp(class(c1), 'cell'));

%% size() of cell
c10 = {1, 2, 3};
assert(isequal(size(c10), [1 3]));

c11 = cell(3, 1);
assert(isequal(size(c11), [3 1]));

disp('SUCCESS')

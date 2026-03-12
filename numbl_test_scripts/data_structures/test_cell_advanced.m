% Test advanced cell array functionality

%% Deeply nested cells
c4 = {{{1}}};
assert(c4{1}{1}{1} == 1);

%% Cell containing structs
s1.x = 10;
s1.y = 20;
s2.x = 30;
s2.y = 40;
c5 = {s1, s2};
assert(c5{1}.x == 10);
assert(c5{1}.y == 20);
assert(c5{2}.x == 30);
assert(c5{2}.y == 40);

%% Cell containing function handles
c6 = {@sin, @cos, @abs};
assert(c6{1}(0) == 0);
assert(c6{2}(0) == 1);
assert(c6{3}(-5) == 5);

%% Iterating over cells and accumulating
vals = {10, 20, 30, 40};
total = 0;
for i = 1:numel(vals)
    total = total + vals{i};
end
assert(total == 100);

%% Cell with empty elements
c7 = {[], 'text', []};
assert(isempty(c7{1}));
assert(strcmp(c7{2}, 'text'));
assert(isempty(c7{3}));

%% Overwriting cell elements changes type
c9 = {1, 2, 3};
c9{2} = 'replaced';
assert(strcmp(c9{2}, 'replaced'));
assert(c9{1} == 1);
assert(c9{3} == 3);

%% cellfun with anonymous function
nums = {1, 4, 9, 16};
roots = cellfun(@(x) sqrt(x), nums);
assert(isequal(roots, [1, 2, 3, 4]));

%% Cell of cells - building incrementally
outer = cell(1, 3);
outer{1} = {1, 2};
outer{2} = {3, 4};
outer{3} = {5, 6};
assert(outer{1}{1} == 1);
assert(outer{2}{2} == 4);
assert(outer{3}{1} == 5);

%% Cell containing different numeric types
c10 = {1, 3.14, -7, 0, 1e10};
assert(c10{1} == 1);
assert(abs(c10{2} - 3.14) < 1e-10);
assert(c10{3} == -7);
assert(c10{4} == 0);
assert(c10{5} == 1e10);

%% Cell with matrices of different sizes
c11 = {[1 2], [1 2 3], [1; 2; 3]};
assert(isequal(size(c11{1}), [1, 2]));
assert(isequal(size(c11{2}), [1, 3]));
assert(isequal(size(c11{3}), [3, 1]));

%% Building a cell in a loop
n = 5;
c12 = cell(1, n);
for i = 1:n
    c12{i} = i^2;
end
assert(c12{1} == 1);
assert(c12{2} == 4);
assert(c12{3} == 9);
assert(c12{4} == 16);
assert(c12{5} == 25);

%% Nested cell modification
c13 = {{1, 2}, {3, 4}};
c13{1}{2} = 99;
assert(c13{1}{2} == 99);
assert(c13{1}{1} == 1);

disp('SUCCESS')

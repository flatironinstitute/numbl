% Test 2D cell array indexing (row, col subscripts)

%% 2D cell creation with semicolons
c1 = {1, 2; 3, 4};
assert(isequal(size(c1), [2, 2]));
assert(c1{1,1} == 1);
assert(c1{1,2} == 2);
assert(c1{2,1} == 3);
assert(c1{2,2} == 4);

%% 2D cell with mixed types
c2 = {'name', 'Alice'; 'age', 30};
assert(strcmp(c2{1,1}, 'name'));
assert(strcmp(c2{1,2}, 'Alice'));
assert(strcmp(c2{2,1}, 'age'));
assert(c2{2,2} == 30);

%% 2D cell assignment
c3 = cell(2, 3);
c3{1,1} = 'a';
c3{1,2} = 'b';
c3{2,3} = 'c';
assert(strcmp(c3{1,1}, 'a'));
assert(strcmp(c3{1,2}, 'b'));
assert(strcmp(c3{2,3}, 'c'));

%% size with dim argument on 2D cell
c4 = {1, 2, 3; 4, 5, 6};
assert(size(c4, 1) == 2);
assert(size(c4, 2) == 3);
assert(numel(c4) == 6);

%% 2D cell - linear indexing (column-major)
c5 = {'a', 'b'; 'c', 'd'};
% In MATLAB, linear indexing is column-major: c5{1}='a', c5{2}='c', c5{3}='b', c5{4}='d'
assert(strcmp(c5{1}, 'a'));
assert(strcmp(c5{2}, 'c'));
assert(strcmp(c5{3}, 'b'));
assert(strcmp(c5{4}, 'd'));

disp('SUCCESS')

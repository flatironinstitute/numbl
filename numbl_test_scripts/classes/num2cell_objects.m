% num2cell on scalar objects and object arrays
a = Num2cellObj_(1);
c = num2cell(a);
assert(iscell(c) && isequal(size(c), [1 1]));
assert(c{1}.v == 1);
c = num2cell(a(:).', 1);
assert(isequal(size(c), [1 1]) && c{1}.v == 1);

arr = [Num2cellObj_(1), Num2cellObj_(2), Num2cellObj_(3)];
c = num2cell(arr);
assert(isequal(size(c), [1 3]));
assert(c{3}.v == 3);
assert(isa(c{2}, 'Num2cellObj_'));

% the fmm3dbie idiom: one cell per object
c = num2cell(arr(:).', 1);
assert(isequal(size(c), [1 3]));
assert(c{2}.v == 2);
assert(isequal(size(c{2}), [1 1]));

c = num2cell(arr(:));
assert(isequal(size(c), [3 1]));
assert(c{3}.v == 3);

% dim 2 on a row keeps the whole row in one cell
c = num2cell(arr, 2);
assert(isequal(size(c), [1 1]));
assert(isequal(size(c{1}), [1 3]));
assert(c{1}(3).v == 3);

% struct arrays with a dimension argument
s = struct('x', {1, 2, 3});
c = num2cell(s, 1);
assert(isequal(size(c), [1 3]) && c{2}.x == 2);
c = num2cell(s, 2);
assert(isequal(size(c), [1 1]) && numel(c{1}) == 3);

disp('SUCCESS');

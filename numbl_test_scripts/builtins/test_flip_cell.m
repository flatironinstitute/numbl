% flip/flipud/fliplr on cell arrays

% Row vector cell: flip reverses along dim 2
c = {'a', 'b', 'c'};
f = flip(c);
assert(strcmp(f{1}, 'c'));
assert(strcmp(f{2}, 'b'));
assert(strcmp(f{3}, 'a'));

% Column vector cell
c = {1; 2; 3};
f = flip(c);
assert(f{1} == 3);
assert(f{3} == 1);

% Explicit dimension on a 2x2 cell
c = {1, 2; 3, 4};
f1 = flip(c, 1);
assert(f1{1, 1} == 3);
assert(f1{1, 2} == 4);
assert(f1{2, 1} == 1);
f2 = flip(c, 2);
assert(f2{1, 1} == 2);
assert(f2{2, 1} == 4);
assert(f2{1, 2} == 1);

% flipud / fliplr agree with flip along dims 1 / 2
u = flipud(c);
assert(u{1, 1} == f1{1, 1} && u{2, 2} == f1{2, 2});
l = fliplr(c);
assert(l{1, 1} == f2{1, 1} && l{2, 2} == f2{2, 2});

% Empty cell stays empty
e = flip({});
assert(isempty(e));

% Mixed contents survive by reference
c = {'hello', [1 2 3]};
f = flip(c);
assert(isnumeric(f{1}) && numel(f{1}) == 3);
assert(strcmp(f{2}, 'hello'));

disp('SUCCESS')

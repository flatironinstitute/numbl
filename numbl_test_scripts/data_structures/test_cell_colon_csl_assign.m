% Test [c{:}] = func() — colon expansion in multi-output cell assignment.

% Two-element column cell via deal
out = cell(2, 1);
[out{:}] = deal(10, 20);
assert(isequal(out{1}, 10), 'col cell element 1 mismatch');
assert(isequal(out{2}, 20), 'col cell element 2 mismatch');

% Two-element row cell via deal
outr = cell(1, 2);
[outr{:}] = deal('a', 'b');
assert(strcmp(outr{1}, 'a'), 'row cell element 1 mismatch');
assert(strcmp(outr{2}, 'b'), 'row cell element 2 mismatch');

% Three-element via deal with mixed types
m = cell(1, 3);
[m{:}] = deal(1.5, 'hello', [1 2 3]);
assert(m{1} == 1.5, 'm{1}');
assert(strcmp(m{2}, 'hello'), 'm{2}');
assert(isequal(m{3}, [1 2 3]), 'm{3}');

% Single-element colon: equivalent to scalar index
s = cell(1, 1);
[s{:}] = deal(99);
assert(s{1} == 99, 'single element via colon');

% Distributing outputs of a multi-output function via colon
c = cell(1, 2);
[c{:}] = size([1 2 3; 4 5 6]);
assert(c{1} == 2, 'size rows');
assert(c{2} == 3, 'size cols');

% Three-output size via colon
c3 = cell(1, 3);
[c3{:}] = size([1 2; 3 4]);
assert(c3{1} == 2, 'c3{1}');
assert(c3{2} == 2, 'c3{2}');
assert(c3{3} == 1, 'c3{3}');

% Colon on a 2x2 cell expands linearly (column-major)
cm = cell(2, 2);
[cm{:}] = deal(1, 2, 3, 4);
assert(cm{1, 1} == 1, 'cm 1,1');
assert(cm{2, 1} == 2, 'cm 2,1');
assert(cm{1, 2} == 3, 'cm 1,2');
assert(cm{2, 2} == 4, 'cm 2,2');

disp('SUCCESS');

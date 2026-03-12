% Test multi-output assignment to cell array elements via [c{idx}] = func()

% Single output assignment
c = cell(1, 1);
[c{1}] = max([3 1 2]);
assert(isequal(c{1}, 3));
assert(strcmp(class(c{1}), 'double'));

% Two output assignment with range index
c2 = cell(1, 2);
[c2{1:2}] = size([1 2; 3 4]);
assert(c2{1} == 2);
assert(c2{2} == 2);

% Three output assignment with deal
c3 = cell(1, 3);
[c3{1:3}] = deal(10, 20, 30);
assert(c3{1} == 10);
assert(c3{2} == 20);
assert(c3{3} == 30);

% Assignment to non-contiguous indices
c4 = cell(1, 4);
[c4{[1 3]}] = size([1 2; 3 4]);
assert(c4{1} == 2);
assert(c4{3} == 2);

% Single output with string result
c5 = cell(1, 1);
[c5{1}] = class(42);
assert(strcmp(c5{1}, 'double'));

disp('SUCCESS');

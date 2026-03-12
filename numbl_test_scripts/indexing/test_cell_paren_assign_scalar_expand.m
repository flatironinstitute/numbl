% Test cell array parenthesis assignment with scalar expansion
% In MATLAB, C(indices) = { scalar } assigns the scalar to all selected cells

% Numeric multi-index
blocks = cell(1, 3);
blocks(1, [1 3]) = { 42 };
assert(blocks{1} == 42);
assert(isempty(blocks{2}));
assert(blocks{3} == 42);

% Logical index
blocks2 = cell(1, 3);
v = logical([1 0 1]);
blocks2(1, v) = { 99 };
assert(blocks2{1} == 99);
assert(isempty(blocks2{2}));
assert(blocks2{3} == 99);

% All-true logical
blocks3 = cell(1, 2);
blocks3(1, true(1, 2)) = { 'hello' };
assert(strcmp(blocks3{1}, 'hello'));
assert(strcmp(blocks3{2}, 'hello'));

% All-false logical (no assignment)
blocks4 = cell(1, 2);
blocks4{1} = 10;
blocks4{2} = 20;
blocks4(1, false(1, 2)) = { 55 };
assert(blocks4{1} == 10);
assert(blocks4{2} == 20);

% Single-dimension indexing
C = cell(1, 4);
C([2 4]) = { 'x' };
assert(strcmp(C{2}, 'x'));
assert(strcmp(C{4}, 'x'));
assert(isempty(C{1}));
assert(isempty(C{3}));

% String values
S = cell(1, 3);
S([1 2 3]) = { 'test' };
assert(strcmp(S{1}, 'test'));
assert(strcmp(S{2}, 'test'));
assert(strcmp(S{3}, 'test'));

disp('SUCCESS');

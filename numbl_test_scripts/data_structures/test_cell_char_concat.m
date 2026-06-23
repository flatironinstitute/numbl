% Concatenating a cell with non-cell operands yields a cell: each non-cell
% operand is wrapped into one cell, and empty operands ('' / []) are dropped.
a = ['', {'x', 'y'}];
assert(iscell(a) && numel(a) == 2, 'empty char dropped');

b = [5, {'x'}];
assert(iscell(b) && b{1} == 5 && strcmp(b{2}, 'x'), 'numeric kept as-is');

c = [{'x'}; 'ab'];
assert(iscell(c) && numel(c) == 2 && strcmp(c{2}, 'ab'), 'vertical, char wrapped');

d = [[], {'x'}];
assert(iscell(d) && numel(d) == 1, 'empty double dropped');

e = ['[', sprintf('\n'), {'aa', 'bb'}];
assert(iscell(e) && numel(e) == 4 && strcmp(e{1}, '['), 'mixed char + cell');
disp('SUCCESS');

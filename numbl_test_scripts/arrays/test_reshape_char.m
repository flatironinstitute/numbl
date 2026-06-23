% reshape works on char arrays (column-major), like MATLAB.
assert(strcmp(reshape('hello', [1 5]), 'hello'), 'row no-op');
assert(isequal(size(reshape('hello', [5 1])), [5 1]), 'to column');
assert(isequal(size(reshape('hello', 1, 5)), [1 5]), 'scalar dim args');

x = reshape('abcdef', [2 3]);
assert(isequal(size(x), [2 3]), '2x3 shape');
assert(strcmp(x(1, :), 'ace'), 'column-major row 1');
assert(strcmp(x(2, :), 'bdf'), 'column-major row 2');

assert(isequal(size(reshape('abcd', [], 2)), [2 2]), 'auto dimension');
disp('SUCCESS');

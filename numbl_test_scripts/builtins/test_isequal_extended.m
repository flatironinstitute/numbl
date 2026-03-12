% Test isequal with various types (compare.ts coverage)

% Char arrays
assert(isequal('hello', 'hello'));
assert(~isequal('hello', 'world'));

% Tensors
assert(isequal([1 2 3], [1 2 3]));
assert(~isequal([1 2 3], [1 2 4]));
assert(~isequal([1 2], [1 2 3]));

% Complex tensors
assert(isequal([1+2i 3+4i], [1+2i 3+4i]));
assert(~isequal([1+2i 3+4i], [1+2i 3+5i]));

% Tensor with different imaginary parts
assert(~isequal([1+1i 2+0i], [1 2]));

% Cell arrays
assert(isequal({1, 'a'}, {1, 'a'}));
assert(~isequal({1, 'a'}, {1, 'b'}));
assert(~isequal({1}, {1, 2}));

% Nested cells
assert(isequal({1, {2, 3}}, {1, {2, 3}}));
assert(~isequal({1, {2, 3}}, {1, {2, 4}}));

% Complex numbers
assert(isequal(1+2i, 1+2i));
assert(~isequal(1+2i, 1+3i));
assert(~isequal(1+2i, 2+2i));

% Cross-type comparisons
assert(~isequal(1, 'a'));

% Structs (reference equality)
s = struct('a', 1);
assert(isequal(s, s));

disp('SUCCESS');

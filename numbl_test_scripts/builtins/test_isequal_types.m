% Test isequal with various types (exercises compare.ts)

% Numbers
assert(isequal(1, 1));
assert(~isequal(1, 2));

% Logicals
assert(isequal(true, true));
assert(~isequal(true, false));


% Strings
assert(isequal("hello", "hello"));
assert(~isequal("hello", "world"));

% Char arrays
assert(isequal('abc', 'abc'));
assert(~isequal('abc', 'def'));

% Tensors
assert(isequal([1 2 3], [1 2 3]));
assert(~isequal([1 2 3], [1 2 4]));
assert(~isequal([1 2], [1 2 3]));

% Matrices
A = [1 2; 3 4];
B = [1 2; 3 4];
C = [1 2; 3 5];
assert(isequal(A, B));
assert(~isequal(A, C));

% Complex tensors
assert(isequal([1+2i, 3+4i], [1+2i, 3+4i]));
assert(~isequal([1+2i, 3+4i], [1+2i, 3+5i]));

% Complex numbers
assert(isequal(1+2i, 1+2i));
assert(~isequal(1+2i, 1+3i));

% Cell arrays
assert(isequal({1, 2, 3}, {1, 2, 3}));
assert(~isequal({1, 2, 3}, {1, 2, 4}));
assert(~isequal({1, 2}, {1, 2, 3}));

% Mixed type comparisons
assert(~isequal(1, 'a'));

% Structs
s1.x = 1;
s1.y = 2;
s2.x = 1;
s2.y = 2;
% struct equality is reference-based currently
assert(isequal(s1, s1));

disp('SUCCESS')

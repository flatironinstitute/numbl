% Test isequal with cell arrays, structs, and complex numbers

% Cell arrays with mixed types
assert(isequal({[1, 2], 'a', 3}, {[1, 2], 'a', 2+1}));
assert(~isequal({1, 2}, {1, 3}));
assert(~isequal({1, 2, 3}, {1; 2; 3}));

% Nested cell arrays
assert(isequal({{1, 2}, 'x'}, {{1, 2}, 'x'}));
assert(~isequal({{1, 2}}, {{1, 3}}));

% Structs
s1.a = 1; s1.b = 'hi';
s2.a = 1; s2.b = 'hi';
assert(isequal(s1, s2));
s3.a = 1; s3.b = 'bye';
assert(~isequal(s1, s3));

% Complex numbers
assert(isequal(1+2i, 1+2i));
assert(~isequal(1+2i, 1+3i));

disp('SUCCESS');

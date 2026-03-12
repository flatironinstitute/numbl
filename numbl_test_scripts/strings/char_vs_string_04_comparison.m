% Test 4: Comparison operators - char vs string
% Char == char is element-wise (like numeric arrays).
% String == string is a scalar logical.

% char == char: element-wise logical array
r1 = ('hello' == 'hello');
assert(isequal(r1, [1 1 1 1 1]));

% h==w, e==o, l==r, l==l, o==d
r2 = ('hello' == 'world');
assert(isequal(r2, [0 0 0 1 0]));

% char ~= char: element-wise
r3 = ('abc' ~= 'abc');
assert(isequal(r3, [0 0 0]));

% string == string: scalar logical
assert("hello" == "hello");
assert(~("hello" == "world"));

% string ~= string: scalar logical
assert("hello" ~= "world");
assert(~("hello" ~= "hello"));

disp('SUCCESS')

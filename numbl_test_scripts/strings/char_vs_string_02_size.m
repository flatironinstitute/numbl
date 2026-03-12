% Test 2: size / length / numel / strlength
% In MATLAB, 'hello' is a 1x5 char array; "hello" is a 1x1 string scalar.

c = 'hello';
s = "hello";

% size
assert(isequal(size(c), [1, 5]));
assert(isequal(size(s), [1, 1]));

% length (max dimension)
assert(length(c) == 5);
assert(length(s) == 1);

% numel
assert(numel(c) == 5);
assert(numel(s) == 1);

% strlength (counts characters regardless of char vs string)
assert(strlength(c) == 5);
assert(strlength(s) == 5);

% empty char: '' is a 0x0 char array in MATLAB
assert(isequal(size(''), [0, 0]));
assert(length('') == 0);
assert(strlength('') == 0);

% empty string: "" is a 1x1 string scalar
assert(isequal(size(""), [1, 1]));
assert(length("") == 1);
assert(strlength("") == 0);

disp('SUCCESS')

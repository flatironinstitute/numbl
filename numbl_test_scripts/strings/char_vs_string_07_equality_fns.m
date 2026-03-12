% Test 7: Equality functions - isempty, isequal, strcmp with mixed types

% isempty: '' is 0x0 char (empty), "" is 1x1 string (never empty)
assert(isempty(''));
assert(~isempty(""));
assert(~isempty('hello'));
assert(~isempty("hello"));

% isequal: in MATLAB, compares by value; char and string with same
% text are considered equal
assert(isequal('hello', 'hello'));
assert(isequal("hello", "hello"));
assert(isequal('hello', "hello"));   % same text -> equal

% strcmp is type-agnostic: compares by value
assert(strcmp('hello', 'hello'));
assert(strcmp("hello", "hello"));
assert(strcmp('hello', "hello"));
assert(~strcmp('hello', 'world'));

% strcmpi likewise
assert(strcmpi('Hello', "hello"));
assert(strcmpi("WORLD", 'world'));

disp('SUCCESS')

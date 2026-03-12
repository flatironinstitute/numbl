% Test string manipulation functions

% lower/upper
assert(strcmp(lower('HELLO'), 'hello'));
assert(strcmp(upper('world'), 'WORLD'));

% strtrim
assert(strcmp(strtrim('  hello  '), 'hello'));

% length of string
assert(length('hello') == 5);

% Concatenation with []
s = ['foo', 'bar'];
assert(strcmp(s, 'foobar'));

% num2str with format
s1 = num2str(3.14159, '%.2f');
assert(strcmp(s1, '3.14'));

% String comparison (case sensitive)
assert(strcmp('abc', 'abc'));
assert(~strcmp('abc', 'ABC'));

% strcmpi (case insensitive)
assert(strcmpi('Hello', 'hello'));
assert(strcmpi('ABC', 'abc'));

disp('SUCCESS')

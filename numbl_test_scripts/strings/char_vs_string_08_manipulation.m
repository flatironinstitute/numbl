% Test 8: String manipulation functions with char and string inputs
% Most functions accept both char and string.

% strfind works on both
assert(isequal(strfind('hello world', 'l'), [3 4 10]));
assert(isequal(strfind("hello world", "l"), [3 4 10]));
assert(isequal(strfind('hello world', "l"), [3 4 10]));  % mixed

% strrep works on both
assert(strcmp(strrep('hello world', 'world', 'there'), 'hello there'));
assert(strcmp(strrep("hello world", "world", "there"), "hello there"));

% strtrim works on both, preserves type
c1 = strtrim('  hello  ');
assert(ischar(c1));
assert(strcmp(c1, 'hello'));

s1 = strtrim("  hello  ");
assert(isstring(s1));
assert(strcmp(s1, "hello"));

% num2str produces char
n = num2str(42);
assert(ischar(n));
assert(strcmp(n, '42'));

% str2double works on both char and string
assert(str2double('3.14') - 3.14 < 1e-10);
assert(str2double("3.14") - 3.14 < 1e-10);

% str2num works on both
assert(str2num('42') == 42);

disp('SUCCESS')

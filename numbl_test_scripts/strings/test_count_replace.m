% Test count and replace string functions

%% count - basic
assert(count('hello world hello', 'hello') == 2);
assert(count('abcabc', 'abc') == 2);
assert(count('aaa', 'a') == 3);
assert(count('hello', 'xyz') == 0);
assert(count('', 'a') == 0);
assert(count('hello', '') == 6);  % empty pattern matches length+1 positions

%% count - overlapping (MATLAB count does NOT count overlapping)
assert(count('aaa', 'aa') == 1);

%% replace - basic
assert(strcmp(replace('hello world', 'world', 'MATLAB'), 'hello MATLAB'));
assert(strcmp(replace('aaa', 'a', 'bb'), 'bbbbbb'));
assert(strcmp(replace('hello', 'xyz', 'abc'), 'hello'));

%% replace - multiple occurrences
assert(strcmp(replace('abcabc', 'abc', 'x'), 'xx'));

%% replace - empty replacement (deletion)
assert(strcmp(replace('hello world', ' world', ''), 'hello'));

disp('SUCCESS');

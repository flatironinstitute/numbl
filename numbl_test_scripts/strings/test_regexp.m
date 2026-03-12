% Test regexp, regexpi, regexprep

%% regexp basic match
str = 'The quick brown fox jumps over the lazy dog';
[tok] = regexp(str, '\w+', 'match');
assert(length(tok) == 9);
assert(strcmp(tok{1}, 'The'));
assert(strcmp(tok{end}, 'dog'));

%% regexp start indices
idx = regexp(str, '\w+');
assert(idx(1) == 1);
assert(idx(2) == 5);

%% regexp tokens
str2 = 'abc123def456';
tok = regexp(str2, '([a-z]+)(\d+)', 'tokens');
assert(length(tok) == 2);
assert(strcmp(tok{1}{1}, 'abc'));
assert(strcmp(tok{1}{2}, '123'));
assert(strcmp(tok{2}{1}, 'def'));
assert(strcmp(tok{2}{2}, '456'));

%% regexp with 'once'
str3 = 'hello world hello';
m = regexp(str3, 'hello', 'match', 'once');
assert(strcmp(m, 'hello'));

%% regexp start/end
[s, e] = regexp(str2, '\d+');
assert(s(1) == 4);
assert(e(1) == 6);
assert(s(2) == 10);
assert(e(2) == 12);

%% regexpi (case insensitive)
str4 = 'Hello WORLD hello';
tok = regexpi(str4, 'hello', 'match');
assert(length(tok) == 2);
assert(strcmp(tok{1}, 'Hello'));
assert(strcmp(tok{2}, 'hello'));

%% regexprep
result = regexprep('hello world', 'world', 'MATLAB');
assert(strcmp(result, 'hello MATLAB'));

result = regexprep('aaa bbb ccc', '\s+', '-');
assert(strcmp(result, 'aaa-bbb-ccc'));

%% regexprep case insensitive
result = regexprep('Hello hello HELLO', 'hello', 'hi', 'ignorecase');
assert(strcmp(result, 'hi hi hi'));

%% regexp no match returns empty
idx = regexp('abc', 'xyz');
assert(isempty(idx));

tok = regexp('abc', 'xyz', 'match');
assert(isempty(tok));

disp('SUCCESS');

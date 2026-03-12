% String operations

% strcmp
assert(strcmp('hello', 'hello'))
assert(~strcmp('hello', 'world'))

% strcmpi (case-insensitive)
assert(strcmpi('Hello', 'hello'))
assert(~strcmpi('Hello', 'world'))

% length of string
s = 'abcde';
assert(length(s) == 5)

% strcat
result = strcat('foo', 'bar');
assert(strcmp(result, 'foobar'))

% num2str
s2 = num2str(42);
assert(strcmp(s2, '42'))

% str2num / str2double
n = str2double('3.14');
assert(abs(n - 3.14) < 0.01)

% upper / lower
assert(strcmp(upper('hello'), 'HELLO'))
assert(strcmp(lower('WORLD'), 'world'))

% strtrim
assert(strcmp(strtrim('  hi  '), 'hi'))

% strsplit - whitespace splitting (no delimiter)
C = strsplit('hello world');
assert(numel(C) == 2)
assert(strcmp(C{1}, 'hello'))
assert(strcmp(C{2}, 'world'))

% strsplit - consecutive whitespace treated as one
C2 = strsplit('a  b   c');
assert(numel(C2) == 3)
assert(strcmp(C2{1}, 'a'))
assert(strcmp(C2{2}, 'b'))
assert(strcmp(C2{3}, 'c'))

% strsplit - explicit delimiter
C3 = strsplit('Hello,world', ',');
assert(numel(C3) == 2)
assert(strcmp(C3{1}, 'Hello'))
assert(strcmp(C3{2}, 'world'))

% strsplit - consecutive delimiters treated as one
C4 = strsplit('Hello,,,world', ',');
assert(numel(C4) == 2)
assert(strcmp(C4{1}, 'Hello'))
assert(strcmp(C4{2}, 'world'))

% strjoin - join with space (default)
s1 = strjoin({'hello', 'world'});
assert(strcmp(s1, 'hello world'))

% strjoin - join with explicit delimiter
s2 = strjoin({'a', 'b', 'c'}, ',');
assert(strcmp(s2, 'a,b,c'))

% strjoin - join with multi-char delimiter
s3 = strjoin({'one', 'two', 'three'}, ' and ');
assert(strcmp(s3, 'one and two and three'))

% strcmp with cell containing non-string elements — should return false, not error
c = {42, 'polar', struct('x', 1)};
r = strcmp(c, 'polar');
assert(r(1) == 0);
assert(r(2) == 1);
assert(r(3) == 0);

% strcmpi with cell containing non-string elements
r2 = strcmpi(c, 'POLAR');
assert(r2(1) == 0);
assert(r2(2) == 1);
assert(r2(3) == 0);

disp('SUCCESS')

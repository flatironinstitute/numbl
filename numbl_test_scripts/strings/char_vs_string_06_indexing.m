% Test 6: Indexing
% Char indexing returns chars; string indexing returns the whole string.

c = 'hello';

% single index returns a 1x1 char
assert(ischar(c(1)));
assert(strcmp(c(1), 'h'));
assert(strcmp(c(5), 'o'));

% range indexing returns a char slice
assert(strcmp(c(2:4), 'ell'));
assert(ischar(c(2:4)));

% end keyword
assert(strcmp(c(end), 'o'));

% string indexing: s(1) returns the whole string
s = "hello";
assert(isstring(s(1)));
assert(strcmp(s(1), "hello"));

disp('SUCCESS')

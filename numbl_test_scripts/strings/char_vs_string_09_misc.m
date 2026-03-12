% Test 9: sprintf, for-loop over char, cell arrays with mixed types

% sprintf output type follows the format string type
r1 = sprintf('%s', 'hello');    % char format -> char output
assert(ischar(r1));
assert(strcmp(r1, 'hello'));

r2 = sprintf('%d items', 3);   % char format -> char output
assert(ischar(r2));
assert(strcmp(r2, '3 items'));

r3 = sprintf("%s", 'hello');   % string format -> string output
assert(isstring(r3));
assert(strcmp(r3, 'hello'));

% for-loop over char: each iteration gives a 1x1 char
result = '';
for c = 'abc'
  assert(ischar(c));
  assert(length(c) == 1);
  result = [result, c];
end
assert(strcmp(result, 'abc'));

% cell arrays can mix char and string
C = {'hello', "world"};
assert(ischar(C{1}));
assert(isstring(C{2}));
assert(strcmp(C{1}, 'hello'));
assert(strcmp(C{2}, "world"));

disp('SUCCESS')

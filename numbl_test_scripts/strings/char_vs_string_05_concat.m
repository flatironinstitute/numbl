% Test 5: Concatenation
% [] on char arrays: concatenates into a longer char array.
% strcat on char: strips trailing whitespace from each char argument.
% strcat with any string: preserves whitespace, returns string.

% [] char concatenation produces a char
c = ['foo', 'bar'];
assert(ischar(c));
assert(strcmp(c, 'foobar'));
assert(length(c) == 6);

% strcat on char: no trailing whitespace in args -> normal concat
c2 = strcat('foo', 'bar');
assert(ischar(c2));
assert(strcmp(c2, 'foobar'));

% strcat on char: trailing whitespace is stripped
c3 = strcat('foo ', 'bar');
assert(strcmp(c3, 'foobar'));

c4 = strcat('foo', ' bar');
assert(strcmp(c4, 'foo bar'));  % leading space in second arg preserved

% strcat with a string arg: whitespace preserved, result is string
s1 = strcat("foo ", 'bar');
assert(isstring(s1));
assert(strcmp(s1, 'foo bar'));

disp('SUCCESS')

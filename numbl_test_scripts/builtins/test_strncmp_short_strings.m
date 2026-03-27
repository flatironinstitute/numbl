% Test strncmp/strncmpi when strings are shorter than n
% MATLAB: strncmp('hi','hi',5) returns true since both strings match

assert(strncmp('hi', 'hi', 5) == 1, 'strncmp: identical short strings should match');
assert(strncmp('hi', 'hi', 2) == 1, 'strncmp: exact length match');
assert(strncmp('hi', 'hiya', 2) == 1, 'strncmp: prefix match');
assert(strncmp('hi', 'hiya', 5) == 0, 'strncmp: different lengths, n>both');
assert(strncmp('hello', 'hello', 3) == 1, 'strncmp: longer strings prefix');

% strncmpi (case-insensitive)
assert(strncmpi('HI', 'hi', 5) == 1, 'strncmpi: identical short strings case insensitive');
assert(strncmpi('Hello', 'HELLO', 3) == 1, 'strncmpi: prefix match case insensitive');

disp('SUCCESS');

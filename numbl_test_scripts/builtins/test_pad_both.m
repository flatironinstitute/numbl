% Test pad with 'both' side option
% MATLAB: pad(str, n, 'both') centers the string

s = pad('hi', 10, 'both');
assert(length(s) == 10, 'pad both: wrong length');
assert(strcmp(s, '    hi    '), 'pad both: wrong result');

% Odd padding distributes floor on left, ceil on right
s2 = pad('hi', 9, 'both');
assert(length(s2) == 9, 'pad both odd: wrong length');
assert(strcmp(s2, '   hi    '), 'pad both odd: wrong result');

% Left and right should still work
assert(strcmp(pad('hi', 5, 'left'), '   hi'), 'pad left');
assert(strcmp(pad('hi', 5, 'right'), 'hi   '), 'pad right');
assert(strcmp(pad('hi', 5), 'hi   '), 'pad default is right');

disp('SUCCESS');

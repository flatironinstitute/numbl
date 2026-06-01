% %s with a numeric argument interprets the value(s) as character codes
% (MATLAB: sprintf('%s', 65) -> 'A').

assert(strcmp(sprintf('%s', 65), 'A'));
assert(strcmp(sprintf('%s', [72 73]), 'HI'));
assert(strcmp(sprintf('%d:%s', 65, 66), '65:B'));

% a char/string argument is still printed as-is
assert(strcmp(sprintf('%s', 'HI'), 'HI'));

% single numeric element with a trailing literal
assert(strcmp(sprintf('%s!', 65), 'A!'));

disp('SUCCESS')

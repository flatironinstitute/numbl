% Test sprintf %x (hex), %X (upper hex), and %o (octal)

% Hexadecimal lowercase
assert(strcmp(sprintf('%x', 255), 'ff'), 'sprintf %x 255');
assert(strcmp(sprintf('%x', 0), '0'), 'sprintf %x 0');
assert(strcmp(sprintf('%x', 16), '10'), 'sprintf %x 16');

% Hexadecimal uppercase
assert(strcmp(sprintf('%X', 255), 'FF'), 'sprintf %X 255');

% Octal
assert(strcmp(sprintf('%o', 8), '10'), 'sprintf %o 8');
assert(strcmp(sprintf('%o', 0), '0'), 'sprintf %o 0');
assert(strcmp(sprintf('%o', 7), '7'), 'sprintf %o 7');
assert(strcmp(sprintf('%o', 255), '377'), 'sprintf %o 255');

disp('SUCCESS');

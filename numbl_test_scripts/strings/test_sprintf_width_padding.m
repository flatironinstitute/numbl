% Test sprintf width padding for integer formats
% %Nd should space-pad, %0Nd should zero-pad

% Space-padding (default)
assert(strcmp(sprintf('%10d', 42), '        42'));
assert(strcmp(sprintf('%5d', 7), '    7'));
assert(strcmp(sprintf('%3d', 100), '100'));  % exact width, no padding needed

% Zero-padding (explicit 0 flag)
assert(strcmp(sprintf('%010d', 42), '0000000042'));
assert(strcmp(sprintf('%05d', 7), '00007'));

% Left-justify overrides zero-pad
assert(strcmp(sprintf('%-10d', 42), '42        '));

% Negative numbers with width
assert(strcmp(sprintf('%10d', -42), '       -42'));
assert(strcmp(sprintf('%010d', -42), '-000000042'));

% Width with plus flag
assert(strcmp(sprintf('%+10d', 42), '       +42'));

disp('SUCCESS');

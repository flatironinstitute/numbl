% Test sprintf width specifiers and flags for strings and numbers

% Right-aligned string with width
s1 = sprintf('%10s', 'hi');
assert(length(s1) == 10, 'string width 10 length');
assert(strcmp(s1, '        hi'), 'string right-aligned width 10');

% Left-aligned string with width
s2 = sprintf('%-10s', 'hi');
assert(length(s2) == 10, 'left-aligned string width 10 length');
assert(strcmp(s2, 'hi        '), 'string left-aligned width 10');

% Width with longer string (no truncation)
s3 = sprintf('%3s', 'hello');
assert(strcmp(s3, 'hello'), 'string width shorter than content');

% Plus flag for positive integer
s4 = sprintf('%+d', 42);
assert(strcmp(s4, '+42'), 'plus flag positive integer');

% Plus flag for negative integer
s5 = sprintf('%+d', -42);
assert(strcmp(s5, '-42'), 'plus flag negative integer');

% Plus flag for positive float
s6 = sprintf('%+.2f', 3.14);
assert(strcmp(s6, '+3.14'), 'plus flag positive float');

% Plus flag for zero
s7 = sprintf('%+d', 0);
assert(strcmp(s7, '+0'), 'plus flag zero');

disp('SUCCESS');

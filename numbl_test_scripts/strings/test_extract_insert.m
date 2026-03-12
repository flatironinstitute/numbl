% Test extractBefore, extractAfter, extractBetween, insertBefore, insertAfter

%% extractBefore - position
assert(strcmp(extractBefore('hello world', 6), 'hello'));
assert(strcmp(extractBefore('hello world', 1), ''));

%% extractBefore - pattern
assert(strcmp(extractBefore('hello world', ' '), 'hello'));
assert(strcmp(extractBefore('hello world', 'world'), 'hello '));

%% extractAfter - position
assert(strcmp(extractAfter('hello world', 5), ' world'));
assert(strcmp(extractAfter('hello world', 11), ''));

%% extractAfter - pattern
assert(strcmp(extractAfter('hello world', ' '), 'world'));
assert(strcmp(extractAfter('hello world', 'hello'), ' world'));

%% extractBetween - positions
assert(strcmp(extractBetween('hello world', 2, 5), 'ello'));

%% extractBetween - patterns
assert(strcmp(extractBetween('hello [world] end', '[', ']'), 'world'));

%% insertBefore - position
assert(strcmp(insertBefore('hello world', 6, 'big '), 'hellobig  world'));

%% insertBefore - pattern
assert(strcmp(insertBefore('hello world', 'world', 'big '), 'hello big world'));

%% insertAfter - position
assert(strcmp(insertAfter('hello world', 5, ' big'), 'hello big world'));

%% insertAfter - pattern
assert(strcmp(insertAfter('hello world', 'hello', ' big'), 'hello big world'));

disp('SUCCESS');

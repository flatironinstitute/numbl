% Test erase and reverse

%% erase - basic
assert(strcmp(erase('hello world', 'world'), 'hello '));
assert(strcmp(erase('abcabc', 'abc'), ''));
assert(strcmp(erase('hello', 'xyz'), 'hello'));

%% erase - multiple occurrences
assert(strcmp(erase('banana', 'an'), 'ba'));

%% reverse
assert(strcmp(reverse('hello'), 'olleh'));
assert(strcmp(reverse('a'), 'a'));
assert(strcmp(reverse(''), ''));
assert(strcmp(reverse('abcdef'), 'fedcba'));

disp('SUCCESS');

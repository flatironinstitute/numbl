% Test fliplr on char arrays (strings)

% Basic string reversal
assert(strcmp(fliplr('hello'), 'olleh'));
assert(strcmp(fliplr('a'), 'a'));
assert(strcmp(fliplr('ab'), 'ba'));
assert(strcmp(fliplr(''), ''));

% Char array with spaces
assert(strcmp(fliplr('hello world'), 'dlrow olleh'));

% flipud on char row vector should be identity
assert(strcmp(flipud('hello'), 'hello'));

disp('SUCCESS');

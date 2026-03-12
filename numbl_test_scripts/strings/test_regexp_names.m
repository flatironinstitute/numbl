% Test regexp with 'names' option returning a struct

% Basic named groups
str = 'John Smith, 30';
tok = regexp(str, '(?<first>\w+)\s(?<last>\w+),\s(?<age>\d+)', 'names');
assert(isstruct(tok));
assert(strcmp(tok.first, 'John'));
assert(strcmp(tok.last, 'Smith'));
assert(strcmp(tok.age, '30'));

% Single named group with single match
tok2 = regexp('hello42world', '(?<num>\d+)', 'names');
assert(isstruct(tok2));
assert(strcmp(tok2.num, '42'));

% No match returns empty struct
tok3 = regexp('hello', '(?<num>\d+)', 'names');
assert(isempty(tok3));

disp('SUCCESS');

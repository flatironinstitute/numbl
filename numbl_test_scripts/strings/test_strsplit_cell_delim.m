% Test strsplit with cell array of delimiters

% Multiple delimiters
parts = strsplit('a;b,c', {';', ','});
assert(length(parts) == 3);
assert(strcmp(parts{1}, 'a'));
assert(strcmp(parts{2}, 'b'));
assert(strcmp(parts{3}, 'c'));

% Multiple delimiters with consecutive (collapsed)
parts = strsplit('a;,b,,c', {';', ','});
assert(length(parts) == 3);
assert(strcmp(parts{1}, 'a'));
assert(strcmp(parts{2}, 'b'));
assert(strcmp(parts{3}, 'c'));

% Single delimiter in cell (should behave same as string)
parts = strsplit('a-b-c', {'-'});
assert(length(parts) == 3);
assert(strcmp(parts{2}, 'b'));

% Mixed delimiter types
parts = strsplit('hello world;foo,bar', {' ', ';', ','});
assert(length(parts) == 4);
assert(strcmp(parts{1}, 'hello'));
assert(strcmp(parts{4}, 'bar'));

disp('SUCCESS');

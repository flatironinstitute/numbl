% Test strtok

%% Basic - default whitespace delimiter
[tok, rem] = strtok('hello world');
assert(strcmp(tok, 'hello'));
assert(strcmp(rem, ' world'));

%% Leading whitespace
[tok, rem] = strtok('  hello world');
assert(strcmp(tok, 'hello'));
assert(strcmp(rem, ' world'));

%% Single word
[tok, rem] = strtok('hello');
assert(strcmp(tok, 'hello'));
assert(strcmp(rem, ''));

%% Custom delimiter
[tok, rem] = strtok('hello,world', ',');
assert(strcmp(tok, 'hello'));
assert(strcmp(rem, ',world'));

%% Multiple delimiters
[tok, rem] = strtok('hello::world', ':');
assert(strcmp(tok, 'hello'));
assert(strcmp(rem, '::world'));

%% Empty string
[tok, rem] = strtok('');
assert(strcmp(tok, ''));
assert(strcmp(rem, ''));

%% Only delimiters
[tok, rem] = strtok('   ');
assert(strcmp(tok, ''));
assert(strcmp(rem, ''));

%% Single output
tok = strtok('hello world');
assert(strcmp(tok, 'hello'));

disp('SUCCESS');

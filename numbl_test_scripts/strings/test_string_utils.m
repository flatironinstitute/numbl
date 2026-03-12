% Test deblank, strip, pad, blanks, startsWith, endsWith, mat2str

%% deblank - remove trailing whitespace
assert(strcmp(deblank('hello   '), 'hello'));
assert(strcmp(deblank('hello'), 'hello'));
assert(strcmp(deblank('  hello  '), '  hello'));
assert(strcmp(deblank(''), ''));

%% blanks
assert(isempty(blanks(0)));
assert(strcmp(blanks(3), '   '));
assert(length(blanks(5)) == 5);

%% strip - remove leading and trailing whitespace
assert(strcmp(strip('  hello  '), 'hello'));
assert(strcmp(strip('hello'), 'hello'));
assert(strcmp(strip('  hello'), 'hello'));
assert(strcmp(strip('hello  '), 'hello'));

%% strip with side argument
assert(strcmp(strip('  hello  ', 'left'), 'hello  '));
assert(strcmp(strip('  hello  ', 'right'), '  hello'));
assert(strcmp(strip('  hello  ', 'both'), 'hello'));

%% pad
assert(strcmp(pad('hello', 10), 'hello     '));
assert(length(pad('hello', 10)) == 10);
assert(strcmp(pad('hello', 10, 'left'), '     hello'));
assert(strcmp(pad('hello', 10, 'right'), 'hello     '));
assert(strcmp(pad('hello', 3), 'hello'));  % no truncation

%% startsWith
assert(startsWith('hello world', 'hello'));
assert(~startsWith('hello world', 'world'));
assert(startsWith('hello', ''));
assert(startsWith('hello world', 'hello w'));

%% endsWith
assert(endsWith('hello world', 'world'));
assert(~endsWith('hello world', 'hello'));
assert(endsWith('hello', ''));
assert(endsWith('hello world', 'o world'));

%% mat2str
assert(strcmp(mat2str(3), '3'));
assert(strcmp(mat2str([1 2; 3 4]), '[1 2;3 4]'));
assert(strcmp(mat2str([1 2 3]), '[1 2 3]'));

%% mat2str with precision
assert(strcmp(mat2str(pi, 4), '3.142'));

disp('SUCCESS');

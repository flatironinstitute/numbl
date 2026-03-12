% Test 10: Type conversion functions and edge cases

% char() on numeric -> char array
assert(strcmp(char(65), 'A'));
assert(strcmp(char([72 101 108 108 111]), 'Hello'));

% string() on char -> string scalar
s = string('hello');
assert(isstring(s));
assert(strcmp(s, "hello"));

% char() on string -> char array
c = char("world");
assert(ischar(c));
assert(strcmp(c, 'world'));

% Mixing char and string in strcat: result is string (any string input -> string output)
r = strcat('prefix_', "suffix");
assert(isstring(r));
assert(strcmp(r, 'prefix_suffix'));

% int2str returns char
n = int2str(42);
assert(ischar(n));
assert(strcmp(n, '42'));

% num2str returns char
n2 = num2str(3.14, '%.2f');
assert(ischar(n2));
assert(strcmp(n2, '3.14'));

disp('SUCCESS')

% Test that + on MATLAB strings does concatenation (not numeric
% addition).  This is relied on by chunkie/chunkerkerneval which builds
% error messages like
%   "CHUNKERKERNEVAL: second input not of appropriate shape " + ...
%   "number of rows in kern should be 1 or nregion"

% --- string + string ---
s1 = "hello " + "world";
assert(isstring(s1), 's1 should be string');
assert(strcmp(s1, 'hello world'), 's1 value');

% Multi-line continuation style (what chunkie uses)
s2 = "first part " + ...
     "second part";
assert(strcmp(s2, 'first part second part'), 's2 value');

% Empty string edges
assert(strcmp("" + "abc", 'abc'), 'empty left');
assert(strcmp("abc" + "", 'abc'), 'empty right');
assert(strcmp("" + "", ''), 'empty both');

% --- string + char / char + string  → string ---
sc1 = "pre:" + 'suf';
assert(isstring(sc1), 'sc1 should be string');
assert(strcmp(sc1, 'pre:suf'), 'sc1 value');

sc2 = 'pre:' + "suf";
assert(isstring(sc2), 'sc2 should be string');
assert(strcmp(sc2, 'pre:suf'), 'sc2 value');

% --- string + number → string (uses num2str-style formatting) ---
n1 = "int: " + 42;
assert(isstring(n1), 'n1 should be string');
assert(strcmp(n1, 'int: 42'), 'n1 value');

n2 = "neg: " + -7;
assert(strcmp(n2, 'neg: -7'), 'n2 value');

n3 = "float: " + 3.14;
assert(strcmp(n3, 'float: 3.14'), 'n3 value');

n4 = "zero: " + 0;
assert(strcmp(n4, 'zero: 0'), 'n4 value');

% --- number + string → string ---
n5 = 5 + "x";
assert(isstring(n5), 'n5 should be string');
assert(strcmp(n5, '5x'), 'n5 value');

% --- string + logical → string (formatted as "true"/"false") ---
b1 = "b=" + true;
assert(isstring(b1), 'b1 should be string');
assert(strcmp(b1, 'b=true'), 'b1 value');

b2 = "b=" + false;
assert(strcmp(b2, 'b=false'), 'b2 value');

% --- logical + string ---
b3 = true + " is true";
assert(strcmp(b3, 'true is true'), 'b3 value');

% --- chained concatenation ---
s3 = "a" + "b" + "c" + "d";
assert(strcmp(s3, 'abcd'), 's3 chain value');

% --- string + does NOT leak into char + char (char arithmetic preserved) ---
ca = 'A' + 1;  % must stay numeric (66)
assert(isnumeric(ca), 'char+num still numeric');
assert(ca == 66, 'char+num value');

disp('SUCCESS');

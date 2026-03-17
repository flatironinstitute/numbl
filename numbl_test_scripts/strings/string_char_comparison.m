% Test: string vs char comparison operators
% In MATLAB, when a string is compared with a char, the char is converted
% to a string and scalar string comparison is performed.

% == string vs char
assert("abc" == 'abc');
assert(~("abc" == 'abd'));

% ~= string vs char
assert("abc" ~= 'abd');
assert(~("abc" ~= 'abc'));

% char vs string (reversed operand order)
assert('abc' == "abc");
assert('abc' ~= "abd");

% < > <= >= with string vs char
assert("abc" < 'abd');
assert(~("abd" < 'abc'));

assert("abd" > 'abc');
assert(~("abc" > 'abd'));

assert("abc" <= 'abc');
assert("abc" <= 'abd');

assert("abc" >= 'abc');
assert("abd" >= 'abc');

% Single char string vs single char
assert("a" == 'a');
assert("a" ~= 'b');
assert("a" < 'b');

% Empty string vs empty char
assert("" == '');
assert(~("" ~= ''));

disp('SUCCESS')

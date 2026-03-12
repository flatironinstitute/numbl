% Test 3: Char arithmetic
% In MATLAB, chars are numeric under the hood (UTF-16 code units).
% Arithmetic on char produces numeric results.

% double() converts char to its numeric code
assert(double('A') == 65);
assert(double('a') == 97);
assert(isequal(double('ABC'), [65 66 67]));

% char() converts numeric code back to char
assert(strcmp(char(65), 'A'));
assert(strcmp(char([72 101 108 108 111]), 'Hello'));

% char + number gives a number
assert(('A' + 1) == 66);

% char + 0 on a multi-char gives a row vector of codes
v = 'hi' + 0;
assert(isequal(v, [104 105]));

% arithmetic produces numbers, not chars
assert(~ischar('A' + 0));
assert(isnumeric('A' + 0));

disp('SUCCESS')

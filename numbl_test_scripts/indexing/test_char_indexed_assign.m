% Indexed assignment into a char array must keep the char class and
% interpret a numeric RHS as a character code (MATLAB semantics).

x = 'abc'; x(2) = 66;
assert(ischar(x), 'class should stay char');
assert(strcmp(x, 'aBc'));

x = 'abc'; x(2) = 'Z';
assert(strcmp(x, 'aZc'));

x = 'abc'; x(1) = 120;
assert(strcmp(x, 'xbc'));

% element deletion
x = 'abcde'; x(2) = [];
assert(strcmp(x, 'acde'));

% 2-D char element assignment
x = ['ab'; 'cd']; x(1,2) = 'X';
assert(isequal(x, ['aX'; 'cd']));

% scalar auto-grow fills the gap with char(0) and stays char
x = 'ab'; x(5) = 'e';
assert(ischar(x));
assert(isequal(double(x), [97 98 0 0 101]));

% vector RHS
x = 'abc'; x(1:2) = 'XY';
assert(strcmp(x, 'XYc'));

disp('SUCCESS')

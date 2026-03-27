% Test input() builtin - string mode
txt = input('Enter name: ', 's');
assert(ischar(txt), 'input with "s" should return char');
disp(txt);

% Test input() with expression mode
x = input('Enter value: ');
assert(isnumeric(x), 'input without "s" should evaluate expression');
disp(x);

disp('SUCCESS');

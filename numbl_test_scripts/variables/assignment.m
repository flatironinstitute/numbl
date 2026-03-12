% Variable assignment and usage

x = 5;
y = 3;
z = x + y;
assert(z == 8)

% Reassignment
x = 10;
assert(x == 10)

% Multiple assignments
a = 1; b = 2; c = a + b;
assert(c == 3)

% String assignment (basic display, no assert needed yet)
name = 'hello';
assert(strcmp(name, 'hello'))

disp('SUCCESS')

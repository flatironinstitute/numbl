% Assigning a char into a numeric array converts the char to its code(s)
% and leaves the array numeric (double) — MATLAB semantics.

x = [1 2 3]; x(2) = 'A';
assert(isequal(x, [1 65 3]));
assert(isa(x, 'double'));

% vector char RHS
x = [1 2 3 4]; x(2:3) = 'AB';
assert(isequal(x, [1 65 66 4]));

% auto-grow with char RHS (gap filled with 0)
x = [1 2]; x(4) = 'A';
assert(isequal(x, [1 2 0 65]));

disp('SUCCESS')

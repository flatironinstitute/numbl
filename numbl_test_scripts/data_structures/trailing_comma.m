% Test trailing commas in matrices and cell arrays

% Trailing comma before semicolon in matrix
y = [1, ; 2; 3];
assert(isequal(y, [1; 2; 3]));

% Trailing comma before semicolon in cell array
x = {'a', ; 'b'; 'c'};
assert(isequal(size(x), [3, 1]));
assert(strcmp(x{1}, 'a'));
assert(strcmp(x{2}, 'b'));
assert(strcmp(x{3}, 'c'));

% Trailing comma before closing bracket
z = [4, 5, 6,];
assert(isequal(z, [4, 5, 6]));

% Trailing comma before closing brace
w = {'x', 'y', 'z',};
assert(isequal(size(w), [1, 3]));
assert(strcmp(w{1}, 'x'));
assert(strcmp(w{2}, 'y'));
assert(strcmp(w{3}, 'z'));

disp('SUCCESS');

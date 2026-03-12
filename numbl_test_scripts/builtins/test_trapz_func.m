% Test trapz builtin

% Basic trapz with uniform spacing
y = [1 2 3 4 5];
assert(abs(trapz(y) - 12) < 1e-10);

% trapz with x values
x = [0 1 2 3 4];
y2 = [1 4 9 16 25];
result = trapz(x, y2);
assert(abs(result - 42) < 1e-10);

% trapz with non-uniform spacing
x2 = [0 0.5 1 2];
y3 = [0 0.25 1 4];
result2 = trapz(x2, y3);
assert(abs(result2 - 2.875) < 1e-10);

% Single element
assert(trapz([5]) == 0);

% Two elements
assert(abs(trapz([3, 7]) - 5) < 1e-10);

% Column vector
y4 = [1; 2; 3; 4];
assert(abs(trapz(y4) - 7.5) < 1e-10);

disp('SUCCESS');

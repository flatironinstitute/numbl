% Test gradient builtin

% 1-D gradient with default spacing
y = [1 2 4 7 11];
g = gradient(y);
assert(isequal(g, [1 1.5 2.5 3.5 4]));

% 1-D gradient with custom spacing
g2 = gradient(y, 2);
assert(isequal(g2, [0.5 0.75 1.25 1.75 2]));

% 1-D gradient with non-uniform x
x = [0 1 3 5 8];
g3 = gradient(y, x);
expected = [1.0000 1.0000 1.2500 1.4000 1.3333];
assert(norm(g3 - expected) < 0.001);

% Short vectors
g4 = gradient([3 7]);
assert(isequal(g4, [4 4]));

% Column vector
y2 = [1; 4; 9; 16];
g5 = gradient(y2);
assert(isequal(g5, [3; 4; 6; 7]));

disp('SUCCESS');

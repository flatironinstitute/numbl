% Test class defined in @Point folder with separate method files

% Create Point instances
p1 = Point(3, 4);
assert(p1.x == 3);
assert(p1.y == 4);

p2 = Point(0, 0);
assert(p2.x == 0);
assert(p2.y == 0);

% Test distance method (defined in separate file)
d = p1.distance(p2);
assert(d == 5);  % 3-4-5 triangle

% Test translate method (defined in separate file)
p3 = p1.translate(1, 1);
assert(p3.x == 4);
assert(p3.y == 5);

% Distance should now be different
d2 = p3.distance(p2);
assert(abs(d2 - sqrt(41)) < 1e-5);  % sqrt(4^2 + 5^2) = sqrt(41)

disp('SUCCESS')

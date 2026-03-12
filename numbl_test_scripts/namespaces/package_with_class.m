% Test class inside a package

% Create Circle from geometry package
c = geometry.Circle(5);
assert(c.radius == 5);

% Test area method
a = c.area();
expected_area = pi * 25;
assert(abs(a - expected_area) < 1e-5);

% Test circumference method
circ = c.circumference();
expected_circ = 2 * pi * 5;
assert(abs(circ - expected_circ) < 1e-5);

% Create another circle
c2 = geometry.Circle(10);
assert(c2.radius == 10);
assert(abs(c2.area() - pi * 100) < 1e-5);

disp('SUCCESS')

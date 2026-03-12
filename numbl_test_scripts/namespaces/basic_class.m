% Test basic classdef class

% Create Rectangle instance
r = Rectangle_(5, 10);
assert(r.width == 5);
assert(r.height == 10);

% Call methods
a = r.area();
assert(a == 50);

p = r.perimeter();
assert(p == 30);

% Create another instance
r2 = Rectangle_(3, 4);
assert(r2.area() == 12);
assert(r2.perimeter() == 14);

disp('SUCCESS')

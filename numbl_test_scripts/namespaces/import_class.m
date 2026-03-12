% Test import of namespaced class
import geometry.Circle

c = Circle(5);
assert(c.radius == 5);
assert(abs(c.area() - pi * 25) < 1e-10);

disp('SUCCESS')

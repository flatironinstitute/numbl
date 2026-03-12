% Test basic abstract class functionality

% Create a concrete subclass instance
r = ConcreteRect_(5, 10, 'red');

% Test inherited property
assert(strcmp(r.Color, 'red'));

% Test implemented abstract methods
assert(r.area() == 50);
assert(r.perimeter() == 30);

% Test inherited concrete method
assert(strcmp(r.describe(), 'red'));

disp('SUCCESS')

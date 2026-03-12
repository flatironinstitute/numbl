% Test superclass method and constructor calls

% Test 1: Super constructor call initializes base properties
d = Dog_('lab');
assert(strcmp(d.Name, 'dog'));
assert(d.Legs == 4);
assert(strcmp(d.Breed, 'lab'));

% Test 2: Super method call from overriding method
r = d.describe();
assert(r == 40);

disp('SUCCESS')

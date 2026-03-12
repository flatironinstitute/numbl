% Test that isa returns true for complex numbers when queried as 'double'
c = 3 + 4i;
assert(isa(c, 'double'), 'isa(complex, double) should be true');
assert(isa(c, 'numeric'), 'isa(complex, numeric) should be true');
assert(~isa(c, 'char'), 'isa(complex, char) should be false');

% Also test that class() returns 'double' for complex numbers
assert(strcmp(class(c), 'double'), 'class(complex) should be double');

disp('SUCCESS');

% Test int2str with vectors and matrices

% Vector
s = int2str([1.1 2.5 3.9]);
assert(contains(s, '1'));
assert(contains(s, '3'));
assert(contains(s, '4'));

% Scalar still works
assert(strcmp(int2str(3.7), '4'));
assert(strcmp(int2str(-2.3), '-2'));

% Negative vector
s = int2str([-1.1 2.9]);
assert(contains(s, '-1'));
assert(contains(s, '3'));

disp('SUCCESS');

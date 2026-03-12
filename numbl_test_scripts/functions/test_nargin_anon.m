% Test that nargin returns the correct count for anonymous functions

op1 = @(x) x + 1;
assert(nargin(op1) == 1, 'nargin of @(x) should be 1');

op2 = @(x, y) x + y;
assert(nargin(op2) == 2, 'nargin of @(x,y) should be 2');

op3 = @(x, y, z) x + y + z;
assert(nargin(op3) == 3, 'nargin of @(x,y,z) should be 3');

% Anonymous function with captured variable
c = 10;
op4 = @(x, t) exp(c*x.*t);
assert(nargin(op4) == 2, 'nargin of @(x,t) with capture should be 2');

% Nested anonymous function
op5 = @(a, b) op2(a, b);
assert(nargin(op5) == 2, 'nargin of nested anon should be 2');

disp('SUCCESS');

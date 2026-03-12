% Test that anonymous function parameters don't leak into the outer scope.
% In MATLAB, @(x) ... creates a local parameter x that does NOT overwrite
% any outer variable named x.

x = 42;
f = @(x) x + 1;

% Calling f should not change the outer x
y = f(10);
assert(y == 11, 'f(10) should be 11');
assert(x == 42, 'outer x should still be 42 after calling f(10)');

% Call again with a different value
z = f(100);
assert(z == 101, 'f(100) should be 101');
assert(x == 42, 'outer x should still be 42 after calling f(100)');

% Test with a more complex anonymous function
a = [1 2 3];
g = @(a) sum(a);
r = g([10 20 30]);
assert(r == 60, 'g([10 20 30]) should be 60');
assert(isequal(a, [1 2 3]), 'outer a should still be [1 2 3]');

disp('SUCCESS');

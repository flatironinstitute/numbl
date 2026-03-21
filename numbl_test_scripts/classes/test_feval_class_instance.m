% Test that feval(classInstance, args...) dispatches to the class's feval method.
% This pattern is used by chebfun: compose() creates anonymous functions
% like @(r,l,th) feval(op, feval(f, r, l, th, 'flag')) where f is a class
% instance with a custom feval method.

obj = FevalTarget_([10 20 30]);

% Test 1: feval(obj, 2) should call the class's feval method
result = feval(obj, 2);
assert(result == 20, 'feval(obj, idx) should call class feval method');

% Test 2: feval(obj, 3) — another index
result2 = feval(obj, 3);
assert(result2 == 30, 'feval(obj, 3) should return 30');

% Test 3: feval inside an anonymous function (the chebfun compose pattern)
g = @(idx) feval(obj, idx);
assert(g(1) == 10, 'feval(obj,...) inside anon func should work');

% Test 4: compose-like pattern — feval result fed to another function
h = @(idx) abs(feval(obj, idx));
assert(h(2) == 20, 'abs(feval(obj, idx)) should work');

disp('SUCCESS');

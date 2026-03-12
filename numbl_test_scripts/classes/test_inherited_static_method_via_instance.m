% Test that calling an INHERITED static method via a subclass instance
% does NOT prepend the instance as the first argument.
% This mirrors the chebfun pattern: chebtech2 inherits clenshaw from chebtech.

% Create a DerivedTech (subclass of BaseTech) with coeffs [3; 5]
obj = DerivedTech([3; 5]);

% Call inherited static method via instance:
% obj.evaluate(x, coeffs) should call evaluate(x, coeffs), NOT evaluate(obj, x, coeffs)
x = 2;
y = obj.evaluate(x, obj.coeffs);
% Expected: coeffs(1) + coeffs(2)*x = 3 + 5*2 = 13
assert(y == 13, sprintf('Direct call: expected 13, got %g', y));

% Also test via an instance method that delegates
y2 = obj.callEval(4);
% Expected: 3 + 5*4 = 23
assert(y2 == 23, sprintf('Via callEval: expected 23, got %g', y2));

% Call via class name should also work
y3 = BaseTech.evaluate(1, [10; 20]);
% Expected: 10 + 20*1 = 30
assert(y3 == 30, sprintf('Via BaseTech: expected 30, got %g', y3));

disp('SUCCESS');

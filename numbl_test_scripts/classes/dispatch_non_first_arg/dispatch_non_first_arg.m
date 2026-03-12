% Test: class method dispatch when class instance is not the first argument
% In MATLAB, method dispatch checks ALL arguments, not just the first.
% If any argument is a class instance that has the method, it dispatches.
%
% Processor_ has method process(op, values, obj) where obj is 3rd arg.

p = Processor_(3);

% Test 1: known type - process dispatches to Processor_ method
r1 = process(@(x) x + 1, 5, p);
assert(r1 == 18);  % (5+1) * 3 = 18

% Test 2: unknown type - should still dispatch at runtime
X = 0;
X = Processor_(4);
r2 = process(@(x) x * 2, 3, X);
assert(r2 == 24);  % (3*2) * 4 = 24

disp('SUCCESS')

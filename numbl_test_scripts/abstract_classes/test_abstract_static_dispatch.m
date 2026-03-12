% Test that abstract static methods dispatch correctly when called on an instance
% from within a parent class method (like chebfun's populate calling f.refine).
c = ConcreteCalc(10);
assert(c.value == 7, 'Expected c.value == 7');
disp('SUCCESS');

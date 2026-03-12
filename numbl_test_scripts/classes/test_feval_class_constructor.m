% Test that feval with a class name string constructs the class
techName = 'MyTech';
obj = feval(techName);
assert(isa(obj, 'MyTech'));
assert(obj.value == 42);

% Also test with constructor arguments
obj2 = feval('MyTech', 99);
assert(obj2.value == 99);

disp('SUCCESS');

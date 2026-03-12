% Test that a class constructor actually gets called
obj = SimpleCtorTest_(7);
assert(obj.Value == 14);
assert(obj.Initialized == 1);
disp('SUCCESS')

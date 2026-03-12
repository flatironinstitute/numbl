% Test that cell expansion (CSL) works for class constructor arguments
x = {5, 8};
obj = CslCtorArgs_(x{:});
assert(obj.A == 5);
assert(obj.B == 8);

disp('SUCCESS')

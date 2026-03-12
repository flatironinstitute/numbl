% Test multiple imports from different namespaces
import mymath.add_two
import utils.double_it

result1 = add_two(3, 5);
assert(result1 == 8);

result2 = double_it(7);
assert(result2 == 14);

disp('SUCCESS')

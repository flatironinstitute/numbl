% Test explicit import of namespace function
import mymath.add_two

result = add_two(3, 5);
assert(result == 8);

result2 = add_two(10, 20);
assert(result2 == 30);

disp('SUCCESS')

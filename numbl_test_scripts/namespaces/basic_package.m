% Test basic package (namespace) functionality

% Call function from +mymath package
result = mymath.add_two(3, 5);
assert(result == 8);

result2 = mymath.add_two(10, 20);
assert(result2 == 30);

disp('SUCCESS')

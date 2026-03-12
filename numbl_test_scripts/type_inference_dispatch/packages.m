% Test packages: hard scenarios - package functions calling other package
% functions, recursive nested package function, nested package calling
% parent, composing results from different levels

% --- Package function calling another package function internally ---
% calc.square now calls calc.add internally
assert(calc.square(7) == 49);
assert(calc.square(0) == 0);

% --- Package function calling two other package functions ---
% calc.hypotenuse calls calc.add and calc.square
assert(calc.hypotenuse(3, 4) == 5);
assert(calc.hypotenuse(5, 12) == 13);

% --- Recursive nested package function ---
assert(calc.advanced.power(2, 10) == 1024);
assert(calc.advanced.power(3, 3) == 27);
assert(calc.advanced.power(5, 0) == 1);
assert(calc.advanced.power(7, 1) == 7);

% --- Nested package calling parent and sibling package functions ---
% sum_powers(2, 3) = 2^0 + 2^1 + 2^2 + 2^3 = 15
assert(calc.advanced.sum_powers(2, 3) == 15);
assert(calc.advanced.sum_powers(3, 2) == 13, '1+3+9');

% --- Composing package results in expressions ---
r = calc.add(calc.square(3), calc.square(4));
% __inferred_type_str(r) would be "Number" with specialization enabled
assert(r == 25);

% --- Nested composition across levels ---
r2 = calc.hypotenuse(calc.advanced.power(3, 1), calc.advanced.power(4, 1));
assert(r2 == 5);

% --- Package function result in a loop ---
total = 0;
for i = 1:5
    assert(strcmp(__inferred_type_str(i), "Number"));
    total = calc.add(total, calc.square(i));
end
assert(total == 55, '1+4+9+16+25');

% --- Deeply nested: sum_powers uses power uses recursion ---
% sum_powers(2, 4) = 1+2+4+8+16 = 31
assert(calc.advanced.sum_powers(2, 4) == 31);

disp('SUCCESS')

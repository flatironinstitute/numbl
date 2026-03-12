% Test dispatch when multiple classes have the same method name
% The runtime should dispatch based on the class of the first argument

a = OpAdder_(10);
m = OpMultiplier_(10);
assert(strcmp(__inferred_type_str(a), 'ClassInstance<OpAdder_>'));
assert(strcmp(__inferred_type_str(m), 'ClassInstance<OpMultiplier_>'));

% Dot syntax - always correct
assert(a.apply_op(5) == 15, 'adder dot: 10 + 5');
assert(m.apply_op(5) == 50, 'mult dot: 10 * 5');

% Function-call syntax - dispatches on first arg type
assert(apply_op(a, 5) == 15, 'adder func-call dispatch');
assert(apply_op(m, 5) == 50, 'mult func-call dispatch');

% Describe returns different values per class
assert(describe_op(a) == 1, 'adder describe');
assert(describe_op(m) == 2, 'mult describe');
assert(a.describe_op() == 1);
assert(m.describe_op() == 2);

% Dispatch in a loop with mixed types
ops = {OpAdder_(5), OpMultiplier_(3), OpAdder_(1), OpMultiplier_(10)};
inputs = [10, 10, 10, 10];
expected = [15, 30, 11, 100];
for i = 1:4
    assert(strcmp(__inferred_type_str(i), "Number"));
    r = apply_op(ops{i}, inputs(i));
    assert(r == expected(i));
end

% Dispatch result used in expression
r = apply_op(a, 3) + apply_op(m, 2);
assert(r == 33, '(10+3) + (10*2) = 13 + 20 = 33');

disp('SUCCESS')

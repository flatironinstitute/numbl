% Test that class instances maintain proper dispatch when passed through
% multiple layers of function calls

% OpAdder_ and OpMultiplier_ defined in this directory
a = OpAdder_(10);
m = OpMultiplier_(10);
assert(strcmp(__inferred_type_str(a), 'ClassInstance<OpAdder_>'));
assert(strcmp(__inferred_type_str(m), 'ClassInstance<OpMultiplier_>'));

% --- Pass instance through a wrapper class ---
wa = Wrapper_(a, 'adder');
wm = Wrapper_(m, 'mult');
assert(strcmp(__inferred_type_str(wa), 'ClassInstance<Wrapper_>'));
assert(strcmp(__inferred_type_str(wm), 'ClassInstance<Wrapper_>'));

% Wrapper calls method on inner object via dot syntax
assert(wa.apply_inner(5) == 15, 'wrapper -> adder.apply_op');
assert(wm.apply_inner(5) == 50, 'wrapper -> mult.apply_op');

% Get inner object back and call method directly
inner_a = wa.get_inner();
assert(inner_a.apply_op(3) == 13, 'extracted inner still dispatches');

% --- Pass wrapper through external function ---
r1 = apply_to_wrapper(wa, 7);
assert(r1 == 17, 'external func -> wrapper -> adder');

r2 = apply_to_wrapper(wm, 7);
assert(r2 == 70, 'external func -> wrapper -> mult');

% --- Helper that creates and returns an instance ---
obj = create_and_use(20);
assert(obj == 25, 'helper creates OpAdder_ internally');

% --- Pass instances through local function ---
r3 = local_dispatch(a, m, 4);
assert(r3 == 54, '(10+4) + (10*4) = 14 + 40 = 54');

disp('SUCCESS')

function r = create_and_use(base)
    obj = OpAdder_(base);
    r = obj.apply_op(5);
end

function r = local_dispatch(op1, op2, val)
    r = op1.apply_op(val) + op2.apply_op(val);
end

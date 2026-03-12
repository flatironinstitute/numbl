% Test that paren-indexed assignment on a class instance with subsasgn
% routes through the overloaded subsasgn method.

% Test 1: obj(:, []) = empty should call subsasgn, not replace obj with empty
obj = ParenAsgn_(42);
assert(obj.data == 42, 'Initial data should be 42');
assert(obj.call_count == 0, 'Initial call_count should be 0');

obj(:, []) = [];
assert(obj.call_count == 1, 'subsasgn should have been called once');
assert(obj.data == 42, 'data should be unchanged after empty-index assignment');

% Test 2: obj(1) = 99 should call subsasgn and update data
obj2 = ParenAsgn_(0);
obj2(1) = 99;
assert(obj2.call_count == 1, 'subsasgn should have been called');
assert(obj2.data == 99, 'data should be updated to 99');

disp('SUCCESS')

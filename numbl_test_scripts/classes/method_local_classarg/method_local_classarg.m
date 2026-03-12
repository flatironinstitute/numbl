% Test: class method local helper function receiving class instance as first arg
% When a local function in a method file receives a class instance,
% it should still resolve to the local function (not try class method dispatch).

obj = LocalHelperB_(5);

% Call process indirectly through wrapper to get unknown types
result = obj.wrapper(2);
assert(result == 10, 'Expected 10');

obj2 = LocalHelperB_(7);
result2 = obj2.wrapper(2);
assert(result2 == 14, 'Expected 14');

disp('SUCCESS')

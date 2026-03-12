% Test that overloaded subsasgn works even when compile-time type is unknown.
% When the compiler can't infer that a variable is a class instance with
% subsasgn (e.g., retrieved from a cell array), the runtime should still
% route assignments through the overloaded subsasgn method.

% Erase the compile-time type by storing in a cell array
c = {PrefStore2_()};
obj = c{1};  % Type is Unknown at compile time

% Test 1: Single-level assignment through subsasgn
obj.alpha = 42;
assert(obj.alpha == 42, 'Single-level subsasgn should update alpha');

% Test 2: Verify other fields survived (subsasgn routes through prefList)
assert(obj.techPrefs.x == 1, 'techPrefs.x should be preserved after single-level subsasgn');
assert(obj.techPrefs.y == 2, 'techPrefs.y should be preserved after single-level subsasgn');

% Test 3: Unknown field goes to techPrefs via subsasgn
obj.newField = 99;
assert(obj.techPrefs.newField == 99, 'Unknown field should go to techPrefs via subsasgn');

disp('SUCCESS');

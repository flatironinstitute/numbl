% Test: inside a class method, obj(1) = other_obj (same class) should
% bypass subsasgn and do a direct object replacement (MATLAB classdef behavior).
% This avoids infinite recursion and matches what MATLAB does for class internals.

a = ParenAsgn_(10);
b = ParenAsgn_(20);

% Call from inside a class method
result = a.assignSameClassInside(b);

% subsasgn should NOT have been called (call_count stays 0)
assert(result.call_count == 0, 'subsasgn should not be called inside a class method for same-class obj(1)=val');
assert(result.data == 20, 'result should be b (data=20)');

% Sanity check: from outside the class, subsasgn IS called
c = ParenAsgn_(10);
c(1) = 99;
assert(c.call_count == 1, 'subsasgn should be called from outside the class');

disp('SUCCESS');

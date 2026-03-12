% Test: inside class methods, obj(k) and obj(k).prop use built-in
% behavior, not overloaded subsref.

o = SubsrefInternal_(7);

% External access: o(1) should go through overloaded subsref (doubles value)
assert(o(1) == 14);      % 7 * 2 = 14

% Internal access via method: obj(1).myFlag should use built-in behavior
assert(o.getFlag() == 7);  % direct property access, NOT 14

% Internal access via method: obj(1) should return obj itself
s = o.getSelf();
assert(s.getFlag() == 7);

disp('SUCCESS');

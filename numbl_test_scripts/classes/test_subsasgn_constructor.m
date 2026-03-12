% Test that class constructor can assign to properties directly even when
% subsasgn is overloaded (without special-casing the property name).
% This mirrors the chebfunpref pattern where the constructor does
% outPref.prefList = val and subsasgn does NOT handle 'prefList' specially.

% Test 1: Constructor should work without subsasgn interfering
obj = PrefStore2_();
assert(isstruct(obj.prefList), 'prefList should be a struct after construction');
assert(obj.alpha == 10, 'alpha should be 10 after construction');
assert(obj.techPrefs.x == 1, 'techPrefs.x should be 1 after construction');
assert(obj.techPrefs.y == 2, 'techPrefs.y should be 2 after construction');

% Test 2: External subsasgn should still work for known fields
obj.alpha = 42;
assert(obj.alpha == 42, 'External subsasgn should update alpha');

% Test 3: External subsasgn should route unknown fields to techPrefs
obj.newField = 99;
assert(obj.techPrefs.newField == 99, 'Unknown field should go to techPrefs');

disp('SUCCESS');

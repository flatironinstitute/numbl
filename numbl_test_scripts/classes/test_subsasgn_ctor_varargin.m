% Test that constructor property assignments work when the constructor
% output variable type is widened to Unknown by obj = varargin{1}.
% This mimics the chebfunpref pattern where the constructor has:
%   outPref = varargin{1}   (widens type from ClassInstance to Unknown)
%   outPref.prefList = val   (must assign declared property directly)

% Test 1: Basic construction (no copy)
obj1 = PrefStore3_();
assert(isstruct(obj1.prefList), 'prefList should be a struct after construction');
assert(obj1.alpha == 10, 'alpha should be 10 after construction');
assert(obj1.techPrefs.x == 1, 'techPrefs.x should be 1 after construction');
assert(obj1.techPrefs.y == 2, 'techPrefs.y should be 2 after construction');

% Test 2: Copy construction — triggers obj = varargin{1} path
obj2 = PrefStore3_(obj1);
assert(isstruct(obj2.prefList), 'prefList should be a struct after copy construction');
assert(obj2.alpha == 10, 'alpha should be 10 after copy construction');
assert(obj2.techPrefs.x == 1, 'techPrefs.x should be 1 after copy construction');

% Test 3: External subsasgn should still work for known fields
obj1.alpha = 42;
assert(obj1.alpha == 42, 'External subsasgn should update alpha');

% Test 4: External subsasgn should route unknown fields to techPrefs
obj1.newField = 99;
assert(obj1.techPrefs.newField == 99, 'Unknown field should go to techPrefs');

disp('SUCCESS');

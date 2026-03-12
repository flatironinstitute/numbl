% Test that obj.name() routes through subsref when the class
% has a custom subsref and 'name' is not a direct property/method.
% This mirrors the chebfunpref pattern: pref.tech() where tech
% is a field of an internal struct accessed via subsref.

obj = MyPrefClass(@sin);

% Test 1: dot-access should go through subsref
val = obj.tech;
assert(isa(val, 'function_handle'), 'dot-access should return function handle');

% Test 2: dot-access with parens (call) should go through subsref
% In MATLAB, obj.tech() routes through subsref with compound indices
% [{'.',  'tech'}, {'()', {}}]
% For a function handle, () calls the function
result = obj.tech(pi/2);
assert(abs(result - 1) < 1e-10, 'tech(pi/2) should call sin(pi/2) = 1');

% Test 3: dot-access with no args should also work through subsref
result2 = obj.tech(0);
assert(abs(result2) < 1e-10, 'tech(0) should call sin(0) = 0');

% Test 4: simple value access
val2 = obj.alpha;
assert(val2 == 42, 'alpha should be 42');

disp('SUCCESS');

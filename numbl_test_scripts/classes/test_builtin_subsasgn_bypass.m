% Test that builtin('subsasgn', obj, S, val) bypasses the overloaded
% subsasgn method and directly sets the property.  This mirrors the pattern
% used by chebfun's subsasgn.m whose dot-case calls:
%   f = builtin('subsasgn', f, index, val)
% where f is a chebfun instance.  Without a bypass, that call would
% re-enter the overloaded subsasgn and loop forever.

obj = BuiltinSubsasgnClass_();

% Setting 'data' should work through the overloaded subsasgn, which in
% turn calls builtin('subsasgn', ...) to perform the raw assignment.
obj.data = [1, 2, 3];
assert(isequal(obj.data, [1, 2, 3]), 'data should be set to [1,2,3]');

obj.domain = [0, 1];
assert(isequal(obj.domain, [0, 1]), 'domain should be updated to [0,1]');

% Overwrite again to confirm repeated writes work.
obj.data = [4, 5];
assert(isequal(obj.data, [4, 5]), 'data should be updated to [4,5]');

disp('SUCCESS');

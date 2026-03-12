% Test that local function resolution works correctly when multiple
% files in the same class have local functions with the same name.
%
% This mirrors the chebfun bug where:
% - @chebfun/chebfun.m has: function [op, dom, data, pref, flags] = parseInputs(...)
% - @chebfun/inv.m has:     function [tol, opts, pref] = parseInputs(...)
% The constructor should use ITS OWN local parseInputs (5 outputs),
% not the one from inv.m (3 outputs).
%
% The critical detail: inv.m creates a new MultiParser instance, which
% triggers constructor lowering while inv's withMethodScope is active.

% Test constructor - should use the 5-output parseInputs from MultiParser.m
obj = MultiParser(5, 3);
assert(obj.val1 == 5);   % a = op = 5
assert(obj.val2 == 3);   % b = extra = 3
assert(obj.val3 == 8);   % c = op + extra = 8
assert(obj.val4 == 10);  % d = op * 2 = 10
assert(obj.val5 == 6);   % e = extra * 2 = 6

% Test inv method - should use the 3-output parseInputs from inv.m
% inv creates a new MultiParser(tol, opts) = MultiParser(5, 2)
r = inv(obj);
assert(r.val1 == 5);
assert(r.val2 == 2);

disp('SUCCESS');

% Test that obj(k) = same_class_val inside a class method bypasses the
% overloaded subsasgn.  In MATLAB, subscript assignment inside a class method
% uses built-in array mechanics and does NOT dispatch to subsasgn.

% Test 1: known type — obj(1) = other inside method with typed parameter
a = ParenAsgnBypassSubsasgnClass_(5);
b = ParenAsgnBypassSubsasgnClass_(10);
a = a.replaceFirst(b);
assert(a.val == 10, 'val should be 10 after replaceFirst');

% Test 2: Unknown type — F is reassigned from a function call first, then
% F(k) = val inside a loop.  This mirrors the chebfun/restrict.m pattern.
c = ParenAsgnBypassSubsasgnClass_(3);
c = c.loopReplace();
assert(c.val == 6, 'val should be 6 (3*2) after loopReplace');

disp('SUCCESS')

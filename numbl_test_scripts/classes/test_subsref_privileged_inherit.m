% Test: privileged access to own properties inside subsref/subsasgn
% when the class inherits from a parent class.
% This mimics the chebfunpref pattern: chebfunpref < chebpref,
% where prefList is defined in the PARENT class, and the child
% has subsref/subsasgn that access obj.prefList internally.

% Test 1: basic construction and read
obj = PrefStoreChild_();
disp(obj.alpha);
assert(obj.alpha == 10, 'alpha should be 10');

% Test 2: external dot assignment (triggers subsasgn)
obj.alpha = 42;
assert(obj.alpha == 42, 'alpha should be 42 after assignment');

disp('SUCCESS');

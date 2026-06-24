% A class deriving from matlab.mixin.Copyable is a handle class (reference
% semantics). Its handle-style property setter (no output) must persist the
% mutation, for both static (obj.prop = v) and dynamic (obj.(name) = v) field
% assignment. Dynamic field reads must honor property getters too.

o = CopyablePose_();

% Static assignment routes through the converting setter.
o.r = [1 2 3];
assert(isequal(o.r, [1 2 3]));

% matlab.mixin.Copyable is a handle: aliases share state.
o2 = o;
o2.r = [4 5 6];
assert(isequal(o.r, [4 5 6]), 'expected handle reference semantics');

% Dynamic field assignment must also invoke the setter.
pname = 'r';
o.(pname) = [7 8 9];
assert(isequal(o.r, [7 8 9]));

% Dynamic field read must invoke the getter of a Dependent property.
gname = 'rsum';
assert(o.(gname) == 24);

disp('SUCCESS')

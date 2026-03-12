% Test that class properties without explicit defaults are initialized to []
% (empty double), not 0. This matches MATLAB semantics.

obj = PropDefaultTest_();
assert(isempty(obj.x), 'uninitialized property x should be empty []');
assert(isequal(obj.y, 5), 'initialized property y should be 5');
assert(isequal(class(obj.x), 'double'), 'default empty property should be double');

disp('SUCCESS')

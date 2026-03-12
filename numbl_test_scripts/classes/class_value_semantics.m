% Test that class instances have value semantics (not reference semantics)
% In MATLAB, value classes are copied when assigned or passed to functions.

% Test assignment creates independent copy
a = MyVal([1, 2, 3]);
b = a;
b.data(1) = 99;
assert(a.data(1) == 1, 'class instance should have value semantics on assignment');

% Test that method call returns modified copy without mutating original
c = MyVal([10, 20, 30]);
d = add(c, 5);
assert(c.data(1) == 10, 'class instance should not be mutated by method call');
assert(d.data(1) == 15, 'method should return modified copy');

% Test function argument not mutated
e = MyVal([1, 2, 3]);
mutate_obj(e);
assert(e.data(1) == 1, 'class instance passed to function should not be mutated');

disp('SUCCESS');

function mutate_obj(obj)
    obj.data(1) = 999;
end

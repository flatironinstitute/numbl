% Test that handle semantics propagate through inheritance chain
% HandleChild_ < Counter_ < handle

child = HandleChild_();
assert(child.Value == 0, 'inherited default property');
assert(strcmp(child.Name, 'default'), 'own default property');

% Inherited methods should work with handle semantics
child.increment(10);
assert(child.Value == 10, 'inherited method should mutate in place');

% Assignment should create a reference, not a copy
child2 = child;
child2.increment(5);
assert(child.Value == 15, 'handle semantics inherited: shared reference');

% Own methods should also have handle semantics
child.set_name('test');
assert(strcmp(child2.Name, 'test'), 'handle semantics: name change visible via both handles');

% Function should mutate the original
modify_child(child, 100);
assert(child.Value == 100, 'handle semantics through function call');
assert(child2.Value == 100, 'both handles see the function mutation');

fprintf('SUCCESS\n');

function modify_child(obj, val)
    obj.Value = val;
end

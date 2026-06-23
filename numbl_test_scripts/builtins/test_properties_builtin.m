% properties() returns a 0x1 cell for non-objects, and the public property
% names (Nx1 cell) for class objects.

p = properties(magic(3));
assert(iscell(p));
assert(isempty(p));
assert(size(p, 1) == 0 && size(p, 2) == 1);

assert(isempty(properties('not_a_class_name')));
assert(isempty(properties(struct('a', 1, 'b', 2))));
assert(isempty(properties({1, 2, 3})));
assert(isempty(properties(true)));

% A class object lists its public properties (containers.Map keeps its
% backing store in a private property, which must NOT be listed).
m = containers.Map('KeyType', 'char', 'ValueType', 'any');
pm = properties(m);
assert(numel(pm) == 3);
assert(ismember('Count', pm));
assert(ismember('KeyType', pm));
assert(ismember('ValueType', pm));
assert(~ismember('data_', pm));

disp('SUCCESS');

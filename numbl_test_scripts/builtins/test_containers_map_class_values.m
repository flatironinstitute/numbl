% Storing a class-instance value (e.g. a nested containers.Map) into a
% containers.Map must dispatch through subsasgn, not coerce the key to a
% number. Regression: a multi-char key with a class-instance value used to
% throw "Cannot convert multi-char to number".

% Single-char key (always worked) and multi-char key (regressed).
outer = containers.Map('KeyType', 'char', 'ValueType', 'any');
outer('k') = containers.Map('KeyType', 'char', 'ValueType', 'any');
outer('aabb_tree') = containers.Map('KeyType', 'char', 'ValueType', 'any');
assert(outer.Count == 2);

inner = outer('aabb_tree');
assert(isa(inner, 'containers.Map'));
inner('v1') = 10;
assert(inner('v1') == 10);

% Build a nested map structure via the isKey / assign pattern that real
% library code uses (this is what broke `mip load`).
nameMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
names = {'alpha', 'beta_two', 'gamma'};
for i = 1:numel(names)
    if ~nameMap.isKey(names{i})
        nameMap(names{i}) = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end
    vm = nameMap(names{i});
    vm('ver') = i;
    nameMap(names{i}) = vm;
end
assert(nameMap.Count == 3);
v = nameMap('beta_two');
assert(v('ver') == 2);

disp('SUCCESS');

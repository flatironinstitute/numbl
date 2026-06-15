% union / intersect / setdiff on cell arrays of character vectors (cellstr).

u = union({'b', 'a'}, {'c', 'a'});
assert(iscell(u) && numel(u) == 3, 'union cellstr size');
assert(isequal(u, {'a'; 'b'; 'c'}), 'union cellstr sorted column');

i = intersect({'a', 'b', 'c'}, {'b', 'c', 'd'});
assert(isequal(i, {'b'; 'c'}), 'intersect cellstr');

d = setdiff({'a', 'b', 'c'}, {'b'});
assert(isequal(d, {'a'; 'c'}), 'setdiff cellstr');

% Empty cell and char/cell mixing.
e = union({}, {'a'});
assert(isequal(e, {'a'}), 'union with empty cell');

% Index outputs.
[c, ia, ib] = union({'b', 'a'}, {'c'});
assert(isequal(c, {'a'; 'b'; 'c'}), 'union 3-output values');
assert(isequal(ia, [2; 1]) && isequal(ib, 1), 'union index outputs');

% Numeric path unchanged.
assert(isequal(union([3 1], [2 1]), [1 2 3]), 'numeric union still works');

disp('SUCCESS');

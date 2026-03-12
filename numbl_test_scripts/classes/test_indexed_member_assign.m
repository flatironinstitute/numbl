% Test F(j).field = rhs on class instances
% In MATLAB, V(j).prop = rhs uses built-in indexing (not subsref)
% for the () part, then sets the property directly.

% Simple value class with a property
obj = SimpleContainer();
obj.value = 10;
assert(obj.value == 10);

% Indexed member assignment: obj(1).value = rhs
% For a scalar object, obj(1) returns obj itself (built-in indexing).
obj(1).value = 42;
assert(obj.value == 42);

% Nested field: obj(1).data.x = rhs
obj(1).data.x = 7;
assert(obj.data.x == 7);

% Cell indexing through indexed member: obj(1).items{k} = rhs
% First set a property to force a new class instance (value class),
% then modify the cell through the same pattern. This mirrors the
% chebfun uminus pattern: F(j).pointValues = ...; F(j).funs{k} = ...
obj(1).value = 100;
obj(1).items{1} = 'hello';
obj(1).items{2} = 'world';
assert(obj.value == 100);
assert(strcmp(obj.items{1}, 'hello'));
assert(strcmp(obj.items{2}, 'world'));

% Test via class method (gets skipSubsref, mirrors chebfun uminus).
% Sharing the cell (by assigning to another var) triggers COW,
% exposing the bug where indexCellStore result isn't stored back.
obj2 = SimpleContainer();
obj2.value = 5;
obj2.items = {10, 20, 30};
backup = obj2.items;  % share the cell (_rc > 1)
obj2 = negateItems(obj2);
assert(obj2.value == -5);
assert(obj2.items{1} == -10);
assert(obj2.items{2} == -20);
assert(obj2.items{3} == -30);

disp('SUCCESS');
